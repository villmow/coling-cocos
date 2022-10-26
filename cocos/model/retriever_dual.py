import logging
from pathlib import Path
import torch
from typing import Union, Optional

from omegaconf import DictConfig

from torch import nn
from transformers.activations import ACT2FN

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import AdamW, T5Config, T5EncoderModel

from cocos import tokenizer
from cocos.source_code import BatchedCodePairs, BatchedCode
from cocos.model.retriever import Retriever, load_model


log = logging.getLogger(__file__)


def load_encoder(
    checkpoint_path,
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[DictConfig] = None,
) -> T5EncoderModel:
    model = load_model(checkpoint_path, config_path, config)

    if isinstance(model, Retriever):
        return model.model

    raise ValueError("Model type not supported")


class ProjectionLayer(torch.nn.Module):
    """
    Similar to the expander layer in VICReg / Barlow Twins (output_size=8192, activation_function=relu),
    but with LayerNorm instead of BatchNormalization.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_fn: Optional[str],
        do_layer_norm: bool = True,
    ):
        super(ProjectionLayer, self).__init__()

        self.proj = nn.Linear(input_size, output_size, bias=False)
        self.norm = nn.LayerNorm(output_size) if do_layer_norm else None
        self.activation = ACT2FN[activation_fn] if activation_fn is not None else None

        self.input_size = input_size
        self.output_size = output_size

        self._init_weights()

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def _init_weights(self):
        self.proj.weight.data.normal_(mean=0.0, std=(self.input_size**-0.5))
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            self.proj.bias.data.zero_()
        if self.norm is not None:
            self.norm.weight.data.fill_(1.0)


class Projector(torch.nn.Module):
    """
    Similar to the expander/projector in VICReg / Barlow Twins (output_size=8192, activation_function=relu),
    but with LayerNorm instead of BatchNormalization.
    """

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        output_size: int,
        activation_fn: Optional[str],
        do_layer_norm: bool = True,
        layer_norm_in_last_layer: bool = False,
    ):
        super(Projector, self).__init__()

        self.layers = nn.ModuleList()
        for layer in range(num_layers - 1):
            self.layers.append(
                ProjectionLayer(
                    input_size if layer == 0 else output_size,
                    output_size,
                    activation_fn,
                    do_layer_norm=do_layer_norm,
                )
            )
        # last layer no activation and layernorm
        self.layers.append(
            ProjectionLayer(
                output_size,
                output_size,
                activation_fn=None,
                do_layer_norm=layer_norm_in_last_layer,
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DualRetriever(Retriever):
    def __init__(
        self,
        loss,
        optimizer_cfg,
        model_cfg,
        pretrained_encoder_checkpoint: Optional[str] = None,
        share_encoders: bool = True,
        query_projector_cfg: Optional = None,
        key_projector_cfg: Optional = None,
    ):
        super().__init__(loss, optimizer_cfg)
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer.load_tokenizer()

        self.pad_id = tokenizer.get_pad_id()
        config = T5Config(
            # model relevant settings
            **model_cfg,
            vocab_size=len(self.tokenizer),
            pad_token_id=tokenizer.get_pad_id(),
            eos_token_id=tokenizer.get_eos_id(),
            decoder_start_token_id=tokenizer.get_eos_id(),
        )

        self.share_encoders = share_encoders

        if pretrained_encoder_checkpoint is not None:
            try:
                self.query_model = T5EncoderModel.from_pretrained(
                    pretrained_encoder_checkpoint
                )
            except Exception as e:
                from cocos.utils import get_project_root

                log.info(
                    f"Could not load pretrained encoder from {pretrained_encoder_checkpoint}, trying to load from {get_project_root() / pretrained_encoder_checkpoint}"
                )
                pretrained_encoder_checkpoint = (
                    get_project_root() / pretrained_encoder_checkpoint
                )
                self.query_model = T5EncoderModel.from_pretrained(
                    pretrained_encoder_checkpoint
                )

            self.key_model = (
                self.query_model
                if share_encoders
                else T5EncoderModel.from_pretrained(pretrained_encoder_checkpoint)
            )
        else:
            self.query_model = T5EncoderModel(config)
            self.key_model = (
                self.query_model if share_encoders else T5EncoderModel(config)
            )

        if query_projector_cfg is not None:
            self.query_projector = Projector(
                num_layers=query_projector_cfg.num_layers,
                input_size=self.query_model.config.d_model,
                output_size=query_projector_cfg.projection_size,
                activation_fn=query_projector_cfg.activation_fn,
                layer_norm_in_last_layer=query_projector_cfg.layer_norm_in_last_layer,
            )
        else:
            self.query_projector = None

        if key_projector_cfg is not None:
            self.key_projector = Projector(
                num_layers=key_projector_cfg.num_layers,
                input_size=self.key_model.config.d_model,
                output_size=key_projector_cfg.projection_size,
                activation_fn=key_projector_cfg.activation_fn,
                layer_norm_in_last_layer=key_projector_cfg.layer_norm_in_last_layer,
            )
        else:
            self.key_projector = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        assert (
            self.query_model == self.key_model
        ), "Use forward_query or forward_key when using different encoders"
        assert (
            self.query_projector == self.key_projector
        ), "Use forward_query or forward_key when using different projectors"
        return self.get_embedding(self.query_model, *args, **kwargs)

    def get_embedding(
        self,
        model: T5EncoderModel,
        *args,
        projector: Optional[Projector] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Additionally projects if projector is defined."""
        encodings: BaseModelOutputWithPastAndCrossAttentions = model(*args, **kwargs)

        x = encodings.last_hidden_state  # [B, S, D]
        # project before selecting cls token embedding
        if projector is not None:
            x = projector(x)  # [B, S, D]

        embedding = x[:, 0]  # [B, D]
        return embedding

    def get_representation(
        self, model: T5EncoderModel, *args, **kwargs
    ) -> torch.Tensor:
        return self.get_embedding(model, *args, *kwargs, projector=None)

    def forward_query(
        self, batch: Union[BatchedCode, BatchedCodePairs, dict]
    ) -> torch.Tensor:
        if isinstance(batch, BatchedCodePairs):
            return self.forward_query(batch.source)

        if isinstance(batch, BatchedCode):
            return self.get_embedding(
                self.query_model,
                projector=self.query_projector,
                input_ids=batch.tokens,
                attention_mask=batch.attention_mask,
            )
        elif isinstance(batch, dict):
            # pass everything in batch dict to model
            return self.get_embedding(
                self.query_model, projector=self.query_projector, **batch
            )
        raise ValueError("unknown batch type")

    def forward_key(
        self, batch: Union[BatchedCode, BatchedCodePairs, dict]
    ) -> torch.Tensor:
        if isinstance(batch, BatchedCodePairs):
            return self.forward_key(batch.target)

        if isinstance(batch, BatchedCode):
            return self.get_embedding(
                self.key_model,
                projector=self.key_projector,
                input_ids=batch.tokens,
                attention_mask=batch.attention_mask,
            )
        elif isinstance(batch, dict):
            # pass everything in batch dict to model
            return self.get_embedding(
                self.key_model, projector=self.key_projector, **batch
            )

        raise ValueError("unknown batch type")

    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        self.upgrade_state_dict(state_dict)

        return super(DualRetriever, self).load_state_dict(state_dict)

    def upgrade_state_dict(self, state_dict: "OrderedDict[str, Tensor]"):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
        """
        for param in list(state_dict.keys()):
            if param.startswith("model"):
                assert (
                    self.query_model == self.key_model
                ), "query and key model should be the same in order to work with old checkpoint"
                assert (
                    self.share_encoders
                ), "query and key model should be the same in order to work with old checkpoint"
                query_param = "query_" + param
                key_param = "key_" + param
                state_dict[query_param] = state_dict[param]
                state_dict[key_param] = state_dict[param]
                del state_dict[param]
