import logging
import traceback
from typing import Iterator, Iterable, Union, Optional, Callable, Any

import numpy as np
import torch
from torch.utils.data import IterDataPipe, functional_datapipe

from cocos import tensortree
from cocos.datapipes.batching import PairBucketedBatch
from cocos.datapipes.common import SeedMixin
from cocos.source_code import (
    CodeTree,
    truncate_code,
    TruncateCodeConfig,
    CodePair,
    CodeTreePair,
    Code,
    ExtractedCodeTree,
    mask_single_span_cfg,
    MaskSpanConfig,
    mask_random,
    mask_random_spans,
    mask_trees_in_sample,
    calc_quantiles,
    mask_single_span_no_tree_cfg,
    dedent_target,
    mask_identifiers,
    unique_identifiers_in_pair,
    UniqueVariablesConfig,
)
from cocos import tokenizer


log = logging.getLogger(__name__)


class CallableDatapipe(IterDataPipe):
    def __init__(self, datapipe: Iterable):
        super(CallableDatapipe, self).__init__()
        self.datapipe = datapipe

    def __call__(self, sample: Any) -> Optional[Any]:
        """Should define the operation done to the sample."""
        raise NotImplementedError("Should be implemented")

    def __iter__(self) -> Iterator[Any]:
        for sample in self.datapipe:
            if sample is None:
                log.debug(f"[{self.__class__.__name__}] Sample is None")
                continue
            try:
                new_sample = self(sample)
                if new_sample is not None:
                    yield new_sample
                else:
                    log.debug(f"[{self.__class__.__name__}] sample is None. Skipping.")

            except Exception as e:
                log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
                log.error(str(sample))
                log.error(traceback.format_exc())


class NoOpDataPipe(CallableDatapipe):
    def __call__(self, sample: Any) -> Optional[Any]:
        return sample


@functional_datapipe("code_truncate")
class TruncateCodeIterDataPipe(IterDataPipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable,
        cfg: TruncateCodeConfig,
        replacement_tokens: Union[
            list[int], torch.Tensor
        ] = tokenizer.get_fold_token_ids(),
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.datapipe = datapipe
        self.cfg = cfg
        self.replacement_tokens = replacement_tokens
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[CodeTree]:
        for sample in self.datapipe:
            yield from truncate_code(
                sample, self.generator, self.cfg, self.replacement_tokens
            )


@functional_datapipe("code_mask_span")
class CodeSpanMaskerIterDataPipe(SeedMixin, CallableDatapipe):
    def __init__(
        self,
        datapipe: Iterable[Union[dict, CodeTree]],
        cfg: MaskSpanConfig,
        mask_token: torch.Tensor = tokenizer.get_mask_id(),
        seed: Optional[int] = None,
    ):
        super().__init__(datapipe)
        self.mask_token = torch.tensor(mask_token)
        self.cfg = cfg
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample = CodeTree.from_dict(sample)

        if not isinstance(sample, CodeTree):
            raise ValueError(
                f"Sample should be a CodeTree object and not {type(sample)}  {id(sample)}"
            )

        sample = mask_single_span_cfg(
            sample, mask_token=self.mask_token, seed=self.generator, cfg=self.cfg
        )
        return sample


class TokenSpanMaskerIterDataPipe(CodeSpanMaskerIterDataPipe):
    """
    Masks a single span no tree.
    """

    def __init__(
        self,
        datapipe: Iterable[Union[dict, CodeTree]],
        cfg: MaskSpanConfig,
        mask_token: torch.Tensor = tokenizer.get_mask_id(),
        seed: Optional[int] = None,
    ):
        super().__init__(datapipe, cfg, mask_token, seed)

    def __call__(self, sample) -> CodePair:
        if isinstance(sample, dict):
            sample = CodeTree.from_dict(sample)

        if not isinstance(sample, CodeTree):
            raise ValueError(
                f"Sample should be a CodeTree object and not {type(sample)}  {id(sample)}"
            )

        sample = mask_single_span_no_tree_cfg(
            sample, mask_token=self.mask_token, seed=self.generator, cfg=self.cfg
        )
        return sample


class LeavesIterDataPipe(CallableDatapipe):
    def __init__(
        self,
        datapipe: Iterable[Union[CodeTreePair, CodeTree]],
        ignore_non_trees: bool = False,
        prepend_mask_symbol_to_target: bool = True,
    ):
        super().__init__(datapipe)
        self.ignore_non_trees = ignore_non_trees
        self.prepend_mask_symbol_to_target = prepend_mask_symbol_to_target

    def __call__(self, sample):
        if isinstance(sample, ExtractedCodeTree):
            tokens = sample.tree.leaves()

            # prepend masked token
            if self.prepend_mask_symbol_to_target:
                tokens = torch.cat([sample.replacement.view(1), tokens])

            return Code(data=tokens, meta=sample.meta)
        elif isinstance(sample, CodeTree):
            return Code(data=sample.tree.leaves(), meta=sample.meta)
        elif isinstance(sample, CodeTreePair):
            source_tokens = sample.source.tree.leaves()
            target_tokens = sample.target.tree.leaves()

            if self.prepend_mask_symbol_to_target and isinstance(
                sample.target, ExtractedCodeTree
            ):
                print("prepend2")
                target_tokens = torch.cat(
                    [sample.target.replacement.view(1), target_tokens]
                )

            return CodePair(
                source=Code(data=source_tokens, meta=sample.source.meta),
                target=Code(data=target_tokens, meta=sample.target.meta),
            )
        elif self.ignore_non_trees:
            return sample

        raise ValueError(
            f"Sample should be either CodeTree or CodeTreePair and not {sample.__class__.__name__}"
        )


@functional_datapipe("code_dedent")
class CodeDedenterIterDataPipe(CallableDatapipe):
    def __init__(self, datapipe: Iterable[CodeTreePair]):
        super().__init__(datapipe)

    def __call__(self, sample: CodePair):
        if isinstance(sample, CodePair):
            return dedent_target(sample)

        raise ValueError("Class unknown")


@functional_datapipe("code_mask_variables")
class CodeVariableMaskerIterDataPipe(CallableDatapipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        identifier_nonterminal: Union[
            int, torch.Tensor
        ] = tokenizer.get_identifier_nonterminal(),
        replacement_tokens: torch.Tensor = tokenizer.get_mask_token_ids(),
        hide_prob: float = 1,
        reveal_prob: float = 0,
        return_target: bool = True,
        return_trees: bool = False,
        seed: Optional[int] = None,
        sample_hide_prob: bool = False,
        sample_prob_mu: float = 0.6,
        sample_prob_sigma: float = 0.25,
    ):
        super().__init__(datapipe)

        self.identifier_nonterminal = identifier_nonterminal
        self.replacement_tokens = replacement_tokens
        self.hide_prob = hide_prob
        self.reveal_prob = reveal_prob
        self.return_target = return_target
        self.return_trees = return_trees
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

        self.dist = None
        self.sample_hide_prob = sample_hide_prob  # then hide_prob is ignored
        if sample_hide_prob:
            import scipy.stats as stats

            lower, upper = 0.1, 1  # bounds in which numbers will be sampled
            mu = sample_prob_mu
            sigma = sample_prob_sigma
            self.dist = stats.truncnorm(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
            )

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)
        if getattr(self, "dist", None) is not None:
            self.dist.random_state = self.generator

    def get_hide_prob(self):
        if self.dist is not None:
            return self.dist.rvs()
        else:
            return self.hide_prob

    def __call__(self, sample: CodeTree):
        return mask_identifiers(
            sample=sample,
            identifier_nonterminal=self.identifier_nonterminal,
            replacement_tokens=self.replacement_tokens,
            seed=self.generator,
            hide_prob=self.get_hide_prob(),
            reveal_prob=self.reveal_prob,
            return_target=self.return_target,
            return_trees=self.return_trees,
        )


@functional_datapipe("random_mask")
class RandomMaskerIterDataPipe(CallableDatapipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        replacement_tokens: torch.Tensor = tokenizer.get_mask_token_ids(),
        mask_prob: float = 0.15,
        seed: Optional[int] = None,
    ):
        super().__init__(datapipe)

        self.replacement_tokens = replacement_tokens
        self.mask_prob = mask_prob
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample: Union[Code, CodeTree]) -> Optional[CodePair]:
        return mask_random(
            sample,
            replacement_tokens=self.replacement_tokens,
            seed=self.generator,
            mask_prob=self.mask_prob,
        )


@functional_datapipe("random_span_mask")
class RandomSpanMaskerIterDataPipe(CallableDatapipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        replacement_tokens: torch.Tensor = tokenizer.get_mask_token_ids(),
        mean_span_length: int = 3,
        mask_prob: float = 0.15,
        seed: Optional[int] = None,
    ):
        super().__init__(datapipe)

        self.replacement_tokens = replacement_tokens
        self.mean_span_length = mean_span_length
        self.mask_prob = mask_prob
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample: Union[Code, CodeTree]) -> Optional[CodePair]:
        return mask_random_spans(
            sample,
            replacement_tokens=self.replacement_tokens,
            seed=self.generator,
            mask_prob=self.mask_prob,
            mean_span_length=self.mean_span_length,
        )


@functional_datapipe("subtree_mask")
class SubtreeMaskerIterDataPipe(CallableDatapipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        mask_prob: float,
        mean_span_length: int,
        sample_mode: str,
        seed: Optional[int] = None,
        replacement_tokens: torch.Tensor = tokenizer.get_mask_token_ids(),
    ):
        super().__init__(datapipe)

        self.replacement_tokens = replacement_tokens
        self.mean_span_length = mean_span_length
        self.mask_prob = mask_prob
        self.sample_mode = sample_mode
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    @property
    def mean_span_length(self):
        return self._mean_span_length

    @mean_span_length.setter
    def mean_span_length(self, value):
        self._mean_span_length = value
        self.lower_span_bound, self.upper_span_bound = calc_quantiles(
            self.mean_span_length, 0.75
        )

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample: CodeTree) -> Optional[CodePair]:
        return mask_trees_in_sample(
            sample,
            replacement_tokens=self.replacement_tokens,
            seed=self.generator,
            mask_prob=self.mask_prob,
            lower_span_bound=self.lower_span_bound,
            upper_span_bound=self.upper_span_bound,
            sample_mode=self.sample_mode,
        )


@functional_datapipe("code_unique_variables")
class CodePairUniqueVariableMaskerIterDataPipe(CallableDatapipe, SeedMixin):
    def __init__(
        self,
        datapipe: Iterable[CodeTreePair],
        cfg: UniqueVariablesConfig,
        identifier_nonterminal: Union[
            int, torch.Tensor
        ] = tokenizer.get_identifier_nonterminal(),
        replacement_tokens: torch.Tensor = tokenizer.get_mask_token_ids()[1:],
        return_trees: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(datapipe)
        self.cfg = cfg
        self.identifier_nonterminal = identifier_nonterminal
        self.replacement_tokens = replacement_tokens
        self.return_trees = return_trees
        self.generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample: CodeTreePair) -> Optional[Union[CodePair, CodeTreePair]]:
        return unique_identifiers_in_pair(
            sample=sample,
            identifier_nonterminal=self.identifier_nonterminal,
            replacement_tokens=self.replacement_tokens,
            seed=self.generator,
            cfg=self.cfg,
            return_trees=self.return_trees,
        )


class SampleModifierBase(CallableDatapipe):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        *,
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(datapipe)
        self.modify_source = modify_source
        self.modify_target = modify_target

    def modify_code_sample(self, sample: Code) -> Code:
        raise NotImplementedError("implement this")

    def modify_code_tree_sample(self, sample: CodeTree) -> CodeTree:
        raise NotImplementedError("implement this")

    def __call__(
        self, sample: Union[Code, CodeTree, CodePair, CodeTreePair]
    ) -> Union[Code, CodeTree, CodePair, CodeTreePair]:
        if isinstance(sample, CodeTree):
            new_sample = self.modify_code_tree_sample(sample)
        elif isinstance(sample, Code):
            new_sample = self.modify_code_sample(sample)
        elif isinstance(sample, CodeTreePair):
            new_sample = CodeTreePair(
                source=self.modify_code_tree_sample(sample.source)
                if self.modify_source
                else sample.source,
                target=self.modify_code_tree_sample(sample.target)
                if self.modify_target
                else sample.target,
            )
        elif isinstance(sample, CodePair):
            new_sample = CodePair(
                source=self.modify_code_sample(sample.source)
                if self.modify_source
                else sample.source,
                target=self.modify_code_sample(sample.target)
                if self.modify_target
                else sample.target,
            )
        else:
            raise ValueError("Sample class unknown")

        return new_sample


@functional_datapipe("add_token")
class TokenAdderIterDataPipe(SampleModifierBase):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        token: Union[int, torch.Tensor],
        prepend: bool,  # default is append
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe, modify_source=modify_source, modify_target=modify_target
        )
        self.token = (
            torch.tensor([token])
            if (not isinstance(token, torch.Tensor) or token.ndim == 0)
            else token
        )
        self.prepend = prepend

    def modify_code_sample(self, sample: Code) -> Code:
        func = prepend_token_to_tensor if self.prepend else append_token_to_tensor
        return Code(func(sample.data, self.token), meta=sample.meta)

    def modify_code_tree_sample(self, sample: CodeTree) -> CodeTree:
        func = prepend_token_to_tree if self.prepend else append_token_to_tree
        return CodeTree(func(sample.tree, self.token), meta=sample.meta)


@functional_datapipe("prepend")
class TokenPrependerIterDataPipe(TokenAdderIterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        token: Union[int, torch.Tensor],
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe,
            token,
            prepend=True,
            modify_source=modify_source,
            modify_target=modify_target,
        )


@functional_datapipe("append")
class TokenAppenderIterDataPipe(TokenAdderIterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        token: Union[int, torch.Tensor],
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe,
            token,
            prepend=False,
            modify_source=modify_source,
            modify_target=modify_target,
        )


@functional_datapipe("add_eos")
class EosAppenderIterDataPipe(TokenAppenderIterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        token: Union[int, torch.Tensor] = tokenizer.get_eos_id(),
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe, token, modify_source=modify_source, modify_target=modify_target
        )


@functional_datapipe("add_cls")
class ClsPrependerIterDataPipe(TokenPrependerIterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        token: Union[int, torch.Tensor] = tokenizer.get_cls_id(),
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe, token, modify_source=modify_source, modify_target=modify_target
        )


@functional_datapipe("add_language_id")
class LanguageIdAdderIterDataPipe(SampleModifierBase):
    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        prepend: bool = True,
        modify_source: bool = True,
        modify_target: bool = True,
    ):
        super().__init__(
            datapipe, modify_source=modify_source, modify_target=modify_target
        )
        self.prepend = prepend
        self.language_to_id = {
            tokenizer.get_language(lang): lang_id.reshape(1)
            for lang, lang_id in zip(
                tokenizer.language_tokens(), tokenizer.get_language_token_ids()
            )
        }

    def get_token_for_lang(self, language: str):
        return self.language_to_id[language]

    def modify_code_sample(self, sample: Code) -> Code:
        func = prepend_token_to_tensor if self.prepend else append_token_to_tensor
        return Code(
            func(sample.data, self.get_token_for_lang(sample.meta.language)),
            meta=sample.meta,
        )

    def modify_code_tree_sample(self, sample: CodeTree) -> CodeTree:
        func = prepend_token_to_tree if self.prepend else append_token_to_tree
        return CodeTree(
            func(sample.tree, self.get_token_for_lang(sample.meta.language)),
            meta=sample.meta,
        )


@functional_datapipe("truncate_tokens")
class TokenTruncaterIterDataPipe(SampleModifierBase):
    """Hard truncates samples"""

    def __init__(
        self,
        datapipe: Iterable[CodeTree],
        max_length: int,
        substract: int = 0,
        modify_source: bool = True,
        modify_target: bool = True,
        check_specials: bool = True,
    ):
        super().__init__(
            datapipe, modify_source=modify_source, modify_target=modify_target
        )
        self.max_length = max_length - substract

        if check_specials:
            self.specials = tokenizer.get_mask_token_ids().numpy()
        else:
            self.specials = None

    def modify_code_sample(self, sample: Code) -> Code:
        truncated_tokens = sample.data[: self.max_length]
        return Code(truncated_tokens, meta=sample.meta)

    def modify_code_tree_sample(self, sample: CodeTree) -> CodeTree:
        raise NotImplementedError

    def remove_specials_from_target_not_in_source(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # print("-" * 100)
        # print("SOURCE", TOK.decode(source))
        # print("TARGET", TOK.decode(target))
        specials_in_source = self.specials[np.isin(self.specials, source.numpy())]
        target_specials_mask = np.isin(target.numpy(), self.specials)
        target_specials_to_keep_mask = np.isin(
            target[target_specials_mask], specials_in_source
        )

        num_specials_to_keep = target_specials_to_keep_mask.sum()
        if num_specials_to_keep == 0:
            # truncation removed mask token from source, which is bad.
            raise ValueError("truncation removed mask token from source, which is bad.")
        elif num_specials_to_keep == 1:
            return target  # do nothing

        splitted_target = target.tensor_split(
            torch.from_numpy(target_specials_mask).nonzero().squeeze()
        )
        if splitted_target[0].numel() == 0:
            splitted_target = splitted_target[1:]
        elif splitted_target[-1].numel() == 0:
            splitted_target = splitted_target[:-1]
            raise NotImplementedError("Check this case!")
        else:
            assert len(splitted_target) == len(target_specials_to_keep_mask)
            raise NotImplementedError("Check this case!")

        new_target = [
            segment
            for i, segment in enumerate(splitted_target)
            if target_specials_to_keep_mask[i]
        ]
        new_target = torch.cat(new_target)
        # print("NEW TARGET", TOK.decode(new_target))

        return new_target

    def __call__(
        self, sample: Union[Code, CodeTree, CodePair, CodeTreePair]
    ) -> Union[Code, CodeTree, CodePair, CodeTreePair]:
        if isinstance(sample, CodeTree):
            new_sample = self.modify_code_tree_sample(sample)
        elif isinstance(sample, Code):
            new_sample = self.modify_code_sample(sample)
        elif isinstance(sample, CodeTreePair):
            raise NotImplementedError
        elif isinstance(sample, CodePair):
            modified_source = (
                self.modify_code_sample(sample.source)
                if self.modify_source
                else sample.source
            )
            modified_target = (
                self.modify_code_sample(sample.target)
                if self.modify_target
                else sample.target
            )

            if self.specials is not None:
                try:
                    modified_target.data = (
                        self.remove_specials_from_target_not_in_source(
                            modified_source.data, modified_target.data
                        )
                    )
                except ValueError:
                    return  # dont use this sample if we removed the mask token from source!

            new_sample = CodePair(
                source=modified_source,
                target=modified_target,
            )
        else:
            raise ValueError("Sample class unknown")

        return new_sample


def append_token_to_tree(
    tree: tensortree.TensorTree, token: Union[int, torch.Tensor]
) -> tensortree.TensorTree:
    assert tree.node_data[-1] != token, "token is already last in tree."
    # fixme node_data is missing
    return tree.insert_child(
        parent_idx=0, node_data=token
    )  # add EOS as last child of root


def prepend_token_to_tree(
    tree: tensortree.TensorTree, token: Union[int, torch.Tensor]
) -> tensortree.TensorTree:
    assert tree.leaves()[0] != token, "token is already first leaf in tree."
    # fixme node_data is missing
    return tree.insert_child(
        parent_idx=0, node_data=token, right_sibling_idx=1
    )  # add EOS as first child of root


def append_token_to_tensor(
    sequence: torch.Tensor, token: Union[int, torch.Tensor]
) -> torch.Tensor:
    assert sequence[-1] != token, "token is already last."
    if not isinstance(token, torch.Tensor):
        token = sequence.new_tensor([token])
    return torch.cat([sequence, token])


def prepend_token_to_tensor(
    sequence: torch.Tensor, token: Union[int, torch.Tensor]
) -> torch.Tensor:
    assert sequence[0] != token, "token is already first."
    if not isinstance(token, torch.Tensor):
        token = sequence.new_tensor([token])
    return torch.cat([token, sequence])


def get_lang(sample) -> str:
    if isinstance(sample, Code):
        lang = sample.meta.language
    elif isinstance(sample, CodePair):
        lang = sample.source.meta.language
    elif isinstance(sample, dict):
        lang = sample["language"]
    else:
        raise ValueError("Sample class unknown")

    return lang


from cocos.source_code.identifiers import LANGUAGES


class LanguageBatcherIterDataPipe(IterDataPipe):
    """Groups batches by language. See cocos.datapipes.batching"""

    def __init__(
        self,
        datapipe: IterDataPipe,
        collate_fn: Optional[Callable],
        *,
        max_tokens_in_batch: int,
        max_sequence_length: int,
        num_buckets: int,
        size_fn: Callable = len,
        languages: set[str] = LANGUAGES,
    ):
        super().__init__()

        from torch.utils.data.dataloader import default_collate

        self.datapipe = datapipe
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.size_fn = size_fn

        self.buckets = {
            lang: PairBucketedBatch(
                max_tokens_in_batch=max_tokens_in_batch,
                max_sequence_length=max_sequence_length,
                num_buckets=num_buckets,
            )
            for lang in set(languages)
        }

    def __iter__(self):
        try:
            for sample in self.datapipe:
                lang = get_lang(sample)
                bucket = self.buckets[lang]
                batch = bucket.add_sample(
                    sample, sample.size
                )  # returns a full batch if ready

                if batch is not None:
                    yield self.collate_fn(batch)

            # yield unfinalized batches
            for bucket in self.buckets.values():
                for batch in bucket.get_batches():
                    yield self.collate_fn(batch)

        except Exception as e:
            log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
            import traceback

            log.error(traceback.format_exc())


class DispatcherIterDataPipe(CallableDatapipe, SeedMixin):
    """
    Dispatches samples randomly to other datapipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        probs: list[float],
        list_of_datapipes: list[CallableDatapipe],
        seed: Optional[int] = None,
        try_again_on_error: bool = True,
    ):
        super().__init__(datapipe)

        self.probs = np.array(list(probs))
        if self.probs.sum() != 1:
            self.probs /= self.probs.sum()
            log.warning(
                f"Normalizing probs to have sum of 1. Normalized probs: {self.probs} "
            )
        self.try_again_on_error = try_again_on_error

        from omegaconf import DictConfig

        if isinstance(list_of_datapipes, DictConfig):
            list_of_datapipes = list(list_of_datapipes.values())

        self.list_of_datapipes = list_of_datapipes
        for datapipe in self.list_of_datapipes:
            assert isinstance(
                datapipe, CallableDatapipe
            ), "datapipes should be callable"
        self.number_of_datapipes = len(self.list_of_datapipes)

        self.generator: np.random.Generator = None  # will be set with seed
        self.seed = seed  # can be set by worker init function

    def on_seed_change(self, seed):
        self.generator = np.random.default_rng(seed)

    def __call__(self, sample, probs=None):
        choice = self.generator.choice(
            self.number_of_datapipes,
            p=(probs if probs is not None else self.probs),
            replace=False,
        )
        try:
            new_sample = self.list_of_datapipes[choice](sample)
        except Exception as e:
            new_sample = None

        if new_sample is not None or not self.try_again_on_error:
            return new_sample

        # this datapipe failed or produced none so we deactivate it using the probs
        probs = self.probs.copy()
        probs[choice] = 0

        if probs.sum() == 0:
            return

        probs /= probs.sum()
        return self(sample, probs)
