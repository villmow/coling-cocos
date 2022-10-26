from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRPrecision,
)
from tqdm import tqdm

from cocos.data.cocos_datamodule import CCSZSDataModule
from cocos.model.retriever import load_model

datasets.enable_caching()
# datasets.disable_caching()
torch.multiprocessing.set_sharing_strategy("file_system")


def plot_embeddings(
    context_embeddings,
    target_embeddings,
    query_labels,
    target_labels,
    desc,
):
    model = TSNE(
        n_components=2,
        random_state=0,
        init="pca",
        metric="cosine",
        learning_rate="auto",
    )
    context_embeddings_2d = model.fit_transform(context_embeddings)

    model = TSNE(
        n_components=2,
        random_state=0,
        init="pca",
        metric="cosine",
        learning_rate="auto",
    )
    target_embeddings_2d = model.fit_transform(target_embeddings)
    df = pd.DataFrame(
        list(
            zip(
                [desc[l] for l in query_labels] + [desc[l] for l in target_labels],
                list(context_embeddings_2d[:, 0]) + list(target_embeddings_2d[:, 0]),
                list(context_embeddings_2d[:, 1]) + list(target_embeddings_2d[:, 1]),
                ["context"] * len(query_labels) + ["target"] * len(target_labels),
            ),
        ),
        columns=["description", "x", "y", "embedding"],
    )
    sns.relplot(data=df, x="x", y="y", hue="description", col="embedding")
    plt.show()


def compute_metrics(sims, q_labels, t_labels, q_query_ids, t_query_ids):
    num_queries, num_targets = sims.shape
    # print("num_queries",num_queries)
    # print("num_targets", num_targets)
    indexes = torch.arange(num_queries)[:, None].repeat(1, num_targets)

    # print("indexes.shape", indexes.shape)
    # print("indexes", indexes)

    target = q_labels[:, None] == t_labels[None, :]
    # print("target.shape", target.shape)
    # print("target", target)
    # print("target.sum(dim=-1)", set((target).sum(dim=-1).tolist()))
    # print("target.sum(dim=-1)", (target).sum(dim=-1))

    mask = ~(q_query_ids[:, None] == t_query_ids[None, :])
    # print("mask.shape", mask.shape)
    # print("mask", mask)
    # print("mask.sum(dim=-1)", (~mask).sum(dim=-1))

    i = indexes[mask]
    p = sims[mask]
    t = target[mask]
    m = RetrievalMAP()

    rmap = round(m(p, t, indexes=i).item() * 100, 2)
    # print("Number of queries:", len(q_labels))
    print("MAP:", rmap)

    m = RetrievalNormalizedDCG()
    rndcg = round(m(p, t, indexes=i).item() * 100, 2)
    # print("NDCG:", rndcg)

    m = RetrievalRPrecision()
    rrprec = round(m(p, t, indexes=i).item() * 100, 2)
    # print("R-Precision:", rrprec)

    rprecs = []
    for k in [1, 3, 10]:
        m = RetrievalPrecision(top_k=k)
        rprec = round(m(p, t, indexes=i).item() * 100, 2)
        # print(f"P@{k}:", rprec)
        rprecs.append(rprec)

    return {
        "MAP": rmap,
        "NDCG": rndcg,
        "R-Precision": rrprec,
        "P@1": rprecs[0],
        "P@3": rprecs[1],
        "P@10": rprecs[2],
    }


def evaluate(
    model_path: Path,
    max_distraction_snippets=10000,
    device="cuda:6",
):
    # load dataloaders
    dm = CCSZSDataModule(
        "../../dataset/cocos.jsonl",
        "../../dataset/distractors.jsonl",
        batch_size=8,
        max_length=None,
        max_distraction_snippets=max_distraction_snippets,
    )
    dm.setup()
    query_dataloader, search_index_dataloader = dm.val_dataloader()

    # load model and move to gpu
    model = load_model(
        checkpoint_path=model_path,
        config_path=model_path.parent / "config.yaml",
    )

    model.to(device)
    model.eval()

    # embed queries and targets
    q_embeddings, q_labels, q_query_ids = [], [], []
    t_embeddings, t_labels, t_query_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(query_dataloader, "Embedding queries"):
            embeddings = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )

            q_embeddings.append(embeddings.cpu())
            q_labels.append(batch["labels"])
            q_query_ids.append(batch["query_ids"])

        for batch in tqdm(search_index_dataloader, "Embedding targets"):
            embeddings = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            )

            t_embeddings.append(embeddings.cpu())
            t_labels.append(batch["labels"])
            t_query_ids.append(batch["query_ids"])

    del model

    q_embeddings = torch.cat(q_embeddings, dim=0)
    q_labels = torch.cat(q_labels, dim=0)
    q_query_ids = torch.cat(q_query_ids, dim=0)

    t_embeddings = torch.cat(t_embeddings, dim=0)
    t_labels = torch.cat(t_labels, dim=0)
    t_query_ids = torch.cat(t_query_ids, dim=0)

    q_embeddings = torch.nn.functional.normalize(q_embeddings)
    t_embeddings = torch.nn.functional.normalize(t_embeddings)

    sims = (q_embeddings.to(device) @ t_embeddings.t().to(device)).cpu()

    desc = dm.id2description
    desc[0] = "distraction"
    # plot_embeddings(q_embeddings, t_embeddings, q_labels, t_labels, desc)
    metrics = compute_metrics(sims, q_labels, t_labels, q_query_ids, t_query_ids)
    return q_embeddings, t_embeddings, q_labels, t_labels, desc, metrics


def print_table(model_metrics):
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Model Metrics")
    table.add_column("Model Name", style="bold")
    table.add_column("MAP", style="bold")
    table.add_column("NDCG", style="bold")
    table.add_column("R-Precision", style="bold")
    table.add_column("P@1", style="bold")
    table.add_column("P@3", style="bold")
    table.add_column("P@10", style="bold")

    for model_name, row in model_metrics.items():
        table.add_row(
            model_name,
            str(row["MAP"]),
            str(row["NDCG"]),
            str(row["R-Precision"]),
            str(row["P@1"]),
            str(row["P@3"]),
            str(row["P@10"]),
        )

    console = Console()
    console.print(table)


def evaluate_single(model_path):
    evaluate(
        Path(model_path),
        max_distraction_snippets=10000,
        larger_context=False,
        dedent_target=False,
        strip_leading_whitespace=True,
        device="cpu",
    )


def evaluate_all(models: dict[str, Path]):
    results = {}
    for model_name, model_path in models.items():
        print(f"Evaluating {model_name}")
        q_embeddings, t_embeddings, q_labels, t_labels, desc, metrics = evaluate(
            model_path,
            max_distraction_snippets=10000,
            larger_context=False,
            dedent_target=False,
            strip_leading_whitespace=True,
            device="cpu",
        )
        results[model_name] = metrics

        print(f"Results for {model_name}")
        print(results[model_name])
    print(results)
    print_table(results)


if __name__ == "__main__":
    models = {
        "TS,IM,DE": Path(
            "../../checkpoints/paper_models/ts_im_de/epoch=0-step=412499.ckpt"
        ),
        "TS,IM": Path("../../checkpoints/paper_models/ts_im/epoch=0-step=397499.ckpt"),
        "TS,DE": Path("../../checkpoints/paper_models/ts_de/epoch=0-step=389999.ckpt"),
        "TS": Path("../../checkpoints/paper_models/ts/epoch=0-step=397499.ckpt"),
        "None": Path("../../checkpoints/paper_models/none/epoch=0-step=389999.ckpt"),
    }

    #
    evaluate_all(models)
    evaluate_single(models["TS,IM,DE"])
