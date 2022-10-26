import datetime
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
from cocos.source_code import CodeMeta
import html

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from cocos import tokenizer
from dataclasses import dataclass, field

log = logging.getLogger(__file__)


@dataclass
class SearchMetrics:
    """gimme ranks i give you metrics"""

    ranks: torch.Tensor

    num_samples: int = field(init=False)
    hits_at_1: float = field(init=False)
    hits_at_3: float = field(init=False)
    hits_at_5: float = field(init=False)
    hits_at_10: float = field(init=False)
    hits_at_20: float = field(init=False)
    hits_at_100: float = field(init=False)
    mrr: float = field(init=False)
    mr: float = field(init=False)

    def __len__(self):
        return self.num_samples

    def __post_init__(self):
        self.num_samples = self.ranks.size(0)
        self.hits_at_1 = hits_at(self.ranks, 1)
        self.hits_at_3 = hits_at(self.ranks, 3)
        self.hits_at_5 = hits_at(self.ranks, 5)
        self.hits_at_10 = hits_at(self.ranks, 10)
        self.hits_at_20 = hits_at(self.ranks, 20)
        self.hits_at_100 = hits_at(self.ranks, 100)
        self.mrr = mrr(self.ranks)
        self.mr = mr(self.ranks)

    def as_markdown_table(self):
        import datetime

        time_ = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        return f"""| Date | Samples | MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 | Hits@20 | Hits@100 | Mean Rank |
|------|--------:|----:|-------:|-------:|-------:|--------:|--------:|---------:|----------:|
|{time_}| {len(self)} | {self.mrr:.03f} | {self.hits_at_1:.03f} | {self.hits_at_3:.03f} | {self.hits_at_5:.03f} | {self.hits_at_10:.03f} | {self.hits_at_20:.03f} | {self.hits_at_100:.03f} | {self.mr:.1f} |"""


def hits_at(ranks: torch.Tensor, k: int):
    return (ranks <= k).float().mean().item()


def mrr(ranks: torch.Tensor):
    return ranks.reciprocal().mean().item()


def mr(ranks: torch.Tensor):
    return ranks.float().mean().item()


def pairwise_cosine(query_embeddings, key_embeddings):
    query_embeddings = torch.nn.functional.normalize(query_embeddings)
    key_embeddings = torch.nn.functional.normalize(key_embeddings)

    scores = query_embeddings @ key_embeddings.t()
    indices = torch.argsort(scores, descending=True)
    ranks = (indices == torch.arange(indices.size(0))[:, None]).nonzero()[:, 1]
    ranks += 1
    search_metrics = SearchMetrics(ranks)
    return scores, indices, search_metrics


def pairwise_distance(query_embeddings, key_embeddings, p=2):
    scores = torch.cdist(query_embeddings, key_embeddings, p=p)
    indices = torch.argsort(scores, descending=False)
    ranks = (indices == torch.arange(indices.size(0))[:, None]).nonzero()[:, 1]
    ranks += 1
    search_metrics = SearchMetrics(ranks)
    return scores, indices, search_metrics


from cocos.utils import language_for_pygments


def get_answer_row(rank, score, meta: CodeMeta, code: str, is_correct: bool = False):
    return f"""<!-- A message -->
                    <div class="flex items-start mb-4 text-sm p-2 rounded-lg {"bg-green-light" if is_correct else ""}">
                        <div class="flex-1 overflow-hidden">
                            <div>
                                <h2 class="">{rank}. Result ({score:.02f})</h2>
                            </div>
                            <div class="text-sm rounded mt-4 overflow-auto">
                                <pre><code class="language-{language_for_pygments(meta.language)}">{html.escape(code)}</code></pre>
                            </div>
                            <div class="flex items-center space-x-4 mt-4 m-2">
                                <div class="flex-1 min-w-0">
                                    <a  href="{meta.repo_url}" class="text-sm font-medium text-gray-900 truncate dark:text-white  hover:text-gray-500">{meta.repository}</a>
                                    <a  href="{meta.file_url}"  class="  text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900">{meta.path_from_root}</a>
                                </div>
                            </div>
                        </div>
                    </div>"""


def get_question_row(question_meta: CodeMeta, question_code: str, answers: list[tuple]):
    html_str = f"""
        <div class="w-full px-4 h-1/2 flex font-mono antialiased  bg-white gap-2">
        <div class="w-1/2 flex flex-col overflow-x-scroll text-sm ">
            <div class="flex items-center space-x-4 m-2">
                <div class="flex-shrink-0">
                    <img class="w-8 h-8 rounded-full" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="Neil image">
                </div>
                <div class="flex-1 min-w-0">
                    <a  href="{question_meta.repo_url}" class="text-sm font-medium text-gray-900 truncate dark:text-white  hover:text-gray-500">
                        <p>{question_meta.repository}</p>
                    </a>
                    <a  href="{question_meta.file_url}"  class="text-sm text-gray-500 truncate dark:text-gray-400 hover:text-gray-900">
                        <p>{question_meta.path_from_root}</p>
                    </a>
                </div>
                <div class="inline-flex items-center rounded-full bg-blue-darker p-2 font-semibold text-white dark:text-white">
                    {question_meta.language}
                </div>
            </div>
            <div class="flex mt-4">
                <pre><code class="language-{language_for_pygments(question_meta.language)}">{html.escape(question_code)}</code></pre>
            </div>
        </div>
        <!-- Chat content -->
        <div class="flex-1 flex flex-col overflow-hidden">
                <!-- Top bar -->
                <div class="border-b flex px-6 py-2 items-center flex-none">
                    <div class="flex flex-col">
                        <h3 class="text-grey-darkest mb-1 font-extrabold">Results</h3>
                        <div class="text-grey-dark text-sm truncate">
                            Showing the first 10 results
                        </div>
                    </div>
                </div>
            <div class="flex-1 overflow-y-scroll">

            <!-- Chat messages -->
                <div class="px-6 py-4 flex-1 divide-y divide-solid">
                """

    for rank, score, meta, code, is_correct in answers:
        html_str += get_answer_row(rank, score, meta, code, is_correct)

    html_str += """'
                </div>
            </div>
        </div>
    </div>
"""
    return html_str


@dataclass
class SearchResult:
    queries: list[torch.Tensor]
    queries_meta: list[CodeMeta]
    keys: list[torch.Tensor]
    keys_meta: list[CodeMeta]

    scores: torch.Tensor
    indices: torch.Tensor
    ranks: torch.Tensor

    query_str: list[str] = field(init=False)
    key_str: list[str] = field(init=False)

    def __post_init__(self):
        from cocos.tokenizer import load_tokenizer, format_masks

        tokenizer = load_tokenizer()
        self.query_str = [
            format_masks(tokenizer.decode(query), surround_spaces=False)
            for query in self.queries
        ]
        self.key_str = [
            format_masks(tokenizer.decode(key), surround_spaces=False)
            for key in self.keys
        ]

    def search_answers(self, i: int, top_k: int = 10):
        question_meta = self.queries_meta[i]
        question_code = self.query_str[i]
        correct_result_idx = self.ranks[i]
        answers = []

        ranks = list(range(top_k))
        if (correct_result_idx - 1) > top_k:
            ranks.append(correct_result_idx - 1)

        for position in range(top_k):
            rank = position + 1
            result_idx = self.indices[i, position]
            score = self.scores[i, result_idx]

            meta = self.keys_meta[result_idx]
            code = self.key_str[result_idx]
            is_correct = rank == correct_result_idx

            answers.append((rank, score, meta, code, is_correct))

        return get_question_row(question_meta, question_code, answers)

    def format_result(self, i: int, top_k: int = 10):
        from cocos.utils import language_for_pygments

        def format_meta_as_table(meta):
            table = (
                f'<table class="tg"><thead><tr><th class="tg-1wig">Language</th><th class="tg-1wig">Repository</th>'
                f'<th class="tg-1wig">Filepath</th></tr></thead><tbody><tr><td class="tg-0pky">{meta.language}</td>'
                f'<td class="tg-btxf"><a href="{meta.repo_url}" target="_blank" rel="noopener noreferrer">'
                f'{meta.repository}</a></td><td class="tg-0lax"><a href="{meta.file_url}" target="_blank" '
                f'rel="noopener noreferrer">{meta.path}</a></td></tr></tbody></table>'
            )
            return table

        def format_code(code, language):
            return f'<pre><code class="language-{language_for_pygments(language)}">{html.escape(code)}</code></pre>'

        query_language = self.queries_meta[i].language
        lines = [
            f"<hr>",
            f"<h2> Query {i}</h2>",
            format_meta_as_table(self.queries_meta[i]),
            format_code(self.query_str[i], query_language),
        ]
        correct_result_idx = self.ranks[i]

        def build_result_line(rank, score, key_idx):
            nonlocal lines
            lines.append('<div class="result-row">')
            lines.append("<hr>")
            lines.append(f"<h3>Result {rank} ({score:.03f})</h3>")

            if rank == correct_result_idx:
                lines.append(f"<b>Correct</b>")

            lines += [
                f"<p>Score: {score:.03f}</p>",
                format_meta_as_table(self.keys_meta[key_idx]),
                format_code(self.key_str[key_idx], self.keys_meta[key_idx].language),
            ]
            lines.append("</div>")

        for position in range(top_k):
            rank = position + 1
            result_idx = self.indices[i, position]
            score = self.scores[i, result_idx]

            build_result_line(rank, score=score, key_idx=result_idx)

        if (correct_result_idx - 1) > top_k:
            lines.append("<hr> ... <hr>")
            result_idx = self.indices[i, correct_result_idx - 1]
            score = self.scores[i, result_idx]
            build_result_line(correct_result_idx, score=score, key_idx=i)

        return '<div class="">' + "\n".join(lines) + "</div>"

    def as_html(
        self, top_k: int = 10, split_n: Optional[int] = None
    ) -> Union[str, list[str]]:
        if split_n is None:
            return self._create_html(start=0, end=len(self.queries), top_k=top_k)
        else:
            return [
                self._create_html(start=i, end=(i + split_n), top_k=top_k)
                for i in range(0, len(self.queries), split_n)
            ]

    def _create_html(self, start: int, end: int, top_k: int = 10) -> str:
        html_str = """
        <!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://unpkg.com/tailwindcss@0.3.0/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/styles/dark.min.css"
          rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/highlight.min.js"></script>
    <script src="https://cdn.tailwindcss.com/3.0.14"></script>
    <script>hljs.highlightAll();</script>
</head>
<body class="h-screen overflow-hidden flex flex-col items-center overflow-y-scroll gap-16 bg-white divide-y">
"""
        end = min(end, len(self.queries))
        html_str += "\n".join(self.search_answers(i, top_k) for i in range(start, end))
        html_str += """</body></html>"""
        return html_str


class SearchPairsCallback(pl.Callback):
    def __init__(
        self,
        max_samples: int = 100000,
        print_top_k_results: int = 10,
        split_n: Optional[int] = 500,
        log_directory: Optional[str] = None,
        distance: str = "cosine",
    ):
        super(SearchPairsCallback, self).__init__()
        self.tokenizer = tokenizer.load_tokenizer()
        self.pad_id = tokenizer.get_pad_id()

        self.max_samples = max_samples
        self.print_top_k_results = print_top_k_results
        self.split_n = split_n

        from pathlib import Path

        self.log_directory = Path(log_directory) if log_directory is not None else None
        if self.log_directory:
            self.log_directory.mkdir(exist_ok=True)

        self.current_outputs = []
        self.num_current_samples = 0
        self.distance = distance

    def reset(self):
        self.current_outputs = []
        self.num_current_samples = 0

    def log_results(
        self,
        search_result: SearchResult,
        search_metrics: SearchMetrics,
        trainer: pl.Trainer,
    ):
        html_pages = search_result.as_html(
            top_k=self.print_top_k_results, split_n=self.split_n
        )

        if self.log_directory is not None:
            filepath = self.log_directory / f"rank{trainer.global_rank}.metrics"
            filepath.unlink(missing_ok=True)
            with filepath.open("w") as f:
                print(search_metrics.as_markdown_table(), file=f)

            for i, html_str in enumerate(html_pages):
                filename = f"rank{trainer.global_rank}-part{i:03d}.html"
                filepath = self.log_directory / filename
                filepath.unlink(missing_ok=True)

                with filepath.open("w") as f:
                    print(html_str, file=f)

                log.info(f"Saved results under: {filepath.absolute()}")

    def log_metrics(
        self, pl_module: pl.LightningModule, search_metrics: SearchMetrics, split: str
    ):
        pl_module.log(
            f"{split}/num_samples",
            search_metrics.num_samples,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/mr",
            search_metrics.mr,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@100",
            search_metrics.hits_at_100,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@20",
            search_metrics.hits_at_20,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@10",
            search_metrics.hits_at_10,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@5",
            search_metrics.hits_at_5,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@3",
            search_metrics.hits_at_3,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/hits@1",
            search_metrics.hits_at_1,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{split}/mrr",
            search_metrics.mrr,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if outputs is None:
            log.info(f"Can't search without no outputs from validation_step:")
            return

        # if trainer.is_global_zero and self.num_current_samples < self.max_samples:
        if self.num_current_samples < self.max_samples:
            self.current_outputs.append(outputs)
            self.num_current_samples += len(outputs["queries"])

    def search_nn(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: list[dict]
    ):
        """Performs neares neighbor searh on CPU"""
        queries, keys = [], []
        queries_meta, keys_meta = [], []
        query_embeddings, key_embeddings = [], []

        for batch_out in outputs:
            query_embeddings.append(batch_out["query_embeddings"])
            key_embeddings.append(batch_out["key_embeddings"])
            queries += batch_out["queries"]
            keys += batch_out["keys"]
            queries_meta += batch_out["queries_meta"]
            keys_meta += batch_out["keys_meta"]

        query_embeddings = torch.cat(query_embeddings, dim=0)
        key_embeddings = torch.cat(key_embeddings, dim=0)

        log.info(
            f"[Rank {trainer.global_rank}] Searching with {len(query_embeddings)} queries and {len(key_embeddings)} targets."
        )

        try:
            assert len(query_embeddings) == len(key_embeddings)
        except AssertionError as e:
            print(e)
            print("len(query_embeddings)", len(query_embeddings))
            print("len(key_embeddings)", len(key_embeddings))
            print("len(queries)", len(queries))
            print("len(keys)", len(keys))
            print("len(queries_meta)", len(queries_meta))
            print("len(keys_meta)", len(keys_meta))
            raise e

        if self.distance == "cosine":
            scores, indices, search_metrics = pairwise_cosine(
                query_embeddings, key_embeddings
            )
        elif self.distance.lower() == "l1" or self.distance.lower() == "manhattan":
            scores, indices, search_metrics = pairwise_distance(
                query_embeddings, key_embeddings, p=1
            )
        elif self.distance.lower() == "l2" or self.distance.lower() == "euclidean":
            scores, indices, search_metrics = pairwise_distance(
                query_embeddings, key_embeddings, p=2
            )
        else:
            raise ValueError(
                f"distance {self.distance} unknown. Choose from cosine, l1, l2"
            )

        self.log_metrics(pl_module, search_metrics, "valid")

        if len(query_embeddings) == len(queries) == len(keys):
            search_result = SearchResult(
                queries,
                queries_meta,
                keys,
                keys_meta,
                scores,
                indices,
                search_metrics.ranks,
            )
            self.log_results(search_result, search_metrics, trainer)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.search_nn(trainer, pl_module, self.current_outputs)
        self.reset()
