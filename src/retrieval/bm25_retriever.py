"""BM25 page-level OCR retrieval."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

from retrieval.text_retriever import TextRetriever
from utils.io_utils import load_pickle, save_pickle
from utils.rank_utils import topk_from_scores
from utils.text_utils import tokenize_text


class BM25Retriever(TextRetriever):
    """Document-scoped BM25 retriever over OCR page text."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.cache_path = Path(cfg.text_retriever.cache_path)
        self.index: dict[str, dict[str, Any]] = {}

    def build_index(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build and cache BM25 indexes per document."""
        if self.cfg.runtime.use_cache and self.cache_path.exists():
            self.index = load_pickle(self.cache_path)
            return

        self.index = {}
        for doc_id, pages in documents.items():
            corpus = [tokenize_text(page.get("ocr_text", "")) for page in pages]
            self.index[doc_id] = {
                "page_ids": [page["page_id"] for page in pages],
                "page_items": pages,
                "tokenized_pages": corpus,
                "bm25": self._build_bm25(corpus),
            }

        if self.cfg.text_retriever.cache_index:
            save_pickle(self.index, self.cache_path)

    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve OCR text pages with BM25 scores."""
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}

        tokenized_question = tokenize_text(question)
        bm25 = self.index[doc_id]["bm25"]
        scores = bm25.get_scores(tokenized_question).tolist()
        return topk_from_scores(self.index[doc_id]["page_ids"], scores, topk)

    def _build_bm25(self, corpus: list[list[str]]):
        if BM25Okapi is not None:
            return BM25Okapi(corpus, k1=float(self.cfg.text_retriever.k1), b=float(self.cfg.text_retriever.b))
        return _SimpleBM25(corpus, k1=float(self.cfg.text_retriever.k1), b=float(self.cfg.text_retriever.b))


class _SimpleBM25:
    """Minimal BM25 implementation used when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        self.term_doc_freqs: list[dict[str, int]] = []
        self.doc_freqs: dict[str, int] = {}
        for doc in corpus:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.term_doc_freqs.append(freq)
            for token in freq:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]):
        scores = []
        for doc_index, freqs in enumerate(self.term_doc_freqs):
            doc_len = self.doc_lengths[doc_index]
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl > 0 else tf + self.k1
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        import numpy as np

        return np.asarray(scores, dtype=float)
