"""OCR-only BM25 retriever for document-internal page retrieval."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

from src.retrieval.text_retriever import TextRetriever
from src.utils.io_utils import load_json, load_pickle, save_pickle


def load_page_ocr_text(ocr_path: str) -> str:
    """Load one OCR JSON and robustly extract page-level text."""
    try:
        payload = load_json(ocr_path)
    except Exception:
        return ""

    tokens = _extract_text_units(payload)
    cleaned_tokens = [token.strip().lower() for token in tokens if isinstance(token, str) and token.strip()]
    return " ".join(cleaned_tokens)


def _extract_text_units(payload: Any) -> list[str]:
    """Recursively extract text-like units from heterogeneous OCR JSON structures."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        texts: list[str] = []
        for item in payload:
            texts.extend(_extract_text_units(item))
        return texts
    if isinstance(payload, dict):
        texts: list[str] = []
        preferred_keys = ("tokens", "words", "text", "blocks", "lines")
        for key in preferred_keys:
            if key in payload:
                texts.extend(_extract_text_units(payload[key]))
        if texts:
            return texts
        for value in payload.values():
            texts.extend(_extract_text_units(value))
        return texts
    return []


class BM25Retriever(TextRetriever):
    """BM25 retriever that supports both standalone and per-document indexed retrieval."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg
        self.cache_path = Path(cfg.text_retriever.cache_path) if cfg is not None else None
        self.index: dict[str, dict[str, Any]] = {}
        self.page_ids: list[int] = []
        self.page_texts: list[str] = []
        self.tokenized_pages: list[list[str]] = []
        self.bm25: Any | None = None

    def build_index(self, page_texts: list[str] | dict[str, list[dict[str, Any]]], page_ids: list[int] | None = None) -> None:
        """Build a BM25 index either for one document or for a document dictionary."""
        if isinstance(page_texts, dict):
            self._build_document_indexes(page_texts)
            return

        self.page_texts = page_texts
        self.page_ids = page_ids or list(range(len(page_texts)))
        self.tokenized_pages = [self._tokenize(text) for text in page_texts]
        self.bm25 = self._build_bm25(self.tokenized_pages)

    def retrieve(self, query: str, topk: int = 10, doc_id: str | None = None) -> dict[str, list]:
        """Retrieve top-k pages either from the current document index or a stored doc index."""
        if doc_id is not None:
            if doc_id not in self.index:
                return {"page_ids": [], "scores": [], "ranks": []}
            tokenized_query = self._tokenize(query)
            bm25 = self.index[doc_id]["bm25"]
            scores = bm25.get_scores(tokenized_query).tolist()
            return _topk_from_scores(self.index[doc_id]["page_ids"], scores, topk)

        if self.bm25 is None:
            return {"page_ids": [], "scores": [], "ranks": []}
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query).tolist()
        return _topk_from_scores(self.page_ids, scores, topk)

    def _build_document_indexes(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build cached BM25 indexes for a dict of documents."""
        if self.cfg is not None and self.cfg.runtime.use_cache and self.cache_path is not None and self.cache_path.exists():
            self.index = load_pickle(self.cache_path)
            return

        self.index = {}
        for doc_id, pages in documents.items():
            page_ids = [self._page_id_to_int(page["page_id"]) for page in pages]
            page_texts = [str(page.get("ocr_text", "")) for page in pages]
            tokenized_pages = [self._tokenize(text) for text in page_texts]
            self.index[doc_id] = {
                "page_ids": page_ids,
                "page_texts": page_texts,
                "bm25": self._build_bm25(tokenized_pages),
            }

        if self.cfg is not None and self.cfg.text_retriever.cache_index and self.cache_path is not None:
            save_pickle(self.index, self.cache_path)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text with lowercase and whitespace split."""
        return str(text).lower().split()

    def _build_bm25(self, corpus: list[list[str]]):
        k1 = float(self.cfg.text_retriever.k1) if self.cfg is not None else 1.5
        b = float(self.cfg.text_retriever.b) if self.cfg is not None else 0.75
        if not any(corpus):
            return _SimpleBM25(corpus, k1=k1, b=b)
        if BM25Okapi is not None:
            return BM25Okapi(corpus, k1=k1, b=b)
        return _SimpleBM25(corpus, k1=k1, b=b)

    @staticmethod
    def _page_id_to_int(page_id: str | int) -> int:
        """Convert page ids like `docid_p80` into integer page numbers."""
        if isinstance(page_id, int):
            return page_id
        if "_p" in page_id:
            return int(page_id.rsplit("_p", maxsplit=1)[-1])
        return int(page_id)


def _topk_from_scores(page_ids: list[int], scores: list[float], topk: int) -> dict[str, list]:
    """Sort BM25 scores and return the top-k predictions."""
    ranked_items = sorted(zip(page_ids, scores), key=lambda item: item[1], reverse=True)[:topk]
    return {
        "page_ids": [int(page_id) for page_id, _ in ranked_items],
        "scores": [float(score) for _, score in ranked_items],
        "ranks": list(range(1, len(ranked_items) + 1)),
    }


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
        return np.asarray(scores, dtype=float)
