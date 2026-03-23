"""ColPali-style retriever wrapper with lightweight surrogate embeddings for Phase 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.visual_retriever import VisualRetriever
from utils.io_utils import load_pickle, save_pickle
from utils.rank_utils import topk_from_scores
from utils.text_utils import tokenize_text


def _stable_token_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class ColPaliRetriever(VisualRetriever):
    """Phase 1 visual retriever using deterministic hashed surrogate embeddings."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.embedding_dim = int(cfg.visual_retriever.embedding_dim)
        self.cache_path = Path(cfg.visual_retriever.cache_path)
        self.index: dict[str, dict[str, Any]] = {}

    def build_index(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build and optionally cache page embeddings by document."""
        if self.cfg.runtime.use_cache and self.cache_path.exists():
            self.index = load_pickle(self.cache_path)
            return

        self.index = {}
        for doc_id, pages in documents.items():
            embeddings = self.encode_pages(pages)
            self.index[doc_id] = {
                "page_ids": [page["page_id"] for page in pages],
                "page_items": pages,
                "embeddings": embeddings,
            }

        if self.cfg.visual_retriever.cache_embeddings:
            save_pickle(self.index, self.cache_path)

    def encode_pages(self, pages: list[dict[str, Any]]) -> list[list[float]]:
        """Encode page surrogate text or image identifiers into normalized vectors."""
        return [self._text_to_embedding(self._page_surrogate_text(page)) for page in pages]

    def encode_query(self, question: str) -> list[float]:
        """Encode the query into the same hashed vector space."""
        return self._text_to_embedding(question)

    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k pages inside one document."""
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}

        query_embedding = np.asarray(self.encode_query(question), dtype=float)
        page_embeddings = np.asarray(self.index[doc_id]["embeddings"], dtype=float)
        scores = page_embeddings @ query_embedding
        return topk_from_scores(self.index[doc_id]["page_ids"], scores.tolist(), topk)

    def _page_surrogate_text(self, page: dict[str, Any]) -> str:
        if self.cfg.visual_retriever.use_surrogate_text:
            text = page.get("surrogate_text") or page.get("ocr_text") or page.get("image_path", "")
            if text:
                return str(text)
        return Path(page.get("image_path", page.get("page_id", ""))).stem.replace("_", " ")

    def _text_to_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.embedding_dim, dtype=float)
        for token in tokenize_text(text):
            vector[_stable_token_index(token, self.embedding_dim)] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()
