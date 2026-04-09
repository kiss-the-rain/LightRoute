"""Chunk-level dense OCR retrieval backed by a local BGE encoder."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retrieval.ocr_bge_retriever import OCRBGERetriever, format_bge_query


class OCRBGEChunkRetriever:
    """Document-scoped chunk retriever that reuses the OCR BGE encoder."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.encoder = OCRBGERetriever(cfg)
        self.index: dict[str, dict[str, Any]] = {}

    def build_document_index(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        """Encode and cache one document's OCR chunks once."""
        if doc_id in self.index:
            return
        self.encoder._ensure_model()
        chunk_texts = [str(chunk.get("chunk_text", "")) for chunk in chunks]
        embeddings = self.encoder._encode_texts(chunk_texts) if chunk_texts else np.zeros((0, 1), dtype=float)
        self.index[doc_id] = {
            "chunks": [dict(chunk) for chunk in chunks],
            "embeddings": embeddings,
        }

    def retrieve(
        self,
        query: str,
        doc_id: str,
        topk: int,
        query_variant: str = "raw_question",
    ) -> dict[str, list]:
        """Retrieve top-k OCR chunks within one document."""
        if doc_id not in self.index:
            return {"chunk_ids": [], "page_ids": [], "scores": [], "ranks": [], "chunks": []}
        self.encoder._ensure_model()
        query_text = format_bge_query(query, variant=query_variant)
        query_embedding = self.encoder._encode_texts([query_text])[0]
        chunk_embeddings = self.index[doc_id]["embeddings"]
        scores = chunk_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:topk]
        ranked_chunks = [self.index[doc_id]["chunks"][idx] for idx in ranked_indices]
        return {
            "chunk_ids": [str(chunk["chunk_id"]) for chunk in ranked_chunks],
            "page_ids": [int(chunk["page_id"]) for chunk in ranked_chunks],
            "scores": [float(scores[idx]) for idx in ranked_indices],
            "ranks": list(range(1, len(ranked_indices) + 1)),
            "chunks": ranked_chunks,
        }
