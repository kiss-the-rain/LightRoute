"""Hybrid OCR retrieval that combines BM25 pages with BGE chunk retrieval and reranking."""

from __future__ import annotations

from typing import Any

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.ocr_bge_chunk_retriever import OCRBGEChunkRetriever
from src.retrieval.ocr_bge_chunk_reranker import OCRBGEChunkReranker
from src.retrieval.ocr_chunker import aggregate_chunk_scores_to_page


class OCRHybridRetriever:
    """Combine BM25 page retrieval with chunk-level BGE retrieval for one document."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.bm25_indexes: dict[str, BM25Retriever] = {}
        self.chunk_retriever = OCRBGEChunkRetriever(cfg)
        self.chunk_reranker = OCRBGEChunkReranker(cfg)

    def retrieve(
        self,
        question: str,
        doc_id: str,
        page_texts: list[str],
        page_ids: list[int],
        chunks: list[dict[str, Any]],
        topk: int,
        coarse_chunk_topn: int,
        bm25_topn: int,
        aggregation_strategy: str = "max",
        query_variant: str = "instruction_query",
    ) -> dict[str, list]:
        """Run a union-style hybrid retrieval within one document."""
        if doc_id not in self.bm25_indexes:
            bm25 = BM25Retriever()
            bm25.build_index(page_texts=page_texts, page_ids=page_ids)
            self.bm25_indexes[doc_id] = bm25
        bm25_result = self.bm25_indexes[doc_id].retrieve(question, topk=bm25_topn)
        self.chunk_retriever.build_document_index(doc_id=doc_id, chunks=chunks)
        coarse_chunk_result = self.chunk_retriever.retrieve(
            question,
            doc_id=doc_id,
            topk=coarse_chunk_topn,
            query_variant=query_variant,
        )
        reranked_chunk_result = self.chunk_reranker.rerank(
            question,
            chunk_candidates=coarse_chunk_result["chunks"],
            topk=coarse_chunk_topn,
            query_variant=query_variant,
        )

        bm25_pages = {int(page_id): float(score) for page_id, score in zip(bm25_result["page_ids"], bm25_result["scores"])}
        reranked_page_result = aggregate_chunk_scores_to_page(
            [
                {"page_id": page_id, "score": score}
                for page_id, score in zip(reranked_chunk_result["page_ids"], reranked_chunk_result["scores"])
            ],
            strategy=aggregation_strategy,
            topk=max(topk, coarse_chunk_topn),
        )
        reranked_pages = {
            int(page_id): float(score)
            for page_id, score in zip(reranked_page_result["page_ids"], reranked_page_result["scores"])
        }

        candidate_pages = set(bm25_pages) | set(reranked_pages)
        if not candidate_pages:
            return {
                "page_ids": [],
                "scores": [],
                "ranks": [],
                "bm25_page_ids": [],
                "chunk_page_ids": [],
                "reranked_page_ids": [],
            }

        bm25_scores = list(bm25_pages.values())
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_max = max(bm25_scores) if bm25_scores else 0.0
        fused_items: list[tuple[int, float]] = []
        for page_id in candidate_pages:
            bm25_score = bm25_pages.get(page_id, 0.0)
            rerank_score = reranked_pages.get(page_id, 0.0)
            if bm25_max > bm25_min:
                bm25_norm = (bm25_score - bm25_min) / (bm25_max - bm25_min + 1e-8)
            elif page_id in bm25_pages:
                bm25_norm = 1.0
            else:
                bm25_norm = 0.0
            final_score = float(rerank_score + 0.2 * bm25_norm)
            fused_items.append((int(page_id), final_score))

        fused_items.sort(key=lambda item: item[1], reverse=True)
        fused_items = fused_items[:topk]
        return {
            "page_ids": [int(page_id) for page_id, _ in fused_items],
            "scores": [float(score) for _, score in fused_items],
            "ranks": list(range(1, len(fused_items) + 1)),
            "bm25_page_ids": [int(page_id) for page_id in bm25_result["page_ids"]],
            "chunk_page_ids": [int(page_id) for page_id in coarse_chunk_result["page_ids"]],
            "reranked_page_ids": [int(page_id) for page_id in reranked_page_result["page_ids"]],
        }
