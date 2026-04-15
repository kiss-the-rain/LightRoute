from __future__ import annotations

import types
import unittest
from unittest import mock

from src.inference.infer_retrieval import _retrieve_ocr_with_bm25_bge_reranker_for_sample


class DualCoarseHelperTests(unittest.TestCase):
    def test_ocr_page_pipeline_only_uses_routed_and_semantic_subsets(self):
        sample = {
            "doc_id": "doc-1",
            "question": "invoice total",
            "ocr_paths": ["doc-1_p0.json", "doc-1_p1.json", "doc-1_p2.json"],
            "image_paths": ["doc-1_p0.jpg", "doc-1_p1.jpg", "doc-1_p2.jpg"],
            "evidence_pages": [1],
        }
        cfg = types.SimpleNamespace(
            ocr_router=types.SimpleNamespace(
                enable_ocr_page_coarse=True,
                bypass_threshold=1,
                coarse_topk=2,
                coarse_method="bm25",
            ),
            ocr_semantic_retrieval=types.SimpleNamespace(
                enable_bge_m3=True,
                semantic_topk=2,
            ),
            ocr_reranker=types.SimpleNamespace(
                enable_bge_reranker=True,
                rerank_topk=1,
            ),
        )
        retriever = object()
        reranker = object()

        with mock.patch(
            "src.retrieval.ocr_page_pipeline.route_document_pages_with_adaptive_coarse",
            return_value=(
                {
                    "page_ids": [1, 2],
                    "scores": [1.0, 0.5],
                    "ranks": [1, 2],
                    "metadata": {"bypassed": False, "num_pages_before": 3, "num_pages_after": 2},
                },
                {"ocr_bm25_num_pages_before_coarse": 3, "ocr_bm25_num_pages_after_coarse": 2},
            ),
        ), mock.patch(
            "src.retrieval.ocr_page_pipeline.run_ocr_page_bge_m3",
            return_value=(
                {"page_ids": [2], "scores": [0.9], "ranks": [1]},
                {"ocr_num_pages_after_bge": 1, "ocr_bge_embedding_cache_hits": 1, "ocr_bge_embedding_cache_misses": 0},
            ),
        ) as bge_mock, mock.patch(
            "src.retrieval.ocr_page_pipeline.run_ocr_page_reranker",
            return_value=(
                {"page_ids": [2], "scores": [1.2], "ranks": [1]},
                {"ocr_rerank_calls": 1, "ocr_num_pages_after_rerank": 1},
            ),
        ) as rerank_mock:
            result, _stats = _retrieve_ocr_with_bm25_bge_reranker_for_sample(
                cfg=cfg,
                sample=sample,
                topk=10,
                retriever=retriever,
                indexed_docs=set(),
                reranker=reranker,
                bm25_doc_text_cache={},
                bm25_doc_retriever_cache={},
            )

        self.assertEqual(bge_mock.call_args.kwargs["candidate_page_ids"], [1, 2])
        semantic_result = rerank_mock.call_args.kwargs["semantic_result"]
        self.assertEqual(semantic_result["page_ids"], [2])
        self.assertEqual(result["routed_page_ids"], [1, 2])
        self.assertEqual(result["page_ids"], [2])


if __name__ == "__main__":
    unittest.main()
