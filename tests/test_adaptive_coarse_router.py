from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.retrieval.adaptive_coarse_router import route_document_pages_with_adaptive_coarse


class AdaptiveCoarseRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.sample = {
            "doc_id": "doc-1",
            "question": "invoice total",
            "ocr_paths": [str(self.root / f"doc-1_p{i}.json") for i in range(3)],
            "image_paths": [str(self.root / f"doc-1_p{i}.jpg") for i in range(3)],
            "evidence_pages": [1],
        }
        self.cfg = types.SimpleNamespace(
            retrieval_router=types.SimpleNamespace(
                enable_adaptive_coarse=True,
                enable_bm25_coarse=True,
                bypass_threshold=2,
                coarse_topk=2,
                coarse_method="bm25",
            ),
            text_retriever=types.SimpleNamespace(k1=1.5, b=0.75, cache_path=str(self.root / "bm25.pkl")),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("src.retrieval.adaptive_coarse_router.load_page_ocr_text")
    def test_bypass_keeps_all_pages(self, load_text):
        load_text.side_effect = ["invoice", "total", "summary"]
        self.cfg.retrieval_router.bypass_threshold = 3

        result, stats = route_document_pages_with_adaptive_coarse(
            self.cfg,
            self.sample,
            doc_text_cache={},
            doc_retriever_cache={},
        )

        self.assertEqual(result["page_ids"], [0, 1, 2])
        self.assertTrue(result["metadata"]["bypassed"])
        self.assertEqual(stats["num_bypass_samples"], 1)
        self.assertEqual(stats["num_coarse_samples"], 0)

    @mock.patch("src.retrieval.adaptive_coarse_router.load_page_ocr_text")
    def test_coarse_returns_min_n_and_topk_pages(self, load_text):
        load_text.side_effect = ["irrelevant", "invoice total", "invoice"]

        result, stats = route_document_pages_with_adaptive_coarse(
            self.cfg,
            self.sample,
            doc_text_cache={},
            doc_retriever_cache={},
        )

        self.assertEqual(len(result["page_ids"]), 2)
        self.assertFalse(result["metadata"]["bypassed"])
        self.assertEqual(result["metadata"]["coarse_topk"], 2)
        self.assertEqual(stats["num_bypass_samples"], 0)
        self.assertEqual(stats["num_coarse_samples"], 1)

    @mock.patch("src.retrieval.adaptive_coarse_router.load_page_ocr_text")
    def test_disabled_bm25_keeps_full_page_set_without_loading_text(self, load_text):
        self.cfg.retrieval_router.enable_bm25_coarse = False

        result, stats = route_document_pages_with_adaptive_coarse(
            self.cfg,
            self.sample,
            doc_text_cache={},
            doc_retriever_cache={},
        )

        self.assertEqual(result["page_ids"], [0, 1, 2])
        self.assertTrue(result["metadata"]["bypassed"])
        self.assertFalse(result["metadata"]["bm25_coarse_enabled"])
        self.assertEqual(stats["num_pages_before_coarse"], 3)
        self.assertEqual(stats["num_pages_after_coarse"], 3)
        self.assertEqual(stats["num_bypass_samples"], 1)
        self.assertEqual(stats["num_coarse_samples"], 0)
        self.assertEqual(stats["coarse_page_text_cache_hits"], 0)
        self.assertEqual(stats["coarse_page_text_cache_misses"], 0)
        load_text.assert_not_called()

    @mock.patch("src.retrieval.adaptive_coarse_router.load_page_ocr_text")
    def test_ocr_router_uses_custom_enabled_flag(self, load_text):
        load_text.side_effect = ["irrelevant", "invoice total", "invoice"]
        self.cfg.ocr_router = types.SimpleNamespace(
            enable_ocr_page_coarse=True,
            enable_bm25_coarse=True,
            bypass_threshold=2,
            coarse_topk=1,
            coarse_method="bm25",
        )

        result, stats = route_document_pages_with_adaptive_coarse(
            self.cfg,
            self.sample,
            doc_text_cache={},
            doc_retriever_cache={},
            router_cfg=self.cfg.ocr_router,
            stats_prefix="ocr",
            enabled_attr="enable_ocr_page_coarse",
        )

        self.assertEqual(len(result["page_ids"]), 1)
        self.assertEqual(stats["ocr_num_coarse_samples"], 1)


if __name__ == "__main__":
    unittest.main()
