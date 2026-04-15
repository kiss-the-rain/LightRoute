from __future__ import annotations

import unittest

from src.models.gating_mlp import GateNet
from src.retrieval.dynamic_weighting import apply_branch_reweighting, compute_rule_based_weights
from src.utils.config_utils import ConfigNode


class DynamicWeightingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ConfigNode(
            {
                "dynamic_fusion": ConfigNode(
                    {
                        "min_weight": 0.2,
                        "max_weight": 0.8,
                        "query_bias_scale": 0.18,
                        "page_correction_scale": 0.08,
                        "confidence_correction_scale": 0.08,
                    }
                )
            }
        )
        self.sample = {"question": "What is the invoice total for 2024?"}
        self.rows = [
            {
                "page_id": 0,
                "ocr_score": 0.9,
                "normalized_ocr_score": 0.9,
                "visual_score": 0.4,
                "normalized_visual_score": 0.4,
                "ocr_token_count": 140.0,
                "ocr_empty_like_flag": 0.0,
                "ocr_non_alnum_ratio": 0.1,
                "ocr_digit_ratio": 0.2,
                "ocr_reliability": 0.9,
                "question_token_count": 7.0,
                "contains_year": 1.0,
                "contains_amount_cue": 1.0,
                "contains_table_cue": 0.0,
            },
            {
                "page_id": 1,
                "ocr_score": 0.2,
                "normalized_ocr_score": 0.2,
                "visual_score": 0.3,
                "normalized_visual_score": 0.3,
                "ocr_token_count": 120.0,
                "ocr_empty_like_flag": 0.0,
                "ocr_non_alnum_ratio": 0.1,
                "ocr_digit_ratio": 0.1,
                "ocr_reliability": 0.8,
                "question_token_count": 7.0,
                "contains_year": 1.0,
                "contains_amount_cue": 1.0,
                "contains_table_cue": 0.0,
            },
        ]

    def test_rule_weights_are_bounded_and_sample_level(self) -> None:
        alpha_v, alpha_o, _ = compute_rule_based_weights(self.rows, self.sample, "combined", self.cfg)
        self.assertAlmostEqual(alpha_v + alpha_o, 1.0, places=5)
        self.assertGreaterEqual(alpha_v, 0.2)
        self.assertLessEqual(alpha_v, 0.8)
        self.assertGreater(alpha_o, alpha_v)

        weighted_rows = apply_branch_reweighting(self.rows, alpha_v=alpha_v, alpha_o=alpha_o)
        self.assertAlmostEqual(weighted_rows[0]["ocr_weight"], alpha_o)
        self.assertAlmostEqual(weighted_rows[1]["ocr_weight"], alpha_o)
        self.assertAlmostEqual(weighted_rows[0]["visual_weight"], alpha_v)
        self.assertAlmostEqual(weighted_rows[1]["visual_weight"], alpha_v)

    def test_gatenet_outputs_valid_distribution(self) -> None:
        gate = GateNet(input_dim=4, hidden_dim=8, min_weight=0.2, max_weight=0.8)
        alpha_v, alpha_o = gate.predict_weights([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(alpha_v + alpha_o, 1.0, places=5)
        self.assertGreaterEqual(alpha_v, 0.2)
        self.assertLessEqual(alpha_v, 0.8)


if __name__ == "__main__":
    unittest.main()
