"""Case-study export helpers for retrieval and DocVQA inspection."""

from __future__ import annotations

from typing import Any


def export_case_study(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export compact case-study rows with question, evidence, rankings, and answer."""
    exported: list[dict[str, Any]] = []
    for row in rows:
        exported.append(
            {
                "question": row.get("question", ""),
                "gt_evidence": row.get("evidence_page_ids", []),
                "visual_top_k": row.get("visual_top_k", []),
                "ocr_top_k": row.get("ocr_top_k", []),
                "fusion_top_k": row.get("fusion_top_k", []),
                "final_answer": row.get("final_answer", ""),
            }
        )
    return exported
