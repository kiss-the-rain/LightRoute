"""End-to-end DocVQA inference placeholder."""

from __future__ import annotations

from typing import Any


def infer_docvqa_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": sample["qid"],
        "prediction": "",
        "evidence_pages": [],
    }
