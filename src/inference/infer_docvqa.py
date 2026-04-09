"""End-to-end DocVQA inference with retrieval, fusion, evidence selection, and answer generation."""

from __future__ import annotations

from typing import Any

from src.inference.answer_generator import AnswerGenerator
from src.inference.infer_retrieval import infer_retrieval


def infer_docvqa_sample(
    cfg: Any,
    sample: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    fusion_mode: str = "adaptive_fusion",
    top_n: int = 1,
) -> dict[str, Any]:
    """Run end-to-end DocVQA inference for one sample."""
    retrieval_result = infer_retrieval(cfg, sample, retrieval_bundle, mode=fusion_mode)
    top_page_ids = retrieval_result["page_ids"][:top_n]
    ocr_by_page = {entry["page_id"]: entry for entry in sample.get("ocr_results", [])}
    evidence_pages = [ocr_by_page[page_id] for page_id in top_page_ids if page_id in ocr_by_page]
    answer_generator = AnswerGenerator()
    answer = answer_generator.generate_answer(sample["question"], evidence_pages)
    return {
        "qid": sample["qid"],
        "doc_id": sample["doc_id"],
        "question": sample["question"],
        "predicted_answer": answer,
        "evidence_pages": top_page_ids,
        "retrieval": retrieval_result,
    }
