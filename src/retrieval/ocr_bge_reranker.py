"""Local bge-reranker-v2-m3 OCR candidate reranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger


class OCRBGEReranker:
    """Rerank OCR page candidates within one document using a local BGE reranker."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_name = str(cfg.ocr_retrieval.reranker_model_name)
        self.device = str(cfg.ocr_retrieval.device)
        self.local_files_only = bool(cfg.ocr_retrieval.local_files_only)
        self.batch_size = int(cfg.ocr_retrieval.rerank_batch_size)
        self.max_length = int(cfg.ocr_retrieval.rerank_max_length)
        self.logger = get_logger("ocr_bge_reranker")
        self._tokenizer = None
        self._model = None
        self._torch = None

    def rerank(self, question: str, candidates: list[dict[str, Any]], topk: int) -> dict[str, list]:
        """Rerank preselected OCR page candidates by pair scoring."""
        if not candidates:
            return {"page_ids": [], "scores": [], "ranks": []}
        self._ensure_model()
        scores = self._score_pairs(question, [str(item.get("text", "")) for item in candidates])
        ranked_items = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:topk]
        return {
            "page_ids": [int(item[0]["page_id"]) for item in ranked_items],
            "scores": [float(item[1]) for item in ranked_items],
            "ranks": list(range(1, len(ranked_items) + 1)),
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for OCRBGEReranker.") from exc

        resolved_model = Path(self.model_name).resolve() if Path(self.model_name).exists() else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=self.local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(resolved_model, local_files_only=self.local_files_only)
        self._torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Configured OCR reranker device %s is unavailable; falling back to cpu.", self.device)
            self.device = "cpu"
        self._model.to(self.device)
        self._model.eval()
        self.logger.info("Loaded OCR reranker from %s on %s", resolved_model, self.device)

    def _score_pairs(self, question: str, texts: list[str]) -> list[float]:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = self._tokenizer(
                    [question] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self._model(**inputs).logits.reshape(-1)
                scores.extend(logits.detach().cpu().tolist())
        return [float(score) for score in scores]
