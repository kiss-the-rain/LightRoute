"""Local jina-reranker-v3 OCR candidate reranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger


def _resolve_model_path(model_name: str, *, local_files_only: bool) -> str:
    model_path = Path(str(model_name)).expanduser()
    looks_like_local_path = model_path.is_absolute() or "/" in str(model_name)
    if local_files_only and looks_like_local_path:
        if not model_path.exists():
            raise FileNotFoundError(
                "OCR Jina v3 reranker path does not exist: "
                f"{model_path}. Copy the local model directory there or update "
                "configs/retrieval.yaml:ocr_jina_reranker.model_name."
            )
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                "OCR Jina v3 reranker directory is missing config.json: "
                f"{model_path}. The directory must be a complete Hugging Face model snapshot."
            )
        return str(model_path.resolve())
    return str(model_name)


class OCRJinaV3Reranker:
    """Rerank OCR page candidates within one document using a local Jina reranker."""

    def __init__(self, cfg: Any, config_attr: str = "ocr_jina_reranker") -> None:
        self.cfg = cfg
        self.config_attr = str(config_attr)
        config = getattr(cfg, self.config_attr)
        self.model_name = str(getattr(config, "model_name"))
        self.device = str(getattr(config, "device"))
        self.local_files_only = bool(getattr(config, "local_files_only", True))
        self.batch_size = int(getattr(config, "batch_size", 8))
        self.max_length = int(getattr(config, "max_length", 2048))
        self.logger = get_logger("ocr_jina_v3_reranker")
        self._tokenizer = None
        self._model = None
        self._torch = None

    def rerank(self, question: str, candidates: list[dict[str, Any]], topk: int) -> dict[str, list]:
        if not candidates:
            return {"page_ids": [], "scores": [], "ranks": []}
        self._ensure_model()
        scores = self._score_pairs(question, [str(item.get("text", "")) for item in candidates])
        ranked_items = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:topk]
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
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for OCRJinaV3Reranker.") from exc

        resolved_model = _resolve_model_path(self.model_name, local_files_only=self.local_files_only)
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            resolved_model,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        self._torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Configured OCR Jina v3 reranker device %s is unavailable; falling back to cpu.", self.device)
            self.device = "cpu"
        self._model.to(self.device)
        self._model.eval()
        self.logger.info("Loaded OCR Jina v3 reranker from %s on %s", resolved_model, self.device)

    def _score_pairs(self, question: str, texts: list[str]) -> list[float]:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        if hasattr(self._model, "rerank"):
            documents = [str(text or " ") for text in texts]
            results = self._model.rerank(str(question or ""), documents, top_n=None)
            scores = [0.0] * len(documents)
            for item in results:
                index = int(item.get("index", -1))
                if 0 <= index < len(scores):
                    scores[index] = float(item.get("relevance_score", 0.0))
            return scores

        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = [str(text or " ") for text in texts[start : start + self.batch_size]]
                pairs = [(question, text) for text in batch_texts]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self._model(**inputs).logits.squeeze(-1)
                scores.extend(logits.detach().float().cpu().tolist())
        return [float(score) for score in scores]
