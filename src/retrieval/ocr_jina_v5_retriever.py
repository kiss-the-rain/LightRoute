"""Dense OCR page retriever backed by a local jina-embeddings-v5 text retrieval checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.ocr_retriever import OCRRetriever
from src.utils.logger import get_logger


def _resolve_model_path(model_name: str, *, local_files_only: bool) -> str:
    model_path = Path(str(model_name)).expanduser()
    looks_like_local_path = model_path.is_absolute() or "/" in str(model_name)
    if local_files_only and looks_like_local_path:
        if not model_path.exists():
            raise FileNotFoundError(
                "OCR Jina v5 model path does not exist: "
                f"{model_path}. Copy the local model directory there or update "
                "configs/retrieval.yaml:ocr_jina_semantic_retrieval.model_name."
            )
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                "OCR Jina v5 model directory is missing config.json: "
                f"{model_path}. The directory must be a complete Hugging Face model snapshot."
            )
        return str(model_path.resolve())
    return str(model_name)


class OCRJinaV5Retriever(OCRRetriever):
    """Document-scoped dense OCR retriever using Jina v5 text retrieval embeddings."""

    def __init__(self, cfg: Any, config_attr: str = "ocr_jina_semantic_retrieval") -> None:
        self.cfg = cfg
        self.config_attr = str(config_attr)
        config = getattr(cfg, self.config_attr)
        self.model_name = str(getattr(config, "model_name"))
        self.device = str(getattr(config, "device"))
        self.local_files_only = bool(getattr(config, "local_files_only", True))
        self.batch_size = int(getattr(config, "batch_size", 8))
        self.max_length = int(getattr(config, "max_length", 4096))
        self.logger = get_logger("ocr_jina_v5_retriever")
        self.index: dict[str, dict[str, Any]] = {}
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._embedding_dim: int | None = None

    def build_document_index(self, doc_id: str, page_texts: list[str], page_ids: list[int]) -> None:
        if doc_id in self.index:
            return
        self._ensure_model()
        embeddings = self._encode_texts(page_texts)
        self.index[doc_id] = {
            "page_ids": [int(page_id) for page_id in page_ids],
            "page_texts": [str(text) for text in page_texts],
            "embeddings": embeddings,
        }

    def retrieve(self, query: str, doc_id: str, topk: int) -> dict[str, list]:
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}
        self._ensure_model()
        query_embedding = self._encode_texts([query])[0]
        page_embeddings = self.index[doc_id]["embeddings"]
        scores = page_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:topk]
        return {
            "page_ids": [int(self.index[doc_id]["page_ids"][idx]) for idx in ranked_indices],
            "scores": [float(scores[idx]) for idx in ranked_indices],
            "ranks": list(range(1, len(ranked_indices) + 1)),
        }

    def retrieve_subset(
        self,
        query: str,
        doc_id: str,
        candidate_page_ids: list[int],
        topk: int | None = None,
    ) -> dict[str, list]:
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}
        candidate_page_set = {int(page_id) for page_id in candidate_page_ids}
        if not candidate_page_set:
            return {"page_ids": [], "scores": [], "ranks": []}

        doc_page_ids = self.index[doc_id]["page_ids"]
        page_embeddings = self.index[doc_id]["embeddings"]
        filtered_page_ids: list[int] = []
        filtered_embeddings: list[Any] = []
        for page_id, embedding in zip(doc_page_ids, page_embeddings):
            if int(page_id) in candidate_page_set:
                filtered_page_ids.append(int(page_id))
                filtered_embeddings.append(embedding)
        if not filtered_page_ids:
            return {"page_ids": [], "scores": [], "ranks": []}

        self._ensure_model()
        query_embedding = self._encode_texts([query])[0]
        embedding_matrix = np.stack(filtered_embeddings, axis=0)
        scores = embedding_matrix @ query_embedding
        effective_topk = len(filtered_page_ids) if topk is None else min(int(topk), len(filtered_page_ids))
        ranked_indices = np.argsort(scores)[::-1][:effective_topk]
        return {
            "page_ids": [int(filtered_page_ids[idx]) for idx in ranked_indices],
            "scores": [float(scores[idx]) for idx in ranked_indices],
            "ranks": list(range(1, len(ranked_indices) + 1)),
        }

    def get_document_page_texts(self, doc_id: str) -> list[str]:
        return list(self.index.get(doc_id, {}).get("page_texts", []))

    def count_model_tokens(self, text: str) -> int:
        self._ensure_model()
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            str(text or ""),
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids", [])
        if isinstance(input_ids, list):
            return int(len(input_ids))
        return int(getattr(input_ids, "shape", [0])[-1])

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for OCRJinaV5Retriever.") from exc

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
            self.logger.warning("Configured OCR Jina v5 device %s is unavailable; falling back to cpu.", self.device)
            self.device = "cpu"
        self._model.to(self.device)
        self._model.eval()
        self.logger.info("Loaded OCR Jina v5 retriever from %s on %s", resolved_model, self.device)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._get_embedding_dim()), dtype=float)
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        normalized_texts = [str(text or "").strip() for text in texts]
        output_rows: list[np.ndarray | None] = [None] * len(normalized_texts)
        non_empty_items = [(index, text) for index, text in enumerate(normalized_texts) if text]
        with self._torch.no_grad():
            for start in range(0, len(non_empty_items), self.batch_size):
                batch_items = non_empty_items[start : start + self.batch_size]
                batch_texts = [text for _, text in batch_items]
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self._model(**inputs)
                pooled = outputs.last_hidden_state[:, 0]
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                batch_embeddings = pooled.detach().float().cpu().numpy().astype(float)
                self._embedding_dim = int(batch_embeddings.shape[1])
                for (original_index, _), embedding in zip(batch_items, batch_embeddings):
                    output_rows[original_index] = embedding
        embedding_dim = self._get_embedding_dim()
        zero_embedding = np.zeros((embedding_dim,), dtype=float)
        return np.stack([row if row is not None else zero_embedding for row in output_rows], axis=0)

    def _get_embedding_dim(self) -> int:
        if self._embedding_dim is not None:
            return int(self._embedding_dim)
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        with self._torch.no_grad():
            inputs = self._tokenizer(
                ["empty"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self._model(**inputs)
            pooled = outputs.last_hidden_state[:, 0]
            self._embedding_dim = int(pooled.shape[-1])
        return int(self._embedding_dim)
