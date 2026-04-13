"""Dense OCR page retriever backed by a local BGE-M3 checkpoint."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from src.retrieval.ocr_retriever import OCRRetriever
from src.utils.io_utils import load_json
from src.utils.logger import get_logger


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def format_bge_query(question: str, variant: str = "raw_question") -> str:
    """Format a question string for OCR dense retrieval experiments."""
    question = str(question or "").strip()
    mapping = {
        "raw_question": question,
        "instruction_query": f"Find the page that answers the question: {question}",
        "retrieval_query": f"Retrieve the document page containing the answer to: {question}",
    }
    if variant not in mapping:
        raise ValueError(f"Unsupported BGE query variant: {variant}")
    return mapping[variant]


def build_clean_ocr_page_text(page_ocr_json: dict[str, Any], mode: str = "clean_v1") -> str:
    """Build a dense-retrieval-friendly OCR page text string from one OCR JSON payload."""
    tokens = _extract_text_units(page_ocr_json)
    cleaned_tokens: list[str] = []
    seen_recent: list[str] = []
    for raw_token in tokens:
        token = str(raw_token or "").strip().lower()
        if not token:
            continue
        if token.startswith("source:"):
            continue
        if URL_PATTERN.search(token):
            continue
        if len(token) == 1 and not token.isalnum():
            continue
        if len(token) == 1 and not token.isdigit() and not token.isalpha():
            continue
        token = URL_PATTERN.sub("", token).strip()
        token = re.sub(r"\s+", " ", token)
        if not token:
            continue
        if seen_recent[-5:].count(token) >= 2:
            continue
        cleaned_tokens.append(token)
        seen_recent.append(token)

    if mode == "clean_v1":
        return " ".join(cleaned_tokens)
    if mode == "clean_trunc_v1":
        return " ".join(cleaned_tokens[:256])
    raise ValueError(f"Unsupported OCR clean mode: {mode}")


def load_bge_page_text(ocr_path: str, variant: str = "raw_page_text") -> str:
    """Load one OCR page text under the requested dense-retrieval text variant."""
    try:
        payload = load_json(ocr_path)
    except Exception:
        return ""
    if variant == "raw_page_text":
        return " ".join(_normalize_tokens(_extract_text_units(payload)))
    if variant == "clean_page_text":
        return build_clean_ocr_page_text(payload, mode="clean_v1")
    if variant == "clean_trunc_page_text":
        return build_clean_ocr_page_text(payload, mode="clean_trunc_v1")
    raise ValueError(f"Unsupported OCR page text variant: {variant}")


def summarize_page_text_lengths(page_texts: list[str], variant: str) -> dict[str, float | str]:
    """Summarize token-length statistics for one OCR page-text variant."""
    token_lengths = [len(text.split()) for text in page_texts]
    if not token_lengths:
        return {
            "page_text_variant": variant,
            "avg_token_count": 0.0,
            "max_token_count": 0.0,
            "truncated_ratio": 0.0,
        }
    truncated_pages = sum(1 for length in token_lengths if length >= 256)
    return {
        "page_text_variant": variant,
        "avg_token_count": float(sum(token_lengths) / len(token_lengths)),
        "max_token_count": float(max(token_lengths)),
        "truncated_ratio": float(truncated_pages / len(token_lengths)),
    }


def _extract_text_units(payload: Any) -> list[str]:
    """Recursively extract text-like OCR units from heterogeneous payloads."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        texts: list[str] = []
        for item in payload:
            texts.extend(_extract_text_units(item))
        return texts
    if isinstance(payload, dict):
        texts: list[str] = []
        for key in ("tokens", "words", "text", "blocks", "lines"):
            if key in payload:
                texts.extend(_extract_text_units(payload[key]))
        if texts:
            return texts
        for value in payload.values():
            texts.extend(_extract_text_units(value))
        return texts
    return []


def _normalize_tokens(tokens: list[str]) -> list[str]:
    """Normalize OCR text units for dense retrieval without aggressive filtering."""
    normalized: list[str] = []
    for token in tokens:
        token = str(token or "").strip().lower()
        token = re.sub(r"\s+", " ", token)
        if token:
            normalized.append(token)
    return normalized


class OCRBGERetriever(OCRRetriever):
    """Document-scoped dense OCR retriever using BGE-style text embeddings."""

    def __init__(self, cfg: Any, config_attr: str = "ocr_retrieval") -> None:
        self.cfg = cfg
        self.config_attr = str(config_attr)
        config = getattr(cfg, self.config_attr)
        model_name = getattr(config, "model_name", None) or getattr(config, "bge_model_name", None)
        self.model_name = str(model_name)
        self.device = str(getattr(config, "device"))
        self.local_files_only = bool(getattr(config, "local_files_only"))
        self.batch_size = int(getattr(config, "batch_size", 8))
        self.max_length = int(getattr(config, "max_length", 512))
        self.logger = get_logger("ocr_bge_retriever")
        self.index: dict[str, dict[str, Any]] = {}
        self._tokenizer = None
        self._model = None
        self._torch = None

    def build_document_index(self, doc_id: str, page_texts: list[str], page_ids: list[int]) -> None:
        """Encode one document's OCR page texts once and cache embeddings."""
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
        """Retrieve top-k OCR pages within one document by dense similarity."""
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
        """Retrieve and rank only a routed subset of OCR pages from an indexed document."""
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
        """Expose cached page texts for downstream reranking."""
        return list(self.index.get(doc_id, {}).get("page_texts", []))

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for OCRBGERetriever.") from exc

        resolved_model = Path(self.model_name).resolve() if Path(self.model_name).exists() else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=self.local_files_only)
        self._model = AutoModel.from_pretrained(resolved_model, local_files_only=self.local_files_only)
        self._torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Configured OCR BGE device %s is unavailable; falling back to cpu.", self.device)
            self.device = "cpu"
        self._model.to(self.device)
        self._model.eval()
        self.logger.info("Loaded OCR BGE retriever from %s on %s", resolved_model, self.device)

    def count_model_tokens(self, text: str) -> int:
        """Count tokenizer input length for one text without truncation."""
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

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        embeddings: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self._model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.detach().cpu().numpy().astype(float))
        return np.concatenate(embeddings, axis=0)
