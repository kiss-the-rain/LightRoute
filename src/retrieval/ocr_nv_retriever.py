"""Chunk-level OCR retrieval backed by a local NV-Embed-v2 checkpoint."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io_utils import ensure_dir, load_pickle, save_pickle


def build_nv_retrieval_query(question: str, instruction: str | None = None) -> str:
    """Format an NV-Embed-v2 retrieval query with an instruction prefix."""
    question = str(question or "").strip()
    prefix = instruction or "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
    return f"{prefix}{question}"


class OCRNVChunkRetriever:
    """Document-scoped OCR chunk retriever using local NV-Embed-v2 embeddings."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_name = str(cfg.ocr_nv_retrieval.model_name)
        self.device = str(cfg.ocr_nv_retrieval.device)
        self.local_files_only = bool(cfg.ocr_nv_retrieval.local_files_only)
        self.batch_size = int(cfg.ocr_nv_retrieval.batch_size)
        self.max_length = int(cfg.ocr_nv_retrieval.max_length)
        self.query_instruction = str(cfg.ocr_nv_retrieval.query_instruction)
        self.use_disk_cache = bool(getattr(cfg.runtime, "use_cache", True))
        self.overwrite_disk_cache = bool(getattr(cfg.runtime, "overwrite_cache", False))
        self.cache_dir = ensure_dir(Path(cfg.paths.cache_dir) / "ocr_nv_chunk")
        self.index: dict[str, dict[str, Any]] = {}
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_model_path: Path | None = None

    def build_document_index(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        """Encode and cache one document's OCR chunks once."""
        if doc_id in self.index:
            return
        cache_path = self._cache_path_for_doc(doc_id)
        if self.use_disk_cache and not self.overwrite_disk_cache and cache_path.exists():
            payload = load_pickle(cache_path)
            self.index[doc_id] = {
                "chunks": [dict(chunk) for chunk in payload.get("chunks", [])],
                "embeddings": np.asarray(payload.get("embeddings", np.zeros((0, 1), dtype=float)), dtype=float),
            }
            return
        self._ensure_model()
        chunk_texts = [str(chunk.get("chunk_text", "")) for chunk in chunks]
        embeddings = self._encode_texts(chunk_texts, is_query=False) if chunk_texts else np.zeros((0, 1), dtype=float)
        self.index[doc_id] = {
            "chunks": [dict(chunk) for chunk in chunks],
            "embeddings": embeddings,
        }
        if self.use_disk_cache:
            save_pickle(
                {
                    "chunks": self.index[doc_id]["chunks"],
                    "embeddings": embeddings,
                },
                cache_path,
            )

    def retrieve(self, query: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k OCR chunks within one document."""
        if doc_id not in self.index:
            return {"chunk_ids": [], "page_ids": [], "scores": [], "ranks": [], "chunks": []}
        self._ensure_model()
        query_text = build_nv_retrieval_query(query, instruction=self.query_instruction)
        query_embedding = self._encode_texts([query_text], is_query=True)[0]
        chunk_embeddings = self.index[doc_id]["embeddings"]
        scores = chunk_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:topk]
        ranked_chunks = [self.index[doc_id]["chunks"][idx] for idx in ranked_indices]
        return {
            "chunk_ids": [str(chunk["chunk_id"]) for chunk in ranked_chunks],
            "page_ids": [int(chunk["page_id"]) for chunk in ranked_chunks],
            "scores": [float(scores[idx]) for idx in ranked_indices],
            "ranks": list(range(1, len(ranked_indices) + 1)),
            "chunks": ranked_chunks,
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer, modeling_utils
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for OCRNVChunkRetriever.") from exc

        resolved_model = self._resolve_local_model_path()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._patch_transformers_tied_weights_compat(modeling_utils)
        self._patch_transformers_dynamic_cache_compat(DynamicCache)
        config = AutoConfig.from_pretrained(
            resolved_model,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self._patch_config_to_local_path(config=config, resolved_model=resolved_model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            resolved_model,
            config=config,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self._torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self._model.to(self.device)
        self._model.eval()

    def _resolve_local_model_path(self) -> str:
        """Resolve and validate the offline NV-Embed-v2 directory."""
        if self._resolved_model_path is not None:
            return str(self._resolved_model_path)
        candidate = Path(self.model_name).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                "NV-Embed-v2 local model directory not found: "
                f"{candidate}. Set cfg.ocr_nv_retrieval.model_name to an existing local path."
            )
        if not candidate.is_dir():
            raise NotADirectoryError(
                "NV-Embed-v2 model path must be a local directory: "
                f"{candidate}"
            )
        self._resolved_model_path = candidate
        return str(candidate)

    def _cache_path_for_doc(self, doc_id: str) -> Path:
        """Return the document-level disk cache path for chunk embeddings."""
        model_key = str(self._resolved_model_path or self.model_name)
        cache_key = "|".join([doc_id, model_key, str(self.max_length)])
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pkl"

    @staticmethod
    def _patch_config_to_local_path(config: Any, resolved_model: str) -> None:
        """Force nested NV-Embed config references to remain local-only."""
        local_path = str(resolved_model)
        for attr_name in ("_name_or_path", "name_or_path"):
            if hasattr(config, attr_name):
                setattr(config, attr_name, local_path)

        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for attr_name in ("_name_or_path", "name_or_path"):
                if hasattr(text_config, attr_name):
                    setattr(text_config, attr_name, local_path)
            tokenizer_name = getattr(text_config, "tokenizer_name", None)
            if tokenizer_name is not None:
                setattr(text_config, "tokenizer_name", local_path)

        latent_attention_config = getattr(config, "latent_attention_config", None)
        if latent_attention_config is not None:
            for attr_name in ("_name_or_path", "name_or_path"):
                if hasattr(latent_attention_config, attr_name):
                    setattr(latent_attention_config, attr_name, local_path)

    @staticmethod
    def _patch_transformers_tied_weights_compat(modeling_utils: Any) -> None:
        """Add a compatibility shim for custom models missing all_tied_weights_keys."""
        pre_trained_model = modeling_utils.PreTrainedModel
        existing_attr = getattr(pre_trained_model, "all_tied_weights_keys", None)
        if isinstance(existing_attr, property) and existing_attr.fset is not None:
            return

        storage_name = "_codex_all_tied_weights_keys"

        def _expand_default_tied_keys(self: Any) -> dict[str, bool]:
            tied = getattr(self, "_tied_weights_keys", None)
            if tied is None:
                return {}
            if isinstance(tied, dict):
                return {str(key): bool(value) for key, value in tied.items()}
            if isinstance(tied, (list, tuple, set)):
                return {str(key): True for key in tied}
            return {}

        def _getter(self: Any) -> dict[str, bool]:
            stored = getattr(self, storage_name, None)
            if stored is not None:
                return stored
            expanded = _expand_default_tied_keys(self)
            setattr(self, storage_name, expanded)
            return expanded

        def _setter(self: Any, value: Any) -> None:
            if value is None:
                setattr(self, storage_name, {})
                return
            if isinstance(value, dict):
                normalized = {str(key): bool(flag) for key, flag in value.items()}
            elif isinstance(value, (list, tuple, set)):
                normalized = {str(key): True for key in value}
            else:
                normalized = {}
            setattr(self, storage_name, normalized)

        pre_trained_model.all_tied_weights_keys = property(_getter, _setter)

    @staticmethod
    def _patch_transformers_dynamic_cache_compat(dynamic_cache_cls: Any) -> None:
        """Restore DynamicCache.from_legacy_cache for older custom model code."""
        if hasattr(dynamic_cache_cls, "from_legacy_cache"):
            return

        @classmethod
        def _from_legacy_cache(cls, past_key_values: Any) -> Any:
            if past_key_values is None:
                return cls()
            return past_key_values

        dynamic_cache_cls.from_legacy_cache = _from_legacy_cache

    def _encode_texts(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Encode query or passage texts with normalization."""
        if not texts:
            return np.zeros((0, 1), dtype=float)
        assert self._model is not None and self._torch is not None
        embeddings: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                if hasattr(self._model, "encode"):
                    try:
                        output = self._model.encode(
                            batch_texts,
                            instruction="" if not is_query else self.query_instruction,
                            max_length=self.max_length,
                            batch_size=len(batch_texts),
                        )
                        if hasattr(output, "detach"):
                            output = output.detach().cpu()
                        array = np.asarray(output, dtype=float)
                        if array.ndim == 1:
                            array = np.expand_dims(array, axis=0)
                        array = self._l2_normalize(array)
                        embeddings.append(array)
                        continue
                    except TypeError:
                        pass

                assert self._tokenizer is not None
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

    @staticmethod
    def _l2_normalize(array: np.ndarray) -> np.ndarray:
        """Apply row-wise L2 normalization."""
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return array / norms
