"""Offline chunk-level OCR retrieval backed by a local jina-embeddings-v3 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np
from src.utils.io_utils import ensure_dir

PREPARED_MODEL_VERSION = "v5"


class OCRJinaChunkRetriever:
    """Document-scoped OCR chunk retriever using local jina-embeddings-v3 embeddings."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_name = str(cfg.ocr_jina_retrieval.model_name)
        self.dependency_model_name = str(cfg.ocr_jina_retrieval.dependency_model_name)
        self.device = str(cfg.ocr_jina_retrieval.device)
        self.local_files_only = bool(cfg.ocr_jina_retrieval.local_files_only)
        self.batch_size = int(cfg.ocr_jina_retrieval.batch_size)
        self.max_length = int(cfg.ocr_jina_retrieval.max_length)
        self.query_task = str(cfg.ocr_jina_retrieval.query_task)
        self.passage_task = str(cfg.ocr_jina_retrieval.passage_task)
        self.cache_dir = ensure_dir(Path(cfg.paths.cache_dir) / "ocr_jina_chunk")
        self.index: dict[str, dict[str, Any]] = {}
        self._model = None
        self._resolved_model_path: Path | None = None
        self._resolved_dependency_model_path: Path | None = None
        self._prepared_model_path: Path | None = None

    def build_document_index(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        """Encode and cache one document's OCR chunks once."""
        if doc_id in self.index:
            return
        self._ensure_model()
        chunk_texts = [str(chunk.get("chunk_text", "")) for chunk in chunks]
        embeddings = self._encode_texts(chunk_texts, task=self.passage_task) if chunk_texts else np.zeros((0, 1), dtype=float)
        self.index[doc_id] = {
            "chunks": [dict(chunk) for chunk in chunks],
            "embeddings": embeddings,
        }

    def retrieve(self, query: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k OCR chunks within one document."""
        if doc_id not in self.index:
            return {"chunk_ids": [], "page_ids": [], "scores": [], "ranks": [], "chunks": []}
        self._ensure_model()
        query_embedding = self._encode_texts([str(query or "")], task=self.query_task)[0]
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
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required for OCRJinaChunkRetriever.") from exc

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        resolved_model = self._prepare_local_model_dir()
        prepared_path = str(resolved_model)
        if prepared_path not in sys.path:
            sys.path.insert(0, prepared_path)
        self._model = SentenceTransformer(
            prepared_path,
            device=self.device,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

    def _resolve_local_model_path(self) -> str:
        """Resolve and validate the offline jina-embeddings-v3 directory."""
        if self._resolved_model_path is not None:
            return str(self._resolved_model_path)
        candidate = Path(self.model_name).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        config_path = candidate / "config.json"
        if not candidate.exists() or not candidate.is_dir() or not config_path.exists():
            raise FileNotFoundError(
                "Offline jina-embeddings-v3 model directory not found: "
                f"{candidate}. Offline mode is required. Check that the model directory is uploaded, "
                "config.json exists, and cfg.ocr_jina_retrieval.model_name matches the local directory."
            )
        self._resolved_model_path = candidate
        return str(candidate)

    def _resolve_dependency_model_path(self) -> str:
        """Resolve and validate the offline dependency model directory used in auto_map."""
        if self._resolved_dependency_model_path is not None:
            return str(self._resolved_dependency_model_path)
        candidate = Path(self.dependency_model_name).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        required_files = [
            candidate / "configuration_xlm_roberta.py",
            candidate / "modeling_xlm_roberta.py",
            candidate / "modeling_lora.py",
        ]
        if not candidate.exists() or not candidate.is_dir() or not all(path.exists() for path in required_files):
            raise FileNotFoundError(
                "Offline jina dependency model directory not found or incomplete: "
                f"{candidate}. Offline mode is required. Check that "
                "cfg.ocr_jina_retrieval.dependency_model_name points to the local "
                "xlm-roberta-flash-implementation directory with configuration_xlm_roberta.py, "
                "modeling_xlm_roberta.py, and modeling_lora.py."
            )
        self._resolved_dependency_model_path = candidate
        return str(candidate)

    def _prepare_local_model_dir(self) -> Path:
        """Create a prepared local model directory with auto_map rewritten to local dependencies."""
        if self._prepared_model_path is not None:
            return self._prepared_model_path
        resolved_model = Path(self._resolve_local_model_path())
        resolved_dependency = Path(self._resolve_dependency_model_path())
        cache_key = "|".join([PREPARED_MODEL_VERSION, str(resolved_model), str(resolved_dependency)])
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
        prepared_dir = self.cache_dir / f"prepared_model_{PREPARED_MODEL_VERSION}_{digest}"
        if not prepared_dir.exists():
            ensure_dir(prepared_dir)
        self._materialize_local_tree(source_dir=resolved_model, prepared_dir=prepared_dir, skip_names={"config.json"})
        self._materialize_local_tree(source_dir=resolved_dependency, prepared_dir=prepared_dir, skip_names=set())

        config_path = resolved_model / "config.json"
        prepared_config_path = prepared_dir / "config.json"
        config = json.loads(config_path.read_text())
        auto_map = config.get("auto_map", {})
        if isinstance(auto_map, dict):
            rewritten = {}
            for key, value in auto_map.items():
                if isinstance(value, str) and "--" in value:
                    suffix = value.split("--", 1)[1]
                    rewritten[key] = suffix
                else:
                    rewritten[key] = value
            config["auto_map"] = rewritten
        prepared_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
        self._ensure_sentence_transformer_config_aliases(source_dir=resolved_model, prepared_dir=prepared_dir)
        self._patch_prepared_relative_imports(prepared_dir)
        self._patch_prepared_custom_st(prepared_dir)
        self._prepared_model_path = prepared_dir
        return prepared_dir

    @staticmethod
    def _materialize_local_tree(source_dir: Path, prepared_dir: Path, skip_names: set[str]) -> None:
        """Mirror local model files into the prepared directory."""
        for child in source_dir.rglob("*"):
            relative = child.relative_to(source_dir)
            if relative.parts and relative.parts[0] in skip_names:
                continue
            target = prepared_dir / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(child, target)
            except OSError:
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)

    @staticmethod
    def _ensure_sentence_transformer_config_aliases(source_dir: Path, prepared_dir: Path) -> None:
        """Create legacy SentenceTransformers config aliases expected by custom modules."""
        source_config = source_dir / "config_sentence_transformers.json"
        if source_config.exists():
            payload = source_config.read_text()
        else:
            payload = "{}"
        alias_names = [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]
        for name in alias_names:
            alias_path = prepared_dir / name
            if not alias_path.exists():
                alias_path.write_text(payload)

    @staticmethod
    def _patch_prepared_custom_st(prepared_dir: Path) -> None:
        """Force the prepared custom_st.py to keep trust_remote_code/local_files_only on nested loads."""
        custom_st_path = prepared_dir / "custom_st.py"
        if not custom_st_path.exists():
            return
        text = custom_st_path.read_text()
        updated = text
        if "from configuration_xlm_roberta import XLMRobertaFlashConfig" not in updated:
            updated = re.sub(
                r"from transformers import ([^\n]+)",
                lambda match: (
                    f"from transformers import {match.group(1)}\n"
                    "from configuration_xlm_roberta import XLMRobertaFlashConfig\n"
                    "from modeling_lora import XLMRobertaLoRA"
                ),
                updated,
                count=1,
            )
        updated = re.sub(
            r"AutoConfig\.from_pretrained\(",
            "XLMRobertaFlashConfig.from_pretrained(",
            updated,
        )
        updated = re.sub(
            r"AutoModel\.from_pretrained\(",
            "XLMRobertaLoRA.from_pretrained(",
            updated,
        )
        updated = re.sub(
            r"XLMRobertaFlashConfig\.from_pretrained\((.*?)\)",
            lambda m: _ensure_kwarg_block(m.group(0), "trust_remote_code=True, local_files_only=True"),
            updated,
            flags=re.DOTALL,
        )
        updated = re.sub(
            r"XLMRobertaLoRA\.from_pretrained\((.*?)\)",
            lambda m: _ensure_kwarg_block(m.group(0), "trust_remote_code=True, local_files_only=True"),
            updated,
            flags=re.DOTALL,
        )
        if updated != text:
            custom_st_path.write_text(updated)

    @staticmethod
    def _patch_prepared_relative_imports(prepared_dir: Path) -> None:
        """Rewrite package-relative imports to same-directory imports inside the prepared tree."""
        for py_path in prepared_dir.rglob("*.py"):
            text = py_path.read_text()
            updated = re.sub(r"from \.([A-Za-z0-9_]+) import ", r"from \1 import ", text)
            updated = re.sub(r"from \.([A-Za-z0-9_]+) import\(", r"from \1 import(", updated)
            if updated != text:
                py_path.write_text(updated)


def _ensure_kwarg_block(call_text: str, extra_kwargs: str) -> str:
    """Append required kwargs to a from_pretrained call if absent."""
    if "trust_remote_code=" in call_text or "local_files_only=" in call_text:
        return call_text
    return call_text[:-1] + f", {extra_kwargs})"

    def _encode_texts(self, texts: list[str], task: str) -> np.ndarray:
        """Encode texts with task-specific adapters and normalized embeddings."""
        if not texts:
            return np.zeros((0, 1), dtype=float)
        assert self._model is not None
        array = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            task=task,
        )
        array = np.asarray(array, dtype=float)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        return array
