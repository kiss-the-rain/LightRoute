"""Offline ColQwen visual retriever for document page retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from packaging.version import Version
from importlib import metadata as importlib_metadata

from src.utils.logger import get_logger
from src.utils.rank_utils import topk_from_scores


class ColQwenRetriever:
    """Document-scoped visual retriever backed by a local HF-native ColQwen checkpoint."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.model_name = self._resolve_model_path(str(cfg.visual_colqwen_retrieval.model_name))
        self.base_model_name = self._resolve_optional_model_path(
            getattr(cfg.visual_colqwen_retrieval, "base_model_name", None)
        )
        self.local_files_only = bool(cfg.visual_colqwen_retrieval.local_files_only)
        self.device = str(cfg.visual_colqwen_retrieval.device)
        self.batch_size = int(getattr(cfg.visual_colqwen_retrieval, "batch_size", 2))
        self.max_pages_per_doc = int(getattr(cfg.visual_colqwen_retrieval, "max_pages_per_doc", 256))
        self.logger = get_logger("colqwen_retriever")
        self.index: dict[str, dict[str, Any]] = {}
        self._engine_loaded = False
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def build_document_index(self, doc_id: str, image_paths: list[str], page_ids: list[int]) -> None:
        """Build and cache one document's page representations for repeated queries."""
        if doc_id in self.index:
            return
        self._ensure_engine_runtime()
        if self.max_pages_per_doc > 0:
            image_paths = image_paths[: self.max_pages_per_doc]
            page_ids = page_ids[: self.max_pages_per_doc]
        embeddings, kept_page_ids, kept_image_paths = self._encode_document_images(image_paths=image_paths, page_ids=page_ids)
        self.index[doc_id] = {
            "page_ids": kept_page_ids,
            "image_paths": kept_image_paths,
            "embeddings": embeddings,
        }

    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k pages inside one document."""
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}
        self._ensure_engine_runtime()
        query_embedding = self._encode_query_embedding(question)
        scores = self._score_query_against_document(query_embedding, self.index[doc_id]["embeddings"])
        return topk_from_scores(self.index[doc_id]["page_ids"], scores, topk)

    def retrieve_subset(
        self,
        question: str,
        doc_id: str,
        candidate_page_ids: list[int],
        topk: int | None = None,
    ) -> dict[str, list]:
        """Retrieve and rank only a routed subset of pages from an already indexed document."""
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}
        candidate_page_set = {int(page_id) for page_id in candidate_page_ids}
        if not candidate_page_set:
            return {"page_ids": [], "scores": [], "ranks": []}

        page_embeddings = self.index[doc_id]["embeddings"]
        doc_page_ids = self.index[doc_id]["page_ids"]
        filtered_page_ids: list[int] = []
        filtered_embeddings: list[Any] = []
        for page_id, embedding in zip(doc_page_ids, page_embeddings):
            if int(page_id) in candidate_page_set:
                filtered_page_ids.append(int(page_id))
                filtered_embeddings.append(embedding)
        if not filtered_page_ids:
            return {"page_ids": [], "scores": [], "ranks": []}

        self._ensure_engine_runtime()
        query_embedding = self._encode_query_embedding(question)
        scores = self._score_query_against_document(query_embedding, filtered_embeddings)
        effective_topk = len(filtered_page_ids) if topk is None else min(int(topk), len(filtered_page_ids))
        return topk_from_scores(filtered_page_ids, scores, effective_topk)

    def _ensure_engine_runtime(self) -> None:
        """Lazily load the offline ColQwen model and processor from the local directory."""
        if self._engine_loaded:
            return
        self._ensure_colqwen_engine_available()
        model_path, base_model_path, processor_path = self._resolve_runtime_paths()

        import torch
        from peft import PeftModel
        from transformers.utils.import_utils import is_flash_attn_2_available
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else None
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": self.local_files_only,
            "device_map": self.device,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        processor_kwargs = {
            "local_files_only": self.local_files_only,
        }
        if base_model_path is None:
            self.model = ColQwen2_5.from_pretrained(str(model_path), **model_kwargs).eval()
            processor_source = processor_path or model_path
            loaded_from = str(model_path)
        else:
            self.model = ColQwen2_5.from_pretrained(str(base_model_path), **model_kwargs).eval()
            self.model = PeftModel.from_pretrained(
                self.model,
                str(model_path),
                local_files_only=self.local_files_only,
            ).eval()
            processor_source = processor_path or base_model_path
            loaded_from = f"{model_path} (adapter) + {base_model_path} (base)"
        if hasattr(self.model, "to") and "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        self.processor = ColQwen2_5_Processor.from_pretrained(str(processor_source), **processor_kwargs)
        self.torch = torch
        self._engine_loaded = True
        self.logger.info("Loaded ColQwen model from %s on %s", loaded_from, self.device)

    @staticmethod
    def _ensure_colqwen_engine_available() -> None:
        """Validate that offline ColQwen retrieval dependencies are installed."""
        try:
            import torch  # noqa: F401
            import colpali_engine  # noqa: F401
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor  # noqa: F401
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "colpali-engine with ColQwen support is required for the ColQwen visual-only baseline."
            ) from exc
        import colpali_engine
        import transformers

        engine_version_str = getattr(colpali_engine, "__version__", None)
        if not engine_version_str:
            for dist_name in ("colpali-engine", "colpali_engine", "colpali"):
                try:
                    engine_version_str = importlib_metadata.version(dist_name)
                    break
                except importlib_metadata.PackageNotFoundError:
                    continue
        engine_version = Version(engine_version_str or "0.0.0")
        transformers_version = Version(getattr(transformers, "__version__", "0.0.0"))
        if engine_version < Version("0.3.7"):
            raise RuntimeError(
                f"ColQwen2.5 requires colpali-engine==0.3.7 or newer, but found {engine_version}."
            )
        if transformers_version <= Version("4.45.0"):
            raise RuntimeError(
                f"ColQwen2.5 requires transformers>4.45.0, but found {transformers_version}."
            )

    @staticmethod
    def _is_complete_model_dir(path: Path) -> bool:
        """Return True when the directory looks like a full model checkpoint."""
        return (path / "config.json").exists()

    @staticmethod
    def _is_adapter_dir(path: Path) -> bool:
        """Return True when the directory looks like a PEFT adapter checkpoint."""
        return (path / "adapter_config.json").exists()

    def _resolve_runtime_paths(self) -> tuple[Path, Path | None, Path | None]:
        """Resolve the runtime model, optional base model, and processor source paths."""
        model_path = Path(self.model_name)
        if self.local_files_only and not model_path.exists():
            raise FileNotFoundError(f"Offline ColQwen model directory not found: {self.model_name}")
        if self._is_complete_model_dir(model_path):
            return model_path, None, model_path
        if not self._is_adapter_dir(model_path):
            raise FileNotFoundError(
                "Offline ColQwen model directory is missing both config.json and adapter_config.json: "
                f"{self.model_name}"
            )

        base_model_name = self.base_model_name or self._read_base_model_name_from_adapter(model_path)
        if not base_model_name:
            raise ValueError(
                "The configured ColQwen checkpoint is adapter-only but no base model path is available: "
                f"{self.model_name}. Set visual_colqwen_retrieval.base_model_name or provide "
                "base_model_name_or_path in adapter_config.json."
            )
        base_model_path = Path(self._resolve_model_path(base_model_name))
        if self.local_files_only and not base_model_path.exists():
            raise FileNotFoundError(
                "Offline ColQwen base model directory not found for adapter checkpoint: "
                f"{base_model_path}"
            )
        if self.local_files_only and not self._is_complete_model_dir(base_model_path):
            raise FileNotFoundError(
                "Offline ColQwen base model directory is missing config.json: "
                f"{base_model_path}"
            )
        processor_path = model_path if self._has_processor_assets(model_path) else base_model_path
        return model_path, base_model_path, processor_path

    @staticmethod
    def _has_processor_assets(path: Path) -> bool:
        """Return True when the directory contains enough processor/tokenizer files to load locally."""
        expected_files = (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        )
        return any((path / filename).exists() for filename in expected_files)

    @staticmethod
    def _read_base_model_name_from_adapter(path: Path) -> str | None:
        """Read the base model path from adapter_config.json when available."""
        adapter_config_path = path / "adapter_config.json"
        if not adapter_config_path.exists():
            return None
        try:
            with adapter_config_path.open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to read ColQwen adapter_config.json from {adapter_config_path}: {exc}") from exc
        base_model_name = adapter_config.get("base_model_name_or_path")
        if isinstance(base_model_name, str) and base_model_name.strip():
            return base_model_name.strip()
        return None

    def _encode_document_images(
        self,
        image_paths: list[str],
        page_ids: list[int],
    ) -> tuple[list[Any], list[int], list[str]]:
        """Encode one document's pages into ColQwen multi-vector embeddings."""
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")

        images: list[Any] = []
        kept_page_ids: list[int] = []
        kept_image_paths: list[str] = []

        from PIL import Image

        for image_path, page_id in zip(image_paths, page_ids):
            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
                kept_page_ids.append(int(page_id))
                kept_image_paths.append(str(image_path))
            except Exception as exc:
                self.logger.warning("Failed to load page image %s: %s", image_path, exc)

        if not images:
            return [], [], []

        embeddings: list[Any] = []
        with self.torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch_images = images[start : start + self.batch_size]
                processed_batch = self._process_images(batch_images)
                processed_batch = self._move_batch_to_device(processed_batch)
                batch_embeddings = self.model(**processed_batch)
                if hasattr(batch_embeddings, "embeddings"):
                    batch_embeddings = batch_embeddings.embeddings
                embeddings.extend(list(self.torch.unbind(batch_embeddings.to("cpu"))))

        return embeddings, kept_page_ids, kept_image_paths

    def _encode_query_embedding(self, question: str) -> Any:
        """Encode one query into a ColQwen multi-vector query embedding tensor."""
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")
        with self.torch.no_grad():
            processed_query = self._process_queries([question])
            processed_query = self._move_batch_to_device(processed_query)
            query_embeddings = self.model(**processed_query)
            if hasattr(query_embeddings, "embeddings"):
                query_embeddings = query_embeddings.embeddings
        return query_embeddings.to("cpu")

    def _score_query_against_document(self, query_embedding: Any, page_embeddings: list[Any]) -> list[float]:
        """Compute late-interaction scores between one query and a document's page embeddings."""
        if self.processor is None or self.torch is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")
        if not page_embeddings:
            return []
        score_fn = getattr(self.processor, "score_multi_vector", None)
        if score_fn is None:
            score_fn = getattr(self.processor, "score_retrieval", None)
        if score_fn is None:
            raise AttributeError("ColQwen processor does not expose score_multi_vector or score_retrieval.")
        scores = score_fn(query_embedding, page_embeddings)
        if hasattr(scores, "detach"):
            scores = scores.detach().to(dtype=self.torch.float32, device="cpu")
        if hasattr(scores, "numpy"):
            scores = scores.numpy()
        scores_array = np.asarray(scores, dtype=float)
        if scores_array.ndim == 2:
            scores_array = scores_array[0]
        return scores_array.tolist()

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move a processor batch to the configured device."""
        if self.torch is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    def _process_images(self, images: list[Any]) -> Any:
        """Process images following the official ColQwen2.5 usage."""
        if self.processor is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")
        return self.processor.process_images(images)

    def _process_queries(self, queries: list[str]) -> Any:
        """Process queries following the official ColQwen2.5 usage."""
        if self.processor is None:
            raise RuntimeError("ColQwen engine runtime is not initialized.")
        return self.processor.process_queries(queries)

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Resolve a configured model path relative to the current working directory if needed."""
        candidate = Path(model_name).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        if candidate.exists():
            return str(candidate.resolve())
        resolved = (Path.cwd() / candidate).resolve()
        return str(resolved)

    @classmethod
    def _resolve_optional_model_path(cls, model_name: Any) -> str | None:
        """Resolve an optional model path while preserving missing values."""
        if model_name is None:
            return None
        value = str(model_name).strip()
        if not value:
            return None
        return cls._resolve_model_path(value)
