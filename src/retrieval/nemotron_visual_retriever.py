"""Local Nemotron-Colembed visual retriever for ViDoRe Energy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.nemotron_runtime import NemotronPageEmbeddingStore
from src.utils.logger import get_logger
from src.utils.memory_utils import log_ram, release_memory
from src.utils.rank_utils import topk_from_scores


class NemotronVisualRetriever:
    """Visual-only retriever backed by a local nemotron-colembed-vl checkpoint."""

    CACHE_VERSION = "nemotron_visual_qwen3_image_prompt_v2"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        local_files_only: bool = True,
        cache_dir: str | Path = "outputs/cache/visual_nemotron_energy",
        cache_namespace: str = "default",
        enable_cache: bool = True,
        batch_size: int = 4,
        query_batch_size: int | None = None,
        page_batch_size: int | None = None,
        enable_cache_save: bool = True,
        in_memory_cache_limit: int = 128,
        use_lazy_image_decode: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.enable_cache = bool(enable_cache)
        self.batch_size = int(batch_size)
        self.query_batch_size = int(query_batch_size or batch_size)
        self.page_batch_size = int(page_batch_size or batch_size)
        self.enable_cache_save = bool(enable_cache_save)
        self.use_lazy_image_decode = bool(use_lazy_image_decode)
        self.cache_dir = Path(cache_dir)
        self.cache_namespace = str(cache_namespace or "default")
        self.logger = get_logger("nemotron_visual_retriever")
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedding_store = NemotronPageEmbeddingStore(
            self.cache_dir / self.cache_namespace / "image_embedding_shards",
            enabled=self.enable_cache,
            allow_save=self.enable_cache_save,
            in_memory_limit=int(in_memory_cache_limit),
        )

    def load(self) -> None:
        """Load model and processor from local disk only."""
        if self.model is not None and self.processor is not None:
            return
        model_dir = Path(self.model_path)
        if self.local_files_only and not model_dir.exists():
            raise FileNotFoundError(f"Local Nemotron visual model path does not exist: {model_dir}")
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        self._validate_processor_runtime(transformers, model_dir)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
        except ImportError as exc:
            if "Qwen3VLProcessor" in str(exc):
                raise ImportError(self._qwen3vl_processor_error(transformers)) from exc
            raise
        model_kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if self.device.startswith("cuda"):
            model_kwargs["device_map"] = self.device
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                **model_kwargs,
            ).eval()
        except TypeError:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()
            self.model = self.model.to(self.device)
        self.torch = torch
        self._validate_retrieval_runtime()
        self.logger.info("Loaded nemotron visual model from %s on %s", self.model_path, self.device)

    def _validate_processor_runtime(self, transformers_module: Any, model_dir: Path) -> None:
        """Fail early when local model code needs Qwen3VLProcessor but transformers is too old."""
        if hasattr(transformers_module, "Qwen3VLProcessor"):
            return
        if self._local_model_requires_qwen3vl_processor(model_dir):
            raise ImportError(self._qwen3vl_processor_error(transformers_module))

    @staticmethod
    def _local_model_requires_qwen3vl_processor(model_dir: Path) -> bool:
        """Detect local remote-code processors that import transformers.Qwen3VLProcessor."""
        if not model_dir.exists():
            return False
        for path in model_dir.glob("*.py"):
            try:
                if "Qwen3VLProcessor" in path.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except Exception:
                continue
        for path in model_dir.glob("**/*.py"):
            if path.parent == model_dir:
                continue
            try:
                if "Qwen3VLProcessor" in path.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except Exception:
                continue
        return False

    def _qwen3vl_processor_error(self, transformers_module: Any) -> str:
        version = str(getattr(transformers_module, "__version__", "unknown"))
        return (
            "The local nemotron-colembed-vl-8b-v2 processor requires transformers.Qwen3VLProcessor, "
            f"but the active transformers package does not expose it. active_transformers_version={version}. "
            "This is an environment/model-runtime compatibility issue, not a ViDoRe loader issue. "
            "Run this branch in an environment with a transformers version that includes Qwen3VLProcessor "
            "or install the exact transformers revision required by the local model repository. "
            "After updating the environment, rerun: "
            "python -m src.main --mode train_adaptive_fusion_visual_nemotron_ocr_energy --device cuda:0"
        )

    def retrieve(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        topk: int,
    ) -> dict[str, list]:
        """Rank candidate page images for one query."""
        if not candidates:
            return {"page_ids": [], "scores": [], "ranks": []}
        if len(candidates) >= 2000:
            self.logger.warning(
                "Nemotron retrieval is processing a large candidate set: num_candidates=%d device=%s",
                len(candidates),
                self.device,
            )
        self.load()
        query_embedding = self.encode_query(question)
        page_ids: list[int] = []
        image_embeddings: list[Any] = [None] * len(candidates)
        pending_images: list[Any] = []
        pending_positions: list[int] = []
        pending_cache_keys: list[str] = []
        effective_batch_size = max(1, int(self.page_batch_size))

        def flush_pending_batch() -> None:
            if not pending_images:
                return
            log_ram(self.logger, "before_nemotron_page_encode_batch")
            new_embeddings = self.encode_images(pending_images)
            for position, cache_key, embedding in zip(pending_positions, pending_cache_keys, new_embeddings, strict=False):
                image_embeddings[position] = embedding
                if self.enable_cache:
                    self.embedding_store.set(cache_key, embedding)
                    self.cache_misses += 1
            release_memory(new_embeddings, pending_images[:])
            pending_images.clear()
            pending_positions.clear()
            pending_cache_keys.clear()
            log_ram(self.logger, "after_nemotron_page_encode_batch")

        for index, candidate in enumerate(candidates):
            page_id = int(candidate.get("page_id", len(page_ids)))
            cache_key = self._candidate_cache_key(candidate)
            cached_embedding = self.embedding_store.get(cache_key) if self.enable_cache else None
            if cached_embedding is not None:
                image_embeddings[index] = cached_embedding
                self.cache_hits += 1
            else:
                pending_images.append(self._load_candidate_image(candidate))
                pending_positions.append(index)
                pending_cache_keys.append(cache_key)
                if len(pending_images) >= effective_batch_size:
                    flush_pending_batch()
            page_ids.append(page_id)
        flush_pending_batch()
        if any(embedding is None for embedding in image_embeddings):
            raise RuntimeError("Nemotron image embedding construction failed: some candidate embeddings are missing.")
        scores = self.score(query_embedding, image_embeddings)
        release_memory(query_embedding)
        # Nemotron retrieval scores follow standard retrieval semantics: larger is better.
        # topk_from_scores sorts in descending score order.
        return topk_from_scores(page_ids, scores, min(topk, len(page_ids)))

    def encode_query(self, question: str) -> Any:
        return self.encode_queries([question])[0]

    def encode_queries(self, questions: list[str]) -> list[Any]:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("Nemotron visual retriever is not loaded.")
        if hasattr(self.model, "forward_queries"):
            log_ram(self.logger, "before_nemotron_query_encode")
            with self.torch.inference_mode():
                embedding = self.model.forward_queries(questions, batch_size=self.query_batch_size)
            log_ram(self.logger, "after_nemotron_query_encode")
            if isinstance(embedding, self.torch.Tensor):
                return [embedding[index].detach().cpu() for index in range(len(questions))]
            return [embedding[index] for index in range(len(questions))]
        with self.torch.no_grad():
            if hasattr(self.processor, "process_queries"):
                batch = self.processor.process_queries(questions)
                batch = self._move_batch_to_device(batch)
                output = self.model(**batch)
            else:
                batch = self.processor(text=questions, return_tensors="pt", padding=True, truncation=True)
                batch = self._move_batch_to_device(batch)
                output = self.model(**batch)
        embeddings = self._to_numpy_embedding(output)
        return [embeddings[index] for index in range(len(questions))]

    def encode_image_candidate(self, candidate: dict[str, Any]) -> Any:
        return self.encode_images([self._load_candidate_image(candidate)])[0]

    def _load_candidate_image(self, candidate: dict[str, Any]) -> Any:
        image = candidate.get("image")
        image_bytes = candidate.get("image_bytes")
        if image is None and image_bytes is not None:
            from io import BytesIO
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as loaded:
                image = loaded.convert("RGB")
        if image is None:
            from PIL import Image

            image_path = candidate.get("image_path")
            if not image_path:
                raise ValueError(f"Candidate is missing image/image_path: {candidate}")
            with Image.open(str(image_path)) as loaded:
                image = loaded.convert("RGB")
        elif image.__class__.__module__.startswith("PIL."):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported in-memory image type: {type(image).__name__}")
        return image

    def encode_images(self, images: list[Any]) -> list[Any]:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("Nemotron visual retriever is not loaded.")
        if hasattr(self.model, "forward_images"):
            with self.torch.inference_mode():
                embeddings = self.model.forward_images(images, batch_size=self.page_batch_size)
            if isinstance(embeddings, self.torch.Tensor):
                return [embeddings[index].detach().cpu() for index in range(len(images))]
            return [embeddings[index] for index in range(len(images))]
        with self.torch.no_grad():
            if hasattr(self.processor, "process_images"):
                batch = self.processor.process_images(images)
                batch = self._move_batch_to_device(batch)
                output = self.model(**batch)
            else:
                batch = self.processor(
                    text=self._image_text_prompts(len(images)),
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                batch = self._move_batch_to_device(batch)
                output = self.model(**batch)
        embeddings = self._to_numpy_embedding(output)
        return [embeddings[index] for index in range(len(images))]

    def _image_text_prompts(self, num_images: int) -> list[str]:
        """Build one-image prompts for processors that require text alongside images."""
        image_token = str(getattr(self.processor, "image_token", "") or "").strip()
        if not image_token:
            image_token = "<|image_pad|>"
        return [image_token for _ in range(num_images)]

    def score(self, query_embedding: Any, image_embeddings: list[Any]) -> list[float]:
        """Score query against image embeddings with processor scorer or local fallback."""
        if self.model is not None and hasattr(self.model, "get_scores") and self.torch is not None:
            try:
                query_tensor = self._ensure_cpu_tensor(query_embedding)
                image_tensors = [self._ensure_cpu_tensor(embedding) for embedding in image_embeddings]
                scores = self.model.get_scores([query_tensor], image_tensors, batch_size=self.query_batch_size)
                if isinstance(scores, self.torch.Tensor):
                    return [float(value) for value in scores[0].detach().cpu().tolist()]
            except Exception as exc:
                self.logger.warning("Nemotron get_scores failed; falling back to local scorer. error=%s", exc)
        if self.processor is not None and hasattr(self.processor, "score_multi_vector") and self.torch is not None:
            try:
                query_tensor = self.torch.as_tensor(np.asarray([self._embedding_to_numpy(query_embedding)]), device=self.device)
                image_tensor = self.torch.as_tensor(np.asarray([self._embedding_to_numpy(item) for item in image_embeddings]), device=self.device)
                scores = self.processor.score_multi_vector(query_tensor, image_tensor)
                return [float(value) for value in scores.reshape(-1).detach().cpu().tolist()]
            except Exception:
                pass
        return [self._score_pair(query_embedding, image_embedding) for image_embedding in image_embeddings]

    @staticmethod
    def _score_pair(query_embedding: Any, image_embedding: Any) -> float:
        query = np.asarray(query_embedding, dtype=np.float32)
        image = np.asarray(image_embedding, dtype=np.float32)
        if query.ndim == 1 and image.ndim == 1:
            denom = (np.linalg.norm(query) * np.linalg.norm(image)) + 1e-8
            return float(np.dot(query, image) / denom)
        if query.ndim == 2 and image.ndim == 2:
            similarity = query @ image.T
            return float(similarity.max(axis=1).sum())
        query = query.reshape(-1)
        image = image.reshape(-1)
        min_dim = min(query.shape[0], image.shape[0])
        denom = (np.linalg.norm(query[:min_dim]) * np.linalg.norm(image[:min_dim])) + 1e-8
        return float(np.dot(query[:min_dim], image[:min_dim]) / denom)

    def save_cache(self) -> None:
        log_ram(self.logger, "before_nemotron_cache_flush")
        self.embedding_store.flush()
        log_ram(self.logger, "after_nemotron_cache_flush")

    def cache_stats(self) -> dict[str, int]:
        return {
            "embedding_cache_hits": int(self.cache_hits),
            "embedding_cache_misses": int(self.cache_misses),
            **self.embedding_store.stats(),
        }

    def _candidate_cache_key(self, candidate: dict[str, Any]) -> str:
        if candidate.get("image_path"):
            return f"{self.CACHE_VERSION}:path:{Path(str(candidate['image_path'])).resolve()}"
        if candidate.get("image_bytes") is not None:
            return f"{self.CACHE_VERSION}:bytes:{candidate.get('doc_id', '')}:{candidate.get('page_id', '')}:{candidate.get('corpus_id', '')}"
        return f"{self.CACHE_VERSION}:in_memory:{candidate.get('doc_id', '')}:{candidate.get('page_id', '')}"

    def _validate_retrieval_runtime(self) -> None:
        """Confirm that the loaded remote code exposes the official retrieval APIs."""
        missing_methods = [
            method_name
            for method_name in ("forward_queries", "forward_images", "get_scores")
            if not hasattr(self.model, method_name)
        ]
        if missing_methods:
            self.logger.warning(
                "Nemotron remote code is missing official retrieval methods %s; using fallback inference path.",
                missing_methods,
            )
        else:
            self.logger.info(
                "Nemotron retriever will use official APIs: forward_queries, forward_images, get_scores."
            )

    def _ensure_cpu_tensor(self, value: Any) -> Any:
        if self.torch is None:
            return value
        if isinstance(value, self.torch.Tensor):
            return value.detach().cpu()
        return self.torch.as_tensor(np.asarray(value), dtype=self.torch.float32)

    def _embedding_to_numpy(self, value: Any) -> np.ndarray:
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _move_batch_to_device(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
        return batch

    @staticmethod
    def _to_numpy_embedding(output: Any) -> np.ndarray:
        if hasattr(output, "detach"):
            tensor = output
        elif isinstance(output, dict):
            tensor = None
            for key in ("embeddings", "image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
                if key in output and output[key] is not None:
                    tensor = output[key]
                    break
            if tensor is None:
                raise TypeError(f"Unable to find an embedding tensor in output keys: {list(output.keys())}")
        elif hasattr(output, "embeddings"):
            tensor = output.embeddings
        elif hasattr(output, "image_embeds"):
            tensor = output.image_embeds
        elif hasattr(output, "text_embeds"):
            tensor = output.text_embeds
        elif hasattr(output, "pooler_output"):
            tensor = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and output:
            tensor = output[0]
        else:
            raise TypeError(f"Unable to extract embeddings from model output type: {type(output).__name__}")
        if hasattr(tensor, "detach"):
            return tensor.detach().float().cpu().numpy()
        return np.asarray(tensor, dtype=np.float32)
