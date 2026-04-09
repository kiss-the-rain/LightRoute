"""ColPali retriever wrapper with stub and offline transformers-backed retrieval modes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.visual_retriever import VisualRetriever
from src.utils.io_utils import load_pickle, save_pickle
from src.utils.logger import get_logger
from src.utils.rank_utils import topk_from_scores
from src.utils.text_utils import tokenize_text


def _stable_token_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class ColPaliRetriever(VisualRetriever):
    """Visual retriever supporting a cached stub backend and an offline ColPali mode."""

    def __init__(self, cfg: Any, require_engine: bool = False) -> None:
        self.cfg = cfg
        self.require_engine = require_engine
        self.backend_type = "colpali_engine" if require_engine else str(getattr(cfg.visual_retriever, "type", "colpali_stub"))
        self.embedding_dim = int(cfg.visual_retriever.embedding_dim)
        self.cache_path = Path(cfg.visual_retriever.cache_path)
        self.index: dict[str, dict[str, Any]] = {}
        self.model_name = self._resolve_model_path(
            str(getattr(getattr(cfg, "visual", {}), "model_name", "src/models/vidore_colpali_v1.2_hf"))
        )
        self.local_files_only = bool(getattr(getattr(cfg, "visual", {}), "local_files_only", True))
        self.batch_size = int(getattr(getattr(cfg, "visual", {}), "batch_size", 2))
        self.device = self._resolve_device()
        self.logger = get_logger("colpali_retriever")
        self._engine_loaded = False
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def build_index(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build and optionally cache page embeddings by document."""
        if self.backend_type == "colpali_engine":
            for doc_id, pages in documents.items():
                image_paths = [str(page.get("image_path", "")) for page in pages]
                page_ids = [self._page_id_to_int(str(page["page_id"])) for page in pages]
                self.build_document_index(doc_id=doc_id, image_paths=image_paths, page_ids=page_ids)
            return

        if self.cfg.runtime.use_cache and self.cache_path.exists():
            self.index = load_pickle(self.cache_path)
            return

        self.index = {}
        for doc_id, pages in documents.items():
            embeddings = self.encode_pages(pages)
            self.index[doc_id] = {
                "page_ids": [page["page_id"] for page in pages],
                "page_items": pages,
                "embeddings": embeddings,
            }

        if self.cfg.visual_retriever.cache_embeddings:
            save_pickle(self.index, self.cache_path)

    def build_document_index(self, doc_id: str, image_paths: list[str], page_ids: list[int]) -> None:
        """Build and cache one document's page representations for repeated queries."""
        if doc_id in self.index:
            return

        if self.backend_type == "colpali_engine":
            self._ensure_engine_runtime()
            embeddings, kept_page_ids, kept_image_paths = self._encode_document_images(image_paths=image_paths, page_ids=page_ids)
            self.index[doc_id] = {
                "page_ids": kept_page_ids,
                "image_paths": kept_image_paths,
                "embeddings": embeddings,
            }
            return

        pages = [
            {
                "page_id": f"{doc_id}_p{page_id}",
                "doc_id": doc_id,
                "image_path": image_path,
            }
            for page_id, image_path in zip(page_ids, image_paths)
        ]
        embeddings = self.encode_pages(pages)
        self.index[doc_id] = {
            "page_ids": page_ids,
            "page_items": pages,
            "embeddings": embeddings,
        }

    def encode_pages(self, pages: list[dict[str, Any]]) -> list[list[float]]:
        """Encode page surrogate text or image identifiers into normalized vectors."""
        if self.backend_type == "colpali_engine":
            self._ensure_engine_runtime()
            image_paths = [str(page.get("image_path", "")) for page in pages]
            page_ids = [self._page_id_to_int(page["page_id"]) for page in pages]
            embeddings, _, _ = self._encode_document_images(image_paths=image_paths, page_ids=page_ids)
            return [embedding.tolist() for embedding in embeddings]
        return [self._text_to_embedding(self._page_surrogate_text(page)) for page in pages]

    def encode_query(self, question: str) -> list[float]:
        """Encode the query into the same hashed vector space."""
        if self.backend_type == "colpali_engine":
            self._ensure_engine_runtime()
            query_embedding = self._encode_query_embedding(question)
            return query_embedding.reshape(-1).tolist()
        return self._text_to_embedding(question)

    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k pages inside one document."""
        if doc_id not in self.index:
            return {"page_ids": [], "scores": [], "ranks": []}

        if self.backend_type == "colpali_engine":
            self._ensure_engine_runtime()
            query_embedding = self._encode_query_embedding(question)
            scores = self._score_query_against_document(query_embedding, self.index[doc_id]["embeddings"])
            return topk_from_scores(self.index[doc_id]["page_ids"], scores, topk)

        query_embedding = np.asarray(self.encode_query(question), dtype=float)
        page_embeddings = np.asarray(self.index[doc_id]["embeddings"], dtype=float)
        scores = page_embeddings @ query_embedding
        return topk_from_scores(self.index[doc_id]["page_ids"], scores.tolist(), topk)

    def _page_surrogate_text(self, page: dict[str, Any]) -> str:
        if self.cfg.visual_retriever.use_surrogate_text:
            text = page.get("surrogate_text") or page.get("ocr_text") or page.get("image_path", "")
            if text:
                return str(text)
        return Path(page.get("image_path", page.get("page_id", ""))).stem.replace("_", " ")

    def _text_to_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.embedding_dim, dtype=float)
        for token in tokenize_text(text):
            vector[_stable_token_index(token, self.embedding_dim)] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def _ensure_colpali_engine_available(self) -> None:
        """Validate that offline ColPali retrieval dependencies are installed."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from transformers import ColPaliForRetrieval, ColPaliProcessor  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Transformers ColPali dependencies are required for visual-only baseline but are not installed. "
                "Install compatible torch/transformers packages and retry."
            ) from exc

    def _ensure_engine_runtime(self) -> None:
        """Lazily load the offline ColPali model and processor from the local directory."""
        if self._engine_loaded:
            return

        self._ensure_colpali_engine_available()
        model_path = Path(self.model_name)
        if self.local_files_only and not model_path.exists():
            raise FileNotFoundError(
                f"Configured local ColPali model directory does not exist: {self.model_name}"
            )

        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": self.local_files_only,
        }
        processor_kwargs = {
            "local_files_only": self.local_files_only,
            "use_fast": False,
        }

        self.model = ColPaliForRetrieval.from_pretrained(self.model_name, **model_kwargs).eval()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.processor = ColPaliProcessor.from_pretrained(self.model_name, **processor_kwargs)
        self.torch = torch
        self._engine_loaded = True
        self.logger.info("Loaded ColPali model from %s on %s", self.model_name, self.device)

    def _encode_document_images(
        self,
        image_paths: list[str],
        page_ids: list[int],
    ) -> tuple[list[Any], list[int], list[str]]:
        """Encode one document's pages into ColPali multi-vector embeddings."""
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("ColPali engine runtime is not initialized.")

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
                processed_batch = self.processor.process_images(batch_images, return_tensors="pt")
                processed_batch = self._move_batch_to_device(processed_batch)
                batch_embeddings = self.model(**processed_batch).embeddings
                embeddings.extend(list(self.torch.unbind(batch_embeddings.to("cpu"))))

        return embeddings, kept_page_ids, kept_image_paths

    def _encode_query_embedding(self, question: str) -> Any:
        """Encode one query into a ColPali multi-vector query embedding tensor."""
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("ColPali engine runtime is not initialized.")

        with self.torch.no_grad():
            processed_query = self.processor.process_queries([question], return_tensors="pt")
            processed_query = self._move_batch_to_device(processed_query)
            query_embeddings = self.model(**processed_query).embeddings
        return query_embeddings.to("cpu")

    def _score_query_against_document(self, query_embedding: Any, page_embeddings: list[Any]) -> list[float]:
        """Compute late-interaction scores between one query and a document's page embeddings."""
        if self.processor is None or self.torch is None:
            raise RuntimeError("ColPali engine runtime is not initialized.")
        if not page_embeddings:
            return []

        score_fn = getattr(self.processor, "score_retrieval", None)
        if score_fn is None:
            score_fn = getattr(self.processor, "score_multi_vector", None)
        if score_fn is None:
            raise AttributeError("ColPali processor does not expose score_retrieval or score_multi_vector.")

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
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    def _resolve_device(self) -> str:
        """Resolve the requested device string, falling back from CUDA to CPU when necessary."""
        visual_cfg = getattr(self.cfg, "visual", None)
        requested_device = str(getattr(visual_cfg, "device", getattr(self.cfg, "device", "cpu")))
        if requested_device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return requested_device
            except ImportError:
                pass
            return "cpu"
        return requested_device

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Resolve the configured ColPali model directory for offline loading."""
        model_path = Path(model_name)
        if model_path.exists():
            return str(model_path.resolve())
        return model_name

    @staticmethod
    def _page_id_to_int(page_id: str | int) -> int:
        """Convert a page identifier into its integer page index."""
        if isinstance(page_id, int):
            return page_id
        if "_p" in page_id:
            return int(page_id.rsplit("_p", maxsplit=1)[-1])
        return int(page_id)
