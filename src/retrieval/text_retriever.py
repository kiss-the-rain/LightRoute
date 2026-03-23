"""Abstract interface for OCR text retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TextRetriever(ABC):
    """Abstract OCR text retriever interface."""

    @abstractmethod
    def build_index(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build the page-level text index."""

    @abstractmethod
    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k OCR text pages for a question within one document."""
