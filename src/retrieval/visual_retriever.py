"""Abstract interface for page-level visual retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VisualRetriever(ABC):
    """Abstract visual retriever interface."""

    @abstractmethod
    def build_index(self, documents: dict[str, list[dict[str, Any]]]) -> None:
        """Build or cache page embeddings for the provided documents."""

    @abstractmethod
    def encode_pages(self, pages: list[dict[str, Any]]) -> list[list[float]]:
        """Encode page representations."""

    @abstractmethod
    def encode_query(self, question: str) -> list[float]:
        """Encode a question for retrieval."""

    @abstractmethod
    def retrieve(self, question: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k pages for a question within one document."""
