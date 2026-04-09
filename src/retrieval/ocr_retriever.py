"""Abstract interface for OCR dense retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OCRRetriever(ABC):
    """Abstract OCR retrieval interface for document-scoped page retrieval."""

    @abstractmethod
    def build_document_index(self, doc_id: str, page_texts: list[str], page_ids: list[int]) -> None:
        """Build and cache one document's OCR retrieval index."""

    @abstractmethod
    def retrieve(self, query: str, doc_id: str, topk: int) -> dict[str, list]:
        """Retrieve top-k page candidates within one document."""
