"""Backward-compatible visual retriever wrapper."""

from src.retrieval.colpali_retriever import ColPaliRetriever as VisualPageRetriever

__all__ = ["VisualPageRetriever"]
