"""Backward-compatible OCR BM25 retriever wrapper."""

from retrieval.bm25_retriever import BM25Retriever as OCRBM25Retriever

__all__ = ["OCRBM25Retriever"]
