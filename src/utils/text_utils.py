"""Text cleanup and heuristic feature extraction utilities."""

from __future__ import annotations

import re
from typing import Any

QUESTION_KEYWORDS = {
    "digit": ["how many", "number", "amount", "total"],
    "date": ["when", "date", "year", "month", "day"],
    "table": ["table", "row", "column"],
    "chart": ["chart", "graph", "plot"],
    "figure": ["figure", "diagram", "illustration"],
}


def clean_ocr_text(text: str) -> str:
    """Normalize OCR text spacing while preserving content."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def tokenize_text(text: str) -> list[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def detect_question_keywords(question: str) -> dict[str, int]:
    """Detect heuristic keyword groups in a question."""
    question_lower = (question or "").lower()
    return {
        key: int(any(keyword in question_lower for keyword in keywords))
        for key, keywords in QUESTION_KEYWORDS.items()
    }


def question_length_features(question: str) -> dict[str, int]:
    """Return simple length features for a question."""
    tokens = tokenize_text(question)
    return {
        "question_char_len": len(question or ""),
        "question_token_len": len(tokens),
    }


def extract_question_heuristics(question: str) -> dict[str, Any]:
    """Extract heuristic flags for digits, dates, tables, charts, and figures."""
    keyword_flags = detect_question_keywords(question)
    question_lower = (question or "").lower()
    return {
        **keyword_flags,
        "has_digit_token": int(bool(re.search(r"\d", question_lower))),
        "has_date_pattern": int(bool(re.search(r"\b\d{4}\b", question_lower))),
        "is_wh_question": int(question_lower.strip().startswith(("what", "which", "when", "where", "who", "why", "how"))),
    }


def classify_question_type(question: str) -> str:
    """Assign a lightweight heuristic question type."""
    features = extract_question_heuristics(question)
    if features["table"]:
        return "table_lookup"
    if features["chart"]:
        return "chart_reasoning"
    if features["figure"]:
        return "figure_reference"
    if features["date"]:
        return "date_query"
    if features["digit"] or features["has_digit_token"]:
        return "numeric_query"
    return "generic"
