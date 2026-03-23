"""Answer-generator interface reserved for later DocVQA integration."""

from __future__ import annotations

from typing import Any


class AnswerGenerator:
    """Fixed answer generator interface."""

    def generate_answer(self, question: str, evidence_pages: list[dict[str, Any]]) -> str:
        return ""
