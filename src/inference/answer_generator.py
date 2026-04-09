"""Answer generation interface with a lightweight placeholder implementation."""

from __future__ import annotations

from typing import Any


class AnswerGenerator:
    """Fixed answer generator interface."""

    def generate_answer(self, question: str, evidence_pages: list[dict[str, Any]]) -> str:
        """Generate an answer from retrieved evidence pages."""
        if not evidence_pages:
            return ""
        best_page = evidence_pages[0]
        text = str(best_page.get("full_text", "")).strip()
        return text[:200]
