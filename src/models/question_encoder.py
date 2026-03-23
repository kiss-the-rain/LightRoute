"""Question encoding utilities reserved for Phase 2 question-aware fusion."""

from __future__ import annotations

from typing import Any

from utils.text_utils import classify_question_type, extract_question_heuristics, question_length_features


class QuestionEncoder:
    """Lightweight question feature encoder."""

    def encode(self, question: str) -> dict[str, Any]:
        return {
            **question_length_features(question),
            **extract_question_heuristics(question),
            "question_type": classify_question_type(question),
            "question_embedding": [],
        }
