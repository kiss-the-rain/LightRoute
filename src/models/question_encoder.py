"""Lightweight question encoding for question-aware fusion."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.utils.text_utils import classify_question_type, extract_question_heuristics, question_length_features, tokenize_text


class QuestionEncoder:
    """Lightweight question feature encoder."""

    def __init__(self, embedding_dim: int = 32) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, question: str) -> dict[str, Any]:
        """Return a simple hashed embedding and heuristic feature vector."""
        question_lower = (question or "").lower()
        wh_type = self._wh_type(question_lower)
        rule_features = {
            "question_length": float(len(tokenize_text(question))),
            "question_token_count": float(len(tokenize_text(question))),
            "contains_number": float(bool(re.search(r"\d", question_lower))),
            "contains_year": float(bool(re.search(r"\b(19|20)\d{2}\b", question_lower))),
            "contains_date_cue": float(any(token in question_lower for token in ["date", "day", "month", "year", "when"])),
            "contains_amount_cue": float(any(token in question_lower for token in ["amount", "total", "value", "cost", "price"])),
            "contains_percentage_cue": float(any(token in question_lower for token in ["percent", "percentage", "%", "rate"])),
            "contains_company_cue": float(any(token in question_lower for token in ["company", "corporation", "inc", "ltd"])),
            "contains_person_cue": float(any(token in question_lower for token in ["person", "who", "name", "mr", "mrs", "dr"])),
            "contains_table_cue": float(any(token in question_lower for token in ["table", "rate", "value", "amount", "total", "percent"])),
            "contains_name_cue": float(any(token in question_lower for token in ["name", "company", "person", "university"])),
            "contains_location_cue": float(any(token in question_lower for token in ["where", "location", "address"])),
            "contains_form_cue": float(any(token in question_lower for token in ["form", "application", "invoice", "statement", "letter"])),
            "wh_what": float(wh_type == "what"),
            "wh_where": float(wh_type == "where"),
            "wh_who": float(wh_type == "who"),
            "wh_when": float(wh_type == "when"),
            "wh_how_many": float(wh_type == "how_many"),
            "wh_how": float(wh_type == "how"),
            "wh_which": float(wh_type == "which"),
        }
        return {
            "question_embedding": self._embed(question),
            "question_feature_vector": self._feature_vector(question),
            **rule_features,
            **question_length_features(question),
            **extract_question_heuristics(question),
            "wh_type": wh_type,
            "question_type": classify_question_type(question),
        }

    def _embed(self, question: str) -> list[float]:
        vector = [0.0] * self.embedding_dim
        for token in tokenize_text(question):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def _feature_vector(self, question: str) -> list[float]:
        question_lower = (question or "").lower()
        wh_type = self._wh_type(question_lower)
        features = {
            "question_length": float(len(tokenize_text(question))),
            "question_token_count": float(len(tokenize_text(question))),
            "contains_number": float(bool(re.search(r"\d", (question or "").lower()))),
            "contains_year": float(bool(re.search(r"\b(19|20)\d{2}\b", (question or "").lower()))),
            "contains_date_cue": float(any(token in question_lower for token in ["date", "day", "month", "year", "when"])),
            "contains_amount_cue": float(any(token in question_lower for token in ["amount", "total", "value", "cost", "price"])),
            "contains_percentage_cue": float(any(token in question_lower for token in ["percent", "percentage", "%", "rate"])),
            "contains_company_cue": float(any(token in question_lower for token in ["company", "corporation", "inc", "ltd"])),
            "contains_person_cue": float(any(token in question_lower for token in ["person", "who", "name", "mr", "mrs", "dr"])),
            "contains_table_cue": float(any(token in (question or "").lower() for token in ["table", "rate", "value", "amount", "total", "percent"])),
            "contains_name_cue": float(any(token in (question or "").lower() for token in ["name", "company", "person", "university"])),
            "contains_location_cue": float(any(token in (question or "").lower() for token in ["where", "location", "address"])),
            "contains_form_cue": float(any(token in question_lower for token in ["form", "application", "invoice", "statement", "letter"])),
            "wh_what": float(wh_type == "what"),
            "wh_where": float(wh_type == "where"),
            "wh_who": float(wh_type == "who"),
            "wh_when": float(wh_type == "when"),
            "wh_how_many": float(wh_type == "how_many"),
            "wh_how": float(wh_type == "how"),
            "wh_which": float(wh_type == "which"),
            **question_length_features(question),
            **extract_question_heuristics(question),
        }
        return [float(value) for value in features.values()]

    @staticmethod
    def _wh_type(question_lower: str) -> str:
        """Infer a lightweight WH-type category."""
        if question_lower.startswith("how many") or question_lower.startswith("how much"):
            return "how_many"
        if question_lower.startswith("what"):
            return "what"
        if question_lower.startswith("where"):
            return "where"
        if question_lower.startswith("who"):
            return "who"
        if question_lower.startswith("when"):
            return "when"
        if question_lower.startswith("which"):
            return "which"
        if question_lower.startswith("how"):
            return "how"
        return "other"
