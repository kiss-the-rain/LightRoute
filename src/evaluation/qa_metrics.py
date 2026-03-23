"""QA metrics reserved for later end-to-end DocVQA evaluation."""

from __future__ import annotations

import re


def exact_match(prediction: str, answer: str) -> float:
    return float(_normalize(prediction) == _normalize(answer))


def f1_score(prediction: str, answer: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    unmatched_gold = list(gold_tokens)
    for token in pred_tokens:
        if token in unmatched_gold:
            common += 1
            unmatched_gold.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def anls(prediction: str, answer: str) -> float:
    return exact_match(prediction, answer)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
