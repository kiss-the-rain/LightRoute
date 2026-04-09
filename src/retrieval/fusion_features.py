"""Candidate-level feature construction for adaptive fusion retrieval."""

from __future__ import annotations

import math
from typing import Any

from src.models.question_encoder import QuestionEncoder
from src.utils.io_utils import load_json
from src.utils.text_utils import tokenize_text


FEATURE_ORDER_V1 = [
    "ocr_score",
    "ocr_rank",
    "normalized_ocr_score",
    "reciprocal_ocr_rank",
    "visual_score",
    "visual_rank",
    "normalized_visual_score",
    "reciprocal_visual_rank",
    "overlap_flag",
    "ocr_token_count",
    "ocr_avg_confidence",
    "ocr_reliability",
    "question_length",
    "contains_number",
    "contains_year",
    "contains_table_cue",
    "contains_name_cue",
    "contains_location_cue",
]

QUESTION_AWARE_EXTRA_ORDER = [
    "question_token_count",
    "contains_date_cue",
    "contains_amount_cue",
    "contains_percentage_cue",
    "contains_company_cue",
    "contains_person_cue",
    "contains_form_cue",
    "wh_what",
    "wh_where",
    "wh_who",
    "wh_when",
    "wh_how_many",
    "wh_how",
    "wh_which",
]

OCR_QUALITY_EXTRA_ORDER = [
    "ocr_non_alnum_ratio",
    "ocr_digit_ratio",
    "ocr_short_token_ratio",
    "ocr_empty_like_flag",
]

LEXICAL_EXTRA_ORDER = [
    "question_page_token_overlap",
    "question_page_overlap_count",
    "has_number_match",
    "has_year_match",
    "has_keyword_match",
]

FEATURE_ORDER_ABLATE_Q = FEATURE_ORDER_V1 + QUESTION_AWARE_EXTRA_ORDER
FEATURE_ORDER_ABLATE_OCRQ = FEATURE_ORDER_V1 + OCR_QUALITY_EXTRA_ORDER
FEATURE_ORDER_ABLATE_LEX = FEATURE_ORDER_V1 + LEXICAL_EXTRA_ORDER
FEATURE_ORDER_ABLATE_MLP = FEATURE_ORDER_V1

QUESTION_TYPE_ORDER = [
    "qtype_date_time",
    "qtype_numeric_value",
    "qtype_entity_name",
    "qtype_location",
    "qtype_generic_span",
]

CHUNK_AWARE_ORDER = [
    "ocr_chunk_top1_score",
    "ocr_chunk_top2_mean_score",
    "ocr_chunk_top3_mean_score",
    "ocr_chunk_top1_minus_top2_gap",
    "ocr_chunk_top1_minus_page_score_gap",
    "ocr_chunk_score_std",
    "ocr_chunk_high_conf_count",
]

ROUTE_AGREEMENT_ORDER = [
    "rank_gap_visual_ocr",
    "both_in_topk_flag",
    "visual_top1_is_ocr_topk_flag",
    "ocr_top1_is_visual_topk_flag",
    "visual_ocr_score_gap",
]

FEATURE_ORDER_MLP_OCRQ_CHUNKPLUS = FEATURE_ORDER_ABLATE_OCRQ + QUESTION_TYPE_ORDER + CHUNK_AWARE_ORDER + ROUTE_AGREEMENT_ORDER


def build_candidate_features(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build adaptive-fusion candidate features from OCR and visual retrieval outputs."""
    del ocr_page_texts
    question_encoder = question_encoder or QuestionEncoder()
    ocr_quality_cache = ocr_quality_cache or {}
    doc_id = str(sample["doc_id"])
    question_features = question_encoder.encode(sample.get("question", ""))

    ocr_page_ids = [int(page_id) for page_id in ocr_result.get("page_ids", [])]
    visual_page_ids = [int(page_id) for page_id in visual_result.get("page_ids", [])]
    ocr_score_map = {page_id: float(score) for page_id, score in zip(ocr_page_ids, _normalize_scores(ocr_result.get("scores", [])))}
    visual_score_map = {
        page_id: float(score)
        for page_id, score in zip(visual_page_ids, _normalize_scores(visual_result.get("scores", [])))
    }
    ocr_rank_map = {int(page_id): int(rank) for page_id, rank in zip(ocr_result.get("page_ids", []), ocr_result.get("ranks", []))}
    visual_rank_map = {
        int(page_id): int(rank)
        for page_id, rank in zip(visual_result.get("page_ids", []), visual_result.get("ranks", []))
    }

    union_page_ids = sorted(set(ocr_page_ids) | set(visual_page_ids))
    feature_rows: list[dict[str, Any]] = []
    for page_id in union_page_ids:
        quality = _get_page_ocr_quality(sample, page_id, doc_id, ocr_quality_cache)
        row = {
            "page_id": int(page_id),
            "ocr_score": float(ocr_score_map.get(page_id, 0.0)),
            "ocr_rank": int(ocr_rank_map.get(page_id, 0)),
            "normalized_ocr_score": float(ocr_score_map.get(page_id, 0.0)),
            "reciprocal_ocr_rank": _reciprocal_rank(ocr_rank_map.get(page_id)),
            "visual_score": float(visual_score_map.get(page_id, 0.0)),
            "visual_rank": int(visual_rank_map.get(page_id, 0)),
            "normalized_visual_score": float(visual_score_map.get(page_id, 0.0)),
            "reciprocal_visual_rank": _reciprocal_rank(visual_rank_map.get(page_id)),
            "overlap_flag": float(int(page_id in ocr_rank_map and page_id in visual_rank_map)),
            **quality,
            "question_length": float(question_features.get("question_length", 0.0)),
            "contains_number": float(question_features.get("contains_number", 0.0)),
            "contains_year": float(question_features.get("contains_year", 0.0)),
            "contains_table_cue": float(question_features.get("contains_table_cue", 0.0)),
            "contains_name_cue": float(question_features.get("contains_name_cue", 0.0)),
            "contains_location_cue": float(question_features.get("contains_location_cue", 0.0)),
        }
        row["feature_vector"] = _row_to_feature_vector(row)
        feature_rows.append(row)
    return feature_rows


def _normalize_scores(scores: list[float] | list[Any]) -> list[float]:
    """Apply stable min-max normalization to one route's score list."""
    scores = [float(score) for score in scores]
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score <= 1e-8:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score + 1e-8) for score in scores]


def _safe_std(values: list[float]) -> float:
    """Return a stable standard deviation for a short float list."""
    if not values:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return float(math.sqrt(max(variance, 0.0)))


def _derive_question_type_features(question_features: dict[str, Any]) -> dict[str, float]:
    """Map lightweight question cues into a compact one-hot question-type vector."""
    raw_question_type = str(question_features.get("question_type", "")).strip().lower()
    is_date = bool(
        question_features.get("contains_date_cue", 0.0)
        or question_features.get("contains_year", 0.0)
        or question_features.get("wh_when", 0.0)
        or raw_question_type == "date_query"
    )
    is_numeric = bool(
        question_features.get("contains_amount_cue", 0.0)
        or question_features.get("contains_percentage_cue", 0.0)
        or question_features.get("contains_number", 0.0)
        or question_features.get("wh_how_many", 0.0)
        or raw_question_type == "numeric_query"
    )
    is_entity = bool(
        question_features.get("contains_company_cue", 0.0)
        or question_features.get("contains_person_cue", 0.0)
        or question_features.get("contains_name_cue", 0.0)
    )
    is_location = bool(
        question_features.get("contains_location_cue", 0.0)
        or question_features.get("wh_where", 0.0)
    )

    if is_date:
        selected = "qtype_date_time"
    elif is_numeric:
        selected = "qtype_numeric_value"
    elif is_entity:
        selected = "qtype_entity_name"
    elif is_location:
        selected = "qtype_location"
    else:
        selected = "qtype_generic_span"

    return {feature_name: 1.0 if feature_name == selected else 0.0 for feature_name in QUESTION_TYPE_ORDER}


def _reciprocal_rank(rank: int | None) -> float:
    """Return reciprocal rank with 0 for missing or invalid ranks."""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def _get_page_ocr_quality(
    sample: dict[str, Any],
    page_id: int,
    doc_id: str,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]],
) -> dict[str, float]:
    """Load or reuse OCR quality features for one page."""
    if doc_id in ocr_quality_cache and page_id in ocr_quality_cache[doc_id]:
        return ocr_quality_cache[doc_id][page_id]

    ocr_paths = sample.get("ocr_paths", [])
    if page_id < 0 or page_id >= len(ocr_paths):
        quality = {"ocr_token_count": 0.0, "ocr_avg_confidence": 0.0, "ocr_reliability": 0.0}
    else:
        quality = _extract_ocr_quality_from_json(ocr_paths[page_id])

    ocr_quality_cache.setdefault(doc_id, {})[page_id] = quality
    return quality


def _extract_ocr_quality_from_json(ocr_path: str) -> dict[str, float]:
    """Extract lightweight OCR quality indicators from one page-level OCR JSON."""
    try:
        payload = load_json(ocr_path)
    except Exception:
        return {"ocr_token_count": 0.0, "ocr_avg_confidence": 0.0, "ocr_reliability": 0.0}

    token_texts: list[str] = []
    confidences: list[float] = []
    _collect_ocr_units(payload, token_texts, confidences)
    token_count = float(len([token for token in token_texts if token.strip()]))
    avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    token_factor = min(1.0, token_count / 50.0)
    confidence_factor = avg_confidence if avg_confidence > 0 else 1.0
    reliability = float(min(1.0, token_factor * confidence_factor))
    return {
        "ocr_token_count": token_count,
        "ocr_avg_confidence": avg_confidence,
        "ocr_reliability": reliability,
    }


def _collect_ocr_units(payload: Any, token_texts: list[str], confidences: list[float]) -> None:
    """Recursively collect OCR token texts and confidence values from heterogeneous JSON."""
    if payload is None:
        return
    if isinstance(payload, str):
        token_texts.append(payload)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_ocr_units(item, token_texts, confidences)
        return
    if isinstance(payload, dict):
        text_value = payload.get("text")
        if isinstance(text_value, str) and text_value.strip():
            token_texts.append(text_value)
        for key in ("score", "confidence", "conf"):
            if key in payload:
                try:
                    confidences.append(float(payload[key]))
                except Exception:
                    pass
        for key in ("tokens", "words", "blocks", "lines"):
            if key in payload:
                _collect_ocr_units(payload[key], token_texts, confidences)
        return


def _row_to_feature_vector(row: dict[str, Any]) -> list[float]:
    """Convert one candidate feature row into a numeric vector for adaptive fusion."""
    return [
        float(row["ocr_score"]),
        float(row["ocr_rank"]),
        float(row["normalized_ocr_score"]),
        float(row["reciprocal_ocr_rank"]),
        float(row["visual_score"]),
        float(row["visual_rank"]),
        float(row["normalized_visual_score"]),
        float(row["reciprocal_visual_rank"]),
        float(row["overlap_flag"]),
        float(row["ocr_token_count"]),
        float(row["ocr_avg_confidence"]),
        float(row["ocr_reliability"]),
        float(row["question_length"]),
        float(row["contains_number"]),
        float(row["contains_year"]),
        float(row["contains_table_cue"]),
        float(row["contains_name_cue"]),
        float(row["contains_location_cue"]),
    ]


def build_candidate_features_v2(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build enhanced adaptive-fusion candidate features with route interactions and lexical matches."""
    question_encoder = question_encoder or QuestionEncoder()
    ocr_quality_cache = ocr_quality_cache or {}
    doc_id = str(sample["doc_id"])
    question_features = question_encoder.encode(sample.get("question", ""))
    question_tokens = tokenize_text(sample.get("question", ""))
    question_token_set = set(question_tokens)
    question_numbers = {token for token in question_tokens if token.isdigit()}
    question_years = {token for token in question_numbers if len(token) == 4 and token.startswith(("19", "20"))}
    keyword_tokens = {
        token
        for token in question_tokens
        if token not in {"what", "where", "who", "when", "which", "how", "many", "much", "is", "the", "a", "an", "of", "in"}
    }

    ocr_page_ids = [int(page_id) for page_id in ocr_result.get("page_ids", [])]
    visual_page_ids = [int(page_id) for page_id in visual_result.get("page_ids", [])]
    ocr_norm_scores = _normalize_scores(ocr_result.get("scores", []))
    visual_norm_scores = _normalize_scores(visual_result.get("scores", []))
    ocr_score_map = {page_id: float(score) for page_id, score in zip(ocr_page_ids, ocr_norm_scores)}
    visual_score_map = {page_id: float(score) for page_id, score in zip(visual_page_ids, visual_norm_scores)}
    ocr_rank_map = {int(page_id): int(rank) for page_id, rank in zip(ocr_result.get("page_ids", []), ocr_result.get("ranks", []))}
    visual_rank_map = {int(page_id): int(rank) for page_id, rank in zip(visual_result.get("page_ids", []), visual_result.get("ranks", []))}
    union_page_ids = sorted(set(ocr_page_ids) | set(visual_page_ids))

    feature_rows: list[dict[str, Any]] = []
    for page_id in union_page_ids:
        quality = _get_page_ocr_quality_v2(sample, page_id, doc_id, ocr_quality_cache, ocr_page_texts)
        page_tokens = quality.pop("ocr_page_tokens", [])
        page_token_set = set(page_tokens)
        overlap_count = len(question_token_set & page_token_set)
        overlap_ratio = overlap_count / max(len(question_token_set), 1)
        has_number_match = float(bool(question_numbers & page_token_set))
        has_year_match = float(bool(question_years & page_token_set))
        has_keyword_match = float(bool(keyword_tokens & page_token_set))

        visual_score = float(visual_score_map.get(page_id, 0.0))
        ocr_score = float(ocr_score_map.get(page_id, 0.0))
        visual_rank = int(visual_rank_map.get(page_id, 0))
        ocr_rank = int(ocr_rank_map.get(page_id, 0))
        row = {
            "page_id": int(page_id),
            "ocr_score": ocr_score,
            "ocr_rank": ocr_rank,
            "normalized_ocr_score": ocr_score,
            "reciprocal_ocr_rank": _reciprocal_rank(ocr_rank),
            "visual_score": visual_score,
            "visual_rank": visual_rank,
            "normalized_visual_score": visual_score,
            "reciprocal_visual_rank": _reciprocal_rank(visual_rank),
            "overlap_flag": float(int(page_id in ocr_rank_map and page_id in visual_rank_map)),
            "score_gap": visual_score - ocr_score,
            "abs_score_gap": abs(visual_score - ocr_score),
            "rank_gap": float(visual_rank - ocr_rank) if visual_rank > 0 and ocr_rank > 0 else 0.0,
            "abs_rank_gap": abs(float(visual_rank - ocr_rank)) if visual_rank > 0 and ocr_rank > 0 else 0.0,
            "both_top1_flag": float(int(visual_rank == 1 and ocr_rank == 1)),
            "both_topk_flag": float(int(visual_rank > 0 and ocr_rank > 0)),
            "question_page_token_overlap": float(overlap_ratio),
            "question_page_overlap_count": float(overlap_count),
            "has_number_match": has_number_match,
            "has_year_match": has_year_match,
            "has_keyword_match": has_keyword_match,
            **quality,
            "question_length": float(question_features.get("question_length", 0.0)),
            "question_token_count": float(question_features.get("question_token_count", 0.0)),
            "contains_number": float(question_features.get("contains_number", 0.0)),
            "contains_year": float(question_features.get("contains_year", 0.0)),
            "contains_date_cue": float(question_features.get("contains_date_cue", 0.0)),
            "contains_amount_cue": float(question_features.get("contains_amount_cue", 0.0)),
            "contains_percentage_cue": float(question_features.get("contains_percentage_cue", 0.0)),
            "contains_company_cue": float(question_features.get("contains_company_cue", 0.0)),
            "contains_person_cue": float(question_features.get("contains_person_cue", 0.0)),
            "contains_location_cue": float(question_features.get("contains_location_cue", 0.0)),
            "contains_table_cue": float(question_features.get("contains_table_cue", 0.0)),
            "contains_form_cue": float(question_features.get("contains_form_cue", 0.0)),
            "wh_what": float(question_features.get("wh_what", 0.0)),
            "wh_where": float(question_features.get("wh_where", 0.0)),
            "wh_who": float(question_features.get("wh_who", 0.0)),
            "wh_when": float(question_features.get("wh_when", 0.0)),
            "wh_how_many": float(question_features.get("wh_how_many", 0.0)),
            "wh_how": float(question_features.get("wh_how", 0.0)),
            "wh_which": float(question_features.get("wh_which", 0.0)),
        }
        row["feature_vector"] = _row_to_feature_vector_v2(row)
        feature_rows.append(row)
    return feature_rows


def _get_page_ocr_quality_v2(
    sample: dict[str, Any],
    page_id: int,
    doc_id: str,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]],
    ocr_page_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Load or reuse enhanced OCR quality and lexical features for one page."""
    if doc_id in ocr_quality_cache and page_id in ocr_quality_cache[doc_id]:
        return dict(ocr_quality_cache[doc_id][page_id])

    ocr_paths = sample.get("ocr_paths", [])
    if page_id < 0 or page_id >= len(ocr_paths):
        quality = {
            "ocr_token_count": 0.0,
            "ocr_avg_confidence": 0.0,
            "ocr_reliability": 0.0,
            "ocr_non_alnum_ratio": 0.0,
            "ocr_digit_ratio": 0.0,
            "ocr_short_token_ratio": 0.0,
            "ocr_empty_like_flag": 1.0,
            "ocr_page_tokens": [],
        }
    else:
        quality = _extract_ocr_quality_v2_from_json(ocr_paths[page_id], ocr_page_texts[page_id] if ocr_page_texts and page_id < len(ocr_page_texts) else None)

    ocr_quality_cache.setdefault(doc_id, {})[page_id] = quality
    return dict(quality)


def _extract_ocr_quality_v2_from_json(ocr_path: str, page_text: str | None = None) -> dict[str, Any]:
    """Extract enhanced OCR quality and lexical-support features from one OCR JSON."""
    try:
        payload = load_json(ocr_path)
    except Exception:
        return {
            "ocr_token_count": 0.0,
            "ocr_avg_confidence": 0.0,
            "ocr_reliability": 0.0,
            "ocr_non_alnum_ratio": 0.0,
            "ocr_digit_ratio": 0.0,
            "ocr_short_token_ratio": 0.0,
            "ocr_empty_like_flag": 1.0,
            "ocr_page_tokens": [],
        }

    token_texts: list[str] = []
    confidences: list[float] = []
    _collect_ocr_units(payload, token_texts, confidences)
    if page_text and not token_texts:
        token_texts = tokenize_text(page_text)

    valid_tokens = [token.strip() for token in token_texts if isinstance(token, str) and token.strip()]
    page_tokens = tokenize_text(" ".join(valid_tokens))
    token_count = float(len(valid_tokens))
    avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    token_factor = min(1.0, token_count / 50.0)
    confidence_factor = avg_confidence if avg_confidence > 0 else 1.0
    reliability = float(min(1.0, token_factor * confidence_factor))

    raw_text = " ".join(valid_tokens)
    non_alnum_chars = sum(1 for ch in raw_text if not ch.isalnum() and not ch.isspace())
    non_space_chars = max(sum(1 for ch in raw_text if not ch.isspace()), 1)
    digit_tokens = sum(1 for token in page_tokens if token.isdigit())
    short_tokens = sum(1 for token in page_tokens if len(token) <= 2)
    empty_like_flag = float(token_count < 3)

    return {
        "ocr_token_count": token_count,
        "ocr_avg_confidence": avg_confidence,
        "ocr_reliability": reliability,
        "ocr_non_alnum_ratio": float(non_alnum_chars / non_space_chars),
        "ocr_digit_ratio": float(digit_tokens / max(len(page_tokens), 1)),
        "ocr_short_token_ratio": float(short_tokens / max(len(page_tokens), 1)),
        "ocr_empty_like_flag": empty_like_flag,
        "ocr_page_tokens": page_tokens,
    }


def _row_to_feature_vector_v2(row: dict[str, Any]) -> list[float]:
    """Convert one enhanced candidate row into a numeric vector for adaptive fusion v2."""
    return [
        float(row["ocr_score"]),
        float(row["ocr_rank"]),
        float(row["normalized_ocr_score"]),
        float(row["reciprocal_ocr_rank"]),
        float(row["visual_score"]),
        float(row["visual_rank"]),
        float(row["normalized_visual_score"]),
        float(row["reciprocal_visual_rank"]),
        float(row["overlap_flag"]),
        float(row["score_gap"]),
        float(row["abs_score_gap"]),
        float(row["rank_gap"]),
        float(row["abs_rank_gap"]),
        float(row["both_top1_flag"]),
        float(row["both_topk_flag"]),
        float(row["ocr_token_count"]),
        float(row["ocr_avg_confidence"]),
        float(row["ocr_reliability"]),
        float(row["ocr_non_alnum_ratio"]),
        float(row["ocr_digit_ratio"]),
        float(row["ocr_short_token_ratio"]),
        float(row["ocr_empty_like_flag"]),
        float(row["question_page_token_overlap"]),
        float(row["question_page_overlap_count"]),
        float(row["has_number_match"]),
        float(row["has_year_match"]),
        float(row["has_keyword_match"]),
        float(row["question_length"]),
        float(row["question_token_count"]),
        float(row["contains_number"]),
        float(row["contains_year"]),
        float(row["contains_date_cue"]),
        float(row["contains_amount_cue"]),
        float(row["contains_percentage_cue"]),
        float(row["contains_company_cue"]),
        float(row["contains_person_cue"]),
        float(row["contains_location_cue"]),
        float(row["contains_table_cue"]),
        float(row["contains_form_cue"]),
        float(row["wh_what"]),
        float(row["wh_where"]),
        float(row["wh_who"]),
        float(row["wh_when"]),
        float(row["wh_how_many"]),
        float(row["wh_how"]),
        float(row["wh_which"]),
    ]


def get_feature_order_for_variant(variant: str) -> list[str]:
    """Return the fixed feature order for one adaptive-fusion ablation variant."""
    mapping = {
        "ablate_q": FEATURE_ORDER_ABLATE_Q,
        "ablate_ocrq": FEATURE_ORDER_ABLATE_OCRQ,
        "ablate_lex": FEATURE_ORDER_ABLATE_LEX,
        "ablate_mlp": FEATURE_ORDER_ABLATE_MLP,
    }
    if variant not in mapping:
        raise ValueError(f"Unsupported adaptive-fusion ablation variant: {variant}")
    return list(mapping[variant])


def build_candidate_features_ablate_q(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build v1-base candidates with only extra question-aware features added."""
    return _build_candidate_features_ablation(
        sample,
        ocr_result,
        visual_result,
        variant="ablate_q",
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_ocrq(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build v1-base candidates with only enhanced OCR-quality features added."""
    return _build_candidate_features_ablation(
        sample,
        ocr_result,
        visual_result,
        variant="ablate_ocrq",
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_lex(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build v1-base candidates with only lexical matching features added."""
    return _build_candidate_features_ablation(
        sample,
        ocr_result,
        visual_result,
        variant="ablate_lex",
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_mlp(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build the original v1 candidate features for the MLP-only ablation."""
    return _build_candidate_features_ablation(
        sample,
        ocr_result,
        visual_result,
        variant="ablate_mlp",
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_mlp_q(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build ablate-mlp + question-aware combination features."""
    return build_candidate_features_ablate_q(
        sample,
        ocr_result,
        visual_result,
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_mlp_ocrq(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build ablate-mlp + OCR-quality combination features."""
    return build_candidate_features_ablate_ocrq(
        sample,
        ocr_result,
        visual_result,
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_ablate_mlp_lex(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build ablate-mlp + lexical-overlap combination features."""
    return build_candidate_features_ablate_lex(
        sample,
        ocr_result,
        visual_result,
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def build_candidate_features_mlp_ocrq_chunkplus(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build chunk-aware + question-aware + route-agreement features on top of mlp_ocrq."""
    base_rows = build_candidate_features_ablate_mlp_ocrq(
        sample,
        ocr_result,
        visual_result,
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )
    if not base_rows:
        return []

    question_encoder = question_encoder or QuestionEncoder()
    question_features = question_encoder.encode(sample.get("question", ""))
    question_type_features = _derive_question_type_features(question_features)
    ocr_topk_pages = {int(page_id) for page_id in ocr_result.get("page_ids", [])}
    visual_topk_pages = {int(page_id) for page_id in visual_result.get("page_ids", [])}
    visual_top1_page = int(visual_result.get("page_ids", [0])[0]) if visual_result.get("page_ids") else None
    ocr_top1_page = int(ocr_result.get("page_ids", [0])[0]) if ocr_result.get("page_ids") else None
    page_chunk_scores_raw = ocr_result.get("page_chunk_scores", {}) or {}
    page_chunk_scores = {
        int(page_id): _normalize_scores([float(score) for score in scores])
        for page_id, scores in page_chunk_scores_raw.items()
    }

    enriched_rows: list[dict[str, Any]] = []
    for row in base_rows:
        page_id = int(row["page_id"])
        chunk_scores = sorted(page_chunk_scores.get(page_id, []), reverse=True)
        top1 = float(chunk_scores[0]) if chunk_scores else 0.0
        top2_mean = float(sum(chunk_scores[:2]) / max(min(len(chunk_scores), 2), 1)) if chunk_scores else 0.0
        top3_mean = float(sum(chunk_scores[:3]) / max(min(len(chunk_scores), 3), 1)) if chunk_scores else 0.0
        top2 = float(chunk_scores[1]) if len(chunk_scores) > 1 else 0.0
        score_std = _safe_std(chunk_scores)
        high_conf_threshold = top1 * 0.8 if top1 > 0 else top1
        high_conf_count = float(sum(1 for score in chunk_scores if score >= high_conf_threshold)) if chunk_scores else 0.0
        visual_rank = int(row.get("visual_rank", 0))
        ocr_rank = int(row.get("ocr_rank", 0))

        row.update(
            {
                **question_type_features,
                "ocr_chunk_top1_score": top1,
                "ocr_chunk_top2_mean_score": top2_mean,
                "ocr_chunk_top3_mean_score": top3_mean,
                "ocr_chunk_top1_minus_top2_gap": top1 - top2,
                "ocr_chunk_top1_minus_page_score_gap": top1 - float(row.get("normalized_ocr_score", 0.0)),
                "ocr_chunk_score_std": score_std,
                "ocr_chunk_high_conf_count": high_conf_count,
                "rank_gap_visual_ocr": float(visual_rank - ocr_rank) if visual_rank > 0 and ocr_rank > 0 else 0.0,
                "both_in_topk_flag": float(int(page_id in ocr_topk_pages and page_id in visual_topk_pages)),
                "visual_top1_is_ocr_topk_flag": float(int(visual_top1_page is not None and visual_top1_page in ocr_topk_pages)),
                "ocr_top1_is_visual_topk_flag": float(int(ocr_top1_page is not None and ocr_top1_page in visual_topk_pages)),
                "visual_ocr_score_gap": float(row.get("normalized_visual_score", 0.0) - row.get("normalized_ocr_score", 0.0)),
            }
        )
        row["feature_names"] = FEATURE_ORDER_MLP_OCRQ_CHUNKPLUS
        row["feature_vector"] = _vector_from_feature_order(row, FEATURE_ORDER_MLP_OCRQ_CHUNKPLUS)
        enriched_rows.append(row)
    return enriched_rows


def build_candidate_features_visual_colqwen_ocr_chunk(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build the stable mlp_ocrq feature set using ColQwen visual route + OCR chunk route."""
    return build_candidate_features_ablate_mlp_ocrq(
        sample,
        ocr_result,
        visual_result,
        ocr_page_texts=ocr_page_texts,
        question_encoder=question_encoder,
        ocr_quality_cache=ocr_quality_cache,
    )


def _build_candidate_features_ablation(
    sample: dict[str, Any],
    ocr_result: dict[str, Any],
    visual_result: dict[str, Any],
    variant: str,
    ocr_page_texts: list[str] | None = None,
    question_encoder: QuestionEncoder | None = None,
    ocr_quality_cache: dict[str, dict[int, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build single-factor ablation candidate features on top of the v1 base feature set."""
    question_encoder = question_encoder or QuestionEncoder()
    ocr_quality_cache = ocr_quality_cache or {}
    doc_id = str(sample["doc_id"])
    question_features = question_encoder.encode(sample.get("question", ""))
    question_tokens = tokenize_text(sample.get("question", ""))
    question_token_set = set(question_tokens)
    question_numbers = {token for token in question_tokens if token.isdigit()}
    question_years = {token for token in question_numbers if len(token) == 4 and token.startswith(("19", "20"))}
    keyword_tokens = {
        token
        for token in question_tokens
        if token not in {"what", "where", "who", "when", "which", "how", "many", "much", "is", "the", "a", "an", "of", "in"}
    }
    feature_order = get_feature_order_for_variant(variant)

    ocr_page_ids = [int(page_id) for page_id in ocr_result.get("page_ids", [])]
    visual_page_ids = [int(page_id) for page_id in visual_result.get("page_ids", [])]
    ocr_score_map = {page_id: float(score) for page_id, score in zip(ocr_page_ids, _normalize_scores(ocr_result.get("scores", [])))}
    visual_score_map = {page_id: float(score) for page_id, score in zip(visual_page_ids, _normalize_scores(visual_result.get("scores", [])))}
    ocr_rank_map = {int(page_id): int(rank) for page_id, rank in zip(ocr_result.get("page_ids", []), ocr_result.get("ranks", []))}
    visual_rank_map = {int(page_id): int(rank) for page_id, rank in zip(visual_result.get("page_ids", []), visual_result.get("ranks", []))}

    feature_rows: list[dict[str, Any]] = []
    for page_id in sorted(set(ocr_page_ids) | set(visual_page_ids)):
        base_quality = _get_page_ocr_quality(sample, page_id, doc_id, ocr_quality_cache)
        enhanced_quality = _get_page_ocr_quality_v2(sample, page_id, doc_id, ocr_quality_cache, ocr_page_texts)
        page_tokens = enhanced_quality.get("ocr_page_tokens", [])
        page_token_set = set(page_tokens)
        overlap_count = len(question_token_set & page_token_set)
        row = {
            "page_id": int(page_id),
            "ocr_score": float(ocr_score_map.get(page_id, 0.0)),
            "ocr_rank": int(ocr_rank_map.get(page_id, 0)),
            "normalized_ocr_score": float(ocr_score_map.get(page_id, 0.0)),
            "reciprocal_ocr_rank": _reciprocal_rank(ocr_rank_map.get(page_id)),
            "visual_score": float(visual_score_map.get(page_id, 0.0)),
            "visual_rank": int(visual_rank_map.get(page_id, 0)),
            "normalized_visual_score": float(visual_score_map.get(page_id, 0.0)),
            "reciprocal_visual_rank": _reciprocal_rank(visual_rank_map.get(page_id)),
            "overlap_flag": float(int(page_id in ocr_rank_map and page_id in visual_rank_map)),
            "ocr_token_count": float(base_quality.get("ocr_token_count", 0.0)),
            "ocr_avg_confidence": float(base_quality.get("ocr_avg_confidence", 0.0)),
            "ocr_reliability": float(base_quality.get("ocr_reliability", 0.0)),
            "question_length": float(question_features.get("question_length", 0.0)),
            "contains_number": float(question_features.get("contains_number", 0.0)),
            "contains_year": float(question_features.get("contains_year", 0.0)),
            "contains_table_cue": float(question_features.get("contains_table_cue", 0.0)),
            "contains_name_cue": float(question_features.get("contains_name_cue", 0.0)),
            "contains_location_cue": float(question_features.get("contains_location_cue", 0.0)),
            "question_token_count": float(question_features.get("question_token_count", 0.0)),
            "contains_date_cue": float(question_features.get("contains_date_cue", 0.0)),
            "contains_amount_cue": float(question_features.get("contains_amount_cue", 0.0)),
            "contains_percentage_cue": float(question_features.get("contains_percentage_cue", 0.0)),
            "contains_company_cue": float(question_features.get("contains_company_cue", 0.0)),
            "contains_person_cue": float(question_features.get("contains_person_cue", 0.0)),
            "contains_form_cue": float(question_features.get("contains_form_cue", 0.0)),
            "wh_what": float(question_features.get("wh_what", 0.0)),
            "wh_where": float(question_features.get("wh_where", 0.0)),
            "wh_who": float(question_features.get("wh_who", 0.0)),
            "wh_when": float(question_features.get("wh_when", 0.0)),
            "wh_how_many": float(question_features.get("wh_how_many", 0.0)),
            "wh_how": float(question_features.get("wh_how", 0.0)),
            "wh_which": float(question_features.get("wh_which", 0.0)),
            "ocr_non_alnum_ratio": float(enhanced_quality.get("ocr_non_alnum_ratio", 0.0)),
            "ocr_digit_ratio": float(enhanced_quality.get("ocr_digit_ratio", 0.0)),
            "ocr_short_token_ratio": float(enhanced_quality.get("ocr_short_token_ratio", 0.0)),
            "ocr_empty_like_flag": float(enhanced_quality.get("ocr_empty_like_flag", 0.0)),
            "question_page_token_overlap": float(overlap_count / max(len(question_token_set), 1)),
            "question_page_overlap_count": float(overlap_count),
            "has_number_match": float(bool(question_numbers & page_token_set)),
            "has_year_match": float(bool(question_years & page_token_set)),
            "has_keyword_match": float(bool(keyword_tokens & page_token_set)),
        }
        row["feature_names"] = feature_order
        row["feature_vector"] = _vector_from_feature_order(row, feature_order)
        feature_rows.append(row)
    return feature_rows


def _vector_from_feature_order(row: dict[str, Any], feature_order: list[str]) -> list[float]:
    """Build a numeric vector from a fixed feature order with stable scaling and missing-value handling."""
    return [_scaled_feature_value(name, row.get(name, 0.0)) for name in feature_order]


def _scaled_feature_value(name: str, value: Any) -> float:
    """Apply stable clipping/scaling so one large-magnitude feature cannot dominate the MLP."""
    value = 0.0 if value is None else float(value)
    if name in {"ocr_rank", "visual_rank"}:
        return min(max(value, 0.0), 10.0) / 10.0
    if name in {"rank_gap", "abs_rank_gap", "rank_gap_visual_ocr"}:
        return min(max(value, -10.0), 10.0) / 10.0
    if name == "question_length":
        return min(max(value, 0.0), 64.0) / 64.0
    if name == "question_token_count":
        return min(max(value, 0.0), 32.0) / 32.0
    if name == "ocr_token_count":
        return min(max(value, 0.0), 200.0) / 200.0
    if name == "ocr_chunk_high_conf_count":
        return min(max(value, 0.0), 10.0) / 10.0
    if name == "question_page_overlap_count":
        return min(max(value, 0.0), 10.0) / 10.0
    if name in {
        "ocr_avg_confidence",
        "ocr_reliability",
        "ocr_non_alnum_ratio",
        "ocr_digit_ratio",
        "ocr_short_token_ratio",
        "ocr_empty_like_flag",
        "ocr_chunk_score_std",
        "abs_score_gap",
        "abs_rank_gap",
    }:
        return min(max(value, 0.0), 1.0)
    return min(max(value, -1.0), 1.0)
