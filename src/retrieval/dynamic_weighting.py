"""Dynamic OCR/visual weighting helpers for staged fusion experiments."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any


SCORE_FEATURE_NAMES = {
    "ocr_score",
    "normalized_ocr_score",
    "visual_score",
    "normalized_visual_score",
}

NUMERIC_KEYWORDS = {
    "amount",
    "invoice",
    "id",
    "reference",
    "total",
    "price",
    "cost",
    "account",
    "number",
    "reference",
    "serial",
}

VISUAL_KEYWORDS = {
    "chart",
    "figure",
    "trend",
    "curve",
    "layout",
    "region",
    "table",
    "graph",
    "diagram",
}


def _clamp_weights(alpha_v: float, alpha_o: float, min_weight: float, max_weight: float) -> tuple[float, float]:
    alpha_v = min(max(alpha_v, min_weight), max_weight)
    alpha_o = min(max(alpha_o, min_weight), max_weight)
    total = alpha_v + alpha_o
    if total <= 1e-8:
        return 0.5, 0.5
    alpha_v /= total
    alpha_o /= total
    alpha_v = min(max(alpha_v, min_weight), max_weight)
    alpha_o = min(max(alpha_o, min_weight), max_weight)
    total = alpha_v + alpha_o
    return alpha_v / total, alpha_o / total


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
    return float(math.sqrt(max(variance, 0.0)))


def _top_gap(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted((float(value) for value in values), reverse=True)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    return float(sorted_values[0] - sorted_values[1])


def _top_prominence(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted((float(value) for value in values), reverse=True)
    top1 = float(sorted_values[0])
    mean_rest = _safe_mean(sorted_values[1:]) if len(sorted_values) > 1 else 0.0
    return top1 - mean_rest


def _question_signals(sample: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> dict[str, float]:
    question = str(sample.get("question", ""))
    question_lower = question.lower()
    anchor = candidate_rows[0] if candidate_rows else {}
    contains_digits = float(bool(re.search(r"\d", question_lower))) or float(anchor.get("contains_number", 0.0))
    contains_date = float(bool(re.search(r"\b(19|20)\d{2}\b", question_lower))) or float(anchor.get("contains_year", 0.0))
    contains_numeric_cue = float(any(keyword in question_lower for keyword in NUMERIC_KEYWORDS)) or float(anchor.get("contains_amount_cue", 0.0))
    contains_visual_cue = float(any(keyword in question_lower for keyword in VISUAL_KEYWORDS)) or float(anchor.get("contains_table_cue", 0.0))
    contains_id_cue = float(bool(re.search(r"\b(id|ref|reference|invoice|receipt|account)\b", question_lower)))
    query_length = float(anchor.get("question_token_count", anchor.get("question_length", 0.0)))
    return {
        "query_length": query_length,
        "contains_digits": contains_digits,
        "contains_date_like": contains_date,
        "contains_numeric_cue": contains_numeric_cue,
        "contains_visual_cue": contains_visual_cue,
        "contains_id_cue": contains_id_cue,
    }


def _page_signals(candidate_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not candidate_rows:
        return {
            "ocr_token_count_mean": 0.0,
            "ocr_non_empty_ratio": 0.0,
            "ocr_valid_char_ratio_mean": 0.0,
            "digit_density_mean": 0.0,
            "text_density_mean": 0.0,
            "ocr_reliability_mean": 0.0,
        }
    token_counts = [float(row.get("ocr_token_count", 0.0)) for row in candidate_rows]
    empty_flags = [float(row.get("ocr_empty_like_flag", 0.0)) for row in candidate_rows]
    reliabilities = [float(row.get("ocr_reliability", 0.0)) for row in candidate_rows]
    digit_density = [float(row.get("ocr_digit_ratio", 0.0)) for row in candidate_rows]
    non_alnum = [float(row.get("ocr_non_alnum_ratio", 0.0)) for row in candidate_rows]
    non_empty_ratio = 1.0 - _safe_mean(empty_flags)
    valid_char_ratio = 1.0 - min(max(_safe_mean(non_alnum), 0.0), 1.0)
    text_density = min(max(_safe_mean(token_counts) / 120.0, 0.0), 1.0)
    return {
        "ocr_token_count_mean": _safe_mean(token_counts),
        "ocr_non_empty_ratio": non_empty_ratio,
        "ocr_valid_char_ratio_mean": valid_char_ratio,
        "digit_density_mean": _safe_mean(digit_density),
        "text_density_mean": text_density,
        "ocr_reliability_mean": _safe_mean(reliabilities),
    }


def _confidence_signals(candidate_rows: list[dict[str, Any]]) -> dict[str, float]:
    visual_scores = [float(row.get("visual_score", 0.0)) for row in candidate_rows]
    ocr_scores = [float(row.get("ocr_score", 0.0)) for row in candidate_rows]
    return {
        "visual_top_gap": _top_gap(visual_scores),
        "ocr_top_gap": _top_gap(ocr_scores),
        "visual_top_prominence": _top_prominence(visual_scores),
        "ocr_top_prominence": _top_prominence(ocr_scores),
        "visual_score_std": _safe_std(visual_scores),
        "ocr_score_std": _safe_std(ocr_scores),
    }


def build_weight_buckets(query_signals: dict[str, float], page_signals: dict[str, float]) -> dict[str, str]:
    if query_signals.get("contains_visual_cue", 0.0) > 0.5:
        query_bucket = "visual_query"
    elif (
        query_signals.get("contains_digits", 0.0) > 0.5
        or query_signals.get("contains_date_like", 0.0) > 0.5
        or query_signals.get("contains_numeric_cue", 0.0) > 0.5
        or query_signals.get("contains_id_cue", 0.0) > 0.5
    ):
        query_bucket = "ocr_query"
    else:
        query_bucket = "generic_query"

    if page_signals.get("text_density_mean", 0.0) >= 0.5 and page_signals.get("ocr_reliability_mean", 0.0) >= 0.5:
        page_bucket = "text_dense"
    elif page_signals.get("ocr_non_empty_ratio", 0.0) <= 0.4:
        page_bucket = "ocr_sparse"
    else:
        page_bucket = "mixed_page"
    return {"query_bucket": query_bucket, "page_bucket": page_bucket}


def compute_rule_based_weights(
    candidate_rows: list[dict[str, Any]],
    sample: dict[str, Any],
    variant: str,
    cfg: Any,
) -> tuple[float, float, dict[str, Any]]:
    """Compute sample-level OCR/visual weights with conservative bounded rules."""
    dynamic_cfg = getattr(cfg, "dynamic_fusion", {})
    min_weight = float(getattr(dynamic_cfg, "min_weight", 0.2))
    max_weight = float(getattr(dynamic_cfg, "max_weight", 0.8))
    query_scale = float(getattr(dynamic_cfg, "query_bias_scale", 0.18))
    page_scale = float(getattr(dynamic_cfg, "page_correction_scale", 0.08))
    confidence_scale = float(getattr(dynamic_cfg, "confidence_correction_scale", 0.08))

    query_signals = _question_signals(sample, candidate_rows)
    page_signals = _page_signals(candidate_rows)
    confidence_signals = _confidence_signals(candidate_rows)

    alpha_o = 0.5
    alpha_v = 0.5

    query_bias = 0.0
    if query_signals["contains_visual_cue"] > 0.5:
        query_bias -= query_scale
    if (
        query_signals["contains_digits"] > 0.5
        or query_signals["contains_date_like"] > 0.5
        or query_signals["contains_numeric_cue"] > 0.5
        or query_signals["contains_id_cue"] > 0.5
    ):
        query_bias += query_scale

    page_correction = 0.0
    if page_signals["text_density_mean"] >= 0.55 and page_signals["ocr_reliability_mean"] >= 0.5:
        page_correction += page_scale
    if page_signals["ocr_non_empty_ratio"] <= 0.4:
        page_correction -= page_scale
    if page_signals["digit_density_mean"] >= 0.12:
        page_correction += page_scale * 0.5

    confidence_delta = (
        confidence_signals["ocr_top_gap"]
        + 0.5 * confidence_signals["ocr_top_prominence"]
        - confidence_signals["visual_top_gap"]
        - 0.5 * confidence_signals["visual_top_prominence"]
    )
    confidence_correction = max(min(confidence_delta * confidence_scale, confidence_scale), -confidence_scale)

    if variant == "query_only":
        alpha_o += query_bias
        alpha_v -= query_bias
    elif variant == "page_only":
        alpha_o += page_correction
        alpha_v -= page_correction
    elif variant == "confidence_only":
        alpha_o += confidence_correction
        alpha_v -= confidence_correction
    elif variant == "combined":
        alpha_o += query_bias
        alpha_v -= query_bias
        alpha_o += page_correction
        alpha_v -= page_correction
        alpha_o += confidence_correction
        alpha_v -= confidence_correction
    else:
        raise ValueError(f"Unsupported rule-based weighting variant: {variant}")

    alpha_v, alpha_o = _clamp_weights(alpha_v, alpha_o, min_weight=min_weight, max_weight=max_weight)
    return alpha_v, alpha_o, {
        "rule_variant_name": variant,
        "query_signals": query_signals,
        "page_signals": page_signals,
        "confidence_signals": confidence_signals,
        **build_weight_buckets(query_signals, page_signals),
    }


def apply_branch_reweighting(
    candidate_rows: list[dict[str, Any]],
    alpha_v: float,
    alpha_o: float,
) -> list[dict[str, Any]]:
    """Apply sample-level OCR/visual weights to score-like features only."""
    weighted_rows = deepcopy(candidate_rows)
    for row in weighted_rows:
        row.setdefault("raw_ocr_score", float(row.get("ocr_score", 0.0)))
        row.setdefault("raw_normalized_ocr_score", float(row.get("normalized_ocr_score", 0.0)))
        row.setdefault("raw_visual_score", float(row.get("visual_score", 0.0)))
        row.setdefault("raw_normalized_visual_score", float(row.get("normalized_visual_score", 0.0)))
        row["ocr_score"] = float(row["raw_ocr_score"]) * float(alpha_o)
        row["normalized_ocr_score"] = float(row["raw_normalized_ocr_score"]) * float(alpha_o)
        row["visual_score"] = float(row["raw_visual_score"]) * float(alpha_v)
        row["normalized_visual_score"] = float(row["raw_normalized_visual_score"]) * float(alpha_v)
        row["visual_weight"] = float(alpha_v)
        row["ocr_weight"] = float(alpha_o)
    return weighted_rows


def calibrate_route_scores(
    candidate_rows: list[dict[str, Any]],
    option: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Calibrate per-route scores per sample and keep raw values for diagnostics."""
    calibrated_rows = deepcopy(candidate_rows)
    visual_values = [float(row.get("visual_score", 0.0)) for row in calibrated_rows]
    ocr_values = [float(row.get("ocr_score", 0.0)) for row in calibrated_rows]
    visual_calibrated = _calibrate_values(visual_values, option)
    ocr_calibrated = _calibrate_values(ocr_values, option)

    for row, visual_value, ocr_value in zip(calibrated_rows, visual_calibrated, ocr_calibrated):
        row.setdefault("raw_visual_score", float(row.get("visual_score", 0.0)))
        row.setdefault("raw_ocr_score", float(row.get("ocr_score", 0.0)))
        row["calibrated_visual_score"] = float(visual_value)
        row["calibrated_ocr_score"] = float(ocr_value)
        row["normalized_visual_score"] = float(visual_value)
        row["normalized_ocr_score"] = float(ocr_value)

    return calibrated_rows, {
        "visual_pre": _summary_stats(visual_values),
        "visual_post": _summary_stats(visual_calibrated),
        "ocr_pre": _summary_stats(ocr_values),
        "ocr_post": _summary_stats(ocr_calibrated),
    }


def _calibrate_values(values: list[float], option: str) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    if option == "raw":
        return list(values)
    if option == "minmax":
        min_value = min(values)
        max_value = max(values)
        if max_value - min_value <= 1e-8:
            return [1.0 for _ in values]
        return [(value - min_value) / (max_value - min_value + 1e-8) for value in values]
    if option == "zscore":
        mean_value = _safe_mean(values)
        std_value = _safe_std(values)
        if std_value <= 1e-8:
            return [0.0 for _ in values]
        normalized = [(value - mean_value) / std_value for value in values]
        max_abs = max(max(abs(value) for value in normalized), 1.0)
        return [value / max_abs for value in normalized]
    if option == "rank_percentile":
        sorted_pairs = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
        result = [0.0] * len(values)
        denom = max(len(values) - 1, 1)
        for rank, (index, _) in enumerate(sorted_pairs):
            result[index] = 1.0 - float(rank / denom)
        return result
    raise ValueError(f"Unsupported calibration option: {option}")


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_weight_debug(debug_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not debug_items:
        return {}
    avg_visual_weight = _safe_mean([float(item.get("alpha_v", 0.0)) for item in debug_items])
    avg_ocr_weight = _safe_mean([float(item.get("alpha_o", 0.0)) for item in debug_items])
    by_query_bucket: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_page_bucket: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in debug_items:
        query_bucket = str(item.get("query_bucket", "unknown"))
        page_bucket = str(item.get("page_bucket", "unknown"))
        by_query_bucket[query_bucket].append((float(item.get("alpha_v", 0.0)), float(item.get("alpha_o", 0.0))))
        by_page_bucket[page_bucket].append((float(item.get("alpha_v", 0.0)), float(item.get("alpha_o", 0.0))))

    def _bucket_summary(bucket_values: dict[str, list[tuple[float, float]]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for bucket_name, values in bucket_values.items():
            summary[bucket_name] = {
                "avg_visual_weight": _safe_mean([item[0] for item in values]),
                "avg_ocr_weight": _safe_mean([item[1] for item in values]),
            }
        return summary

    collapse_ratio = float(
        sum(1 for item in debug_items if float(item.get("alpha_v", 0.0)) <= 0.21 or float(item.get("alpha_v", 0.0)) >= 0.79)
        / max(len(debug_items), 1)
    )
    return {
        "avg_visual_weight": avg_visual_weight,
        "avg_ocr_weight": avg_ocr_weight,
        "weight_by_query_bucket": _bucket_summary(by_query_bucket),
        "weight_by_page_bucket": _bucket_summary(by_page_bucket),
        "route_collapse_ratio": collapse_ratio,
    }


def derive_gate_targets(
    candidate_rows: list[dict[str, Any]],
    evidence_pages: list[int] | set[int],
    min_weight: float = 0.2,
    max_weight: float = 0.8,
) -> tuple[float, float]:
    """Build a soft OCR/visual target from route-specific gold-page reciprocal ranks."""
    evidence = {int(page_id) for page_id in evidence_pages}
    if not evidence:
        return 0.5, 0.5
    visual_rr = 0.0
    ocr_rr = 0.0
    for row in candidate_rows:
        page_id = int(row.get("page_id", -1))
        if page_id not in evidence:
            continue
        visual_rank = int(row.get("visual_rank", 0) or 0)
        ocr_rank = int(row.get("ocr_rank", 0) or 0)
        visual_rr = max(visual_rr, 1.0 / visual_rank if visual_rank > 0 else 0.0)
        ocr_rr = max(ocr_rr, 1.0 / ocr_rank if ocr_rank > 0 else 0.0)
    if visual_rr <= 0.0 and ocr_rr <= 0.0:
        return 0.5, 0.5
    total = visual_rr + ocr_rr
    alpha_v = visual_rr / total if total > 0 else 0.5
    alpha_o = ocr_rr / total if total > 0 else 0.5
    return _clamp_weights(alpha_v, alpha_o, min_weight=min_weight, max_weight=max_weight)
