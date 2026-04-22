"""ViDoRe V3 Energy local dataset loader.

This loader intentionally treats the local Energy subset as a ViDoRe-style
dataset with separate queries, corpus, and qrels components. It does not fall
back to flat image-folder guessing because that silently produces empty or
incorrect supervision for this dataset family.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_jsonl
from src.utils.logger import get_logger


COMPONENT_NAMES = ("queries", "corpus", "qrels")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_PATH_KEYS = ("image_path", "path", "page_image", "filename", "file_name")
PAGE_TEXT_KEYS = ("markdown", "ocr_text", "page_text", "document_text", "content", "passage", "text")


def load_vidore_energy_dataset(dataset_path: str | Path, max_samples: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load local ViDoRe V3 Energy samples by explicitly joining queries/corpus/qrels."""
    logger = get_logger("vidore_energy_loader")
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Local ViDoRe Energy dataset path does not exist: {root}")

    components, sources = _load_vidore_components(root)
    queries = components.get("queries", [])
    corpus = components.get("corpus", [])
    qrels = components.get("qrels", [])

    examples = {
        "query": _compact_record_for_log(queries[0]) if queries else None,
        "corpus": _compact_record_for_log(corpus[0]) if corpus else None,
        "qrels": _compact_record_for_log(qrels[0]) if qrels else None,
    }
    logger.info("ViDoRe Energy dataset root: %s", root)
    logger.info(
        "ViDoRe component sources: queries=%s corpus=%s qrels=%s",
        sources.get("queries", ""),
        sources.get("corpus", ""),
        sources.get("qrels", ""),
    )
    logger.info(
        "ViDoRe raw counts: num_queries=%d num_corpus_pages=%d num_qrels=%d",
        len(queries),
        len(corpus),
        len(qrels),
    )
    logger.info("ViDoRe example query record: %s", examples["query"])
    logger.info("ViDoRe example corpus record: %s", examples["corpus"])
    logger.info("ViDoRe example qrels record: %s", examples["qrels"])

    _raise_if_empty_component(root, sources, queries, corpus, qrels, examples)

    samples, join_summary = _build_joined_samples(root, queries, corpus, qrels)
    total_joined_samples = len(samples)
    if total_joined_samples == 0:
        _raise_empty_join(root, sources, queries, corpus, qrels, examples, join_summary)

    if max_samples > 0:
        samples = samples[:max_samples]

    summary = {
        "dataset_path": str(root),
        "source": "vidore_components",
        "queries_source": sources.get("queries", ""),
        "corpus_source": sources.get("corpus", ""),
        "qrels_source": sources.get("qrels", ""),
        "num_queries": len(queries),
        "num_corpus_pages": len(corpus),
        "num_candidate_pages": join_summary["num_candidate_pages"],
        "num_candidate_images": join_summary["num_candidate_pages"],
        "num_qrels": len(qrels),
        "num_total_joined_samples_before_limit": total_joined_samples,
        "num_final_samples": len(samples),
        "max_samples": int(max_samples),
        "avg_positives_per_query": join_summary["avg_positives_per_query"],
        "avg_candidate_set_size": join_summary["avg_candidate_set_size"],
        "num_corpus_pages_with_images": join_summary["num_corpus_pages_with_images"],
        "image_parse_summary": join_summary["image_parse_summary"],
        "image_parse_diagnostics": join_summary["image_parse_diagnostics"],
        "query_field": "queries.query_id / queries.query",
        "gold_field": "qrels.query_id / qrels.corpus_id / qrels.score>0",
        "image_field": join_summary["image_field"],
        "text_field": join_summary["text_field"],
        "example_query_record": examples["query"],
        "example_corpus_record": examples["corpus"],
        "example_qrels_record": examples["qrels"],
    }
    logger.info(
        "ViDoRe joined summary: num_final_samples=%d avg_positives_per_query=%.4f avg_candidate_set_size=%.4f",
        len(samples),
        float(summary["avg_positives_per_query"]),
        float(summary["avg_candidate_set_size"]),
    )
    if samples:
        first = samples[0]
        first_candidate = (first.get("candidates") or [{}])[0]
        logger.info(
            "ViDoRe built sample summary: query_id=%s query=%r positive_corpus_ids=%s example_candidate_page_ids=%s",
            first.get("query_id"),
            str(first.get("query", ""))[:160],
            first.get("positive_corpus_ids", [])[:10],
            first.get("candidate_corpus_ids", [])[:10],
        )
        logger.info(
            "ViDoRe example candidate image source: corpus_id=%s page_id=%s source_type=%s has_pil_image=%s image_path=%s",
            first_candidate.get("corpus_id", ""),
            first_candidate.get("page_id", ""),
            first_candidate.get("image_source_type", ""),
            first_candidate.get("image") is not None,
            first_candidate.get("image_path", ""),
        )
    return samples, summary


def _load_vidore_components(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load queries/corpus/qrels from a DatasetDict or component files/directories."""
    dataset_components, dataset_sources = _try_load_hf_datasetdict_components(root)
    if all(dataset_components.get(name) for name in COMPONENT_NAMES):
        return dataset_components, dataset_sources

    components: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for name in COMPONENT_NAMES:
        records, source = _load_component_records(root, name)
        components[name] = records
        sources[name] = source
    return components, sources


def _try_load_hf_datasetdict_components(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if not _looks_like_hf_dataset(root):
        return {}, {}
    try:
        from datasets import DatasetDict, load_from_disk
    except Exception:
        return {}, {}
    try:
        dataset = load_from_disk(str(root))
    except Exception:
        return {}, {}
    if not isinstance(dataset, DatasetDict):
        return {}, {}
    components: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for name in COMPONENT_NAMES:
        if name in dataset:
            components[name] = [dict(item) for item in dataset[name]]
            sources[name] = f"load_from_disk:{root}:{name}"
    return components, sources


def _load_component_records(root: Path, component_name: str) -> tuple[list[dict[str, Any]], str]:
    candidates = _component_path_candidates(root, component_name)
    for path in candidates:
        records = _load_records_from_path(path)
        if records:
            return records, str(path)
    return [], ""


def _component_path_candidates(root: Path, component_name: str) -> list[Path]:
    paths: list[Path] = []
    direct_dir = root / component_name
    data_dir = root / "data" / component_name
    for path in (direct_dir, data_dir):
        if path.exists():
            paths.append(path)
    for parent in (root, root / "data"):
        if not parent.exists():
            continue
        for suffix in (".parquet", ".jsonl", ".json", ".arrow"):
            paths.extend(sorted(parent.glob(f"{component_name}{suffix}")))
            paths.extend(sorted(parent.glob(f"{component_name}-*{suffix}")))
            paths.extend(sorted(parent.glob(f"{component_name}_*{suffix}")))
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _load_records_from_path(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        if _looks_like_hf_dataset(path):
            records = _load_hf_dataset_records(path)
            if records:
                return records
        for suffix in (".parquet", ".jsonl", ".json", ".arrow"):
            files = sorted(child for child in path.rglob(f"*{suffix}") if child.is_file())
            merged: list[dict[str, Any]] = []
            for file_path in files:
                merged.extend(_load_records_from_file(file_path))
            if merged:
                return merged
        return []
    return _load_records_from_file(path)


def _load_hf_dataset_records(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except Exception:
        return []
    try:
        dataset = load_from_disk(str(path))
    except Exception:
        return []
    if isinstance(dataset, DatasetDict):
        records: list[dict[str, Any]] = []
        for split in dataset.values():
            records.extend(dict(item) for item in split)
        return records
    return [dict(item) for item in dataset]


def _load_records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return [dict(item) for item in load_jsonl(path) if isinstance(item, dict)]
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _records_from_json_payload(payload)
        if suffix == ".parquet":
            import pandas as pd

            dataframe = pd.read_parquet(path)
            return dataframe.to_dict(orient="records") if not dataframe.empty else []
        if suffix == ".arrow":
            try:
                from datasets import Dataset
            except Exception:
                return []
            dataset = Dataset.from_file(str(path))
            return [dict(item) for item in dataset]
    except Exception:
        return []
    return []


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "queries", "corpus", "qrels"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _build_joined_samples(
    root: Path,
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    (
        candidates,
        corpus_id_to_page_id,
        corpus_id_to_candidate,
        image_field,
        text_field,
        image_parse_summary,
        image_parse_diagnostics,
    ) = _build_candidate_pool(root, corpus)
    positives_by_query = _build_positive_qrels(qrels, corpus_id_to_page_id)
    samples: list[dict[str, Any]] = []
    sorted_candidates = sorted(candidates, key=lambda item: int(item["page_id"]))
    candidate_corpus_ids = [str(candidate["corpus_id"]) for candidate in sorted_candidates]
    for query_record in sorted(queries, key=lambda item: str(item.get("query_id", ""))):
        query_id = str(query_record.get("query_id", "")).strip()
        query = str(query_record.get("query", "")).strip()
        if not query_id or not query:
            continue
        positive_corpus_ids = sorted(positives_by_query.get(query_id, []))
        positive_page_ids = [
            int(corpus_id_to_page_id[corpus_id])
            for corpus_id in positive_corpus_ids
            if corpus_id in corpus_id_to_page_id
        ]
        if not positive_page_ids:
            continue
        samples.append(
            {
                "qid": query_id,
                "query_id": query_id,
                "doc_id": "vidore_v3_energy",
                "question": query,
                "query": query,
                "evidence_pages": sorted(set(positive_page_ids)),
                "gold_corpus_ids": positive_corpus_ids,
                "positive_corpus_ids": positive_corpus_ids,
                "candidate_corpus_ids": candidate_corpus_ids,
                "candidate_page_records": sorted_candidates,
                "candidates": sorted_candidates,
                "page_id_to_corpus_id": {
                    str(candidate["page_id"]): str(candidate["corpus_id"]) for candidate in sorted_candidates
                },
            }
        )
    avg_positives = (
        sum(len(sample.get("positive_corpus_ids", [])) for sample in samples) / max(len(samples), 1)
        if samples
        else 0.0
    )
    summary = {
        "num_candidate_pages": len(sorted_candidates),
        "num_corpus_pages_with_images": sum(
            1 for candidate in sorted_candidates if candidate.get("image") is not None or candidate.get("image_path")
        ),
        "avg_positives_per_query": avg_positives,
        "avg_candidate_set_size": float(len(sorted_candidates)) if samples else 0.0,
        "image_field": image_field,
        "text_field": text_field,
        "image_parse_summary": image_parse_summary,
        "image_parse_diagnostics": image_parse_diagnostics,
        "num_qrels_with_known_corpus_id": sum(
            1
            for query_positive_ids in positives_by_query.values()
            for corpus_id in query_positive_ids
            if corpus_id in corpus_id_to_candidate
        ),
    }
    return samples, summary


def _build_candidate_pool(
    root: Path,
    corpus: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]], str, str, dict[str, Any], list[dict[str, Any]]]:
    logger = get_logger("vidore_energy_loader")
    sorted_corpus = sorted(corpus, key=lambda item: str(item.get("corpus_id", "")))
    candidates: list[dict[str, Any]] = []
    corpus_id_to_page_id: dict[str, int] = {}
    corpus_id_to_candidate: dict[str, dict[str, Any]] = {}
    detected_image_field = ""
    detected_text_field = ""
    image_parse_summary: dict[str, Any] = {
        "total_corpus_records_inspected": 0,
        "num_images_loaded_via_bytes": 0,
        "num_images_loaded_via_path": 0,
        "num_images_loaded_via_direct_path": 0,
        "num_images_loaded_via_pil": 0,
        "num_images_failed": 0,
    }
    image_parse_diagnostics: list[dict[str, Any]] = []
    for page_id, record in enumerate(sorted_corpus):
        corpus_id = str(record.get("corpus_id", "")).strip()
        if not corpus_id:
            continue
        image_parse_summary["total_corpus_records_inspected"] += 1
        candidate, image_field, text_field, image_meta = _candidate_from_corpus_record(record, root, page_id)
        source_type = str(image_meta.get("source_type", "failed"))
        if source_type == "bytes":
            image_parse_summary["num_images_loaded_via_bytes"] += 1
        elif source_type == "path":
            image_parse_summary["num_images_loaded_via_path"] += 1
        elif source_type == "direct_path":
            image_parse_summary["num_images_loaded_via_direct_path"] += 1
        elif source_type == "pil":
            image_parse_summary["num_images_loaded_via_pil"] += 1
        else:
            image_parse_summary["num_images_failed"] += 1
        if len(image_parse_diagnostics) < 5:
            diagnostic = {
                "corpus_id": corpus_id,
                "page_id": int(page_id),
                **image_meta,
            }
            image_parse_diagnostics.append(diagnostic)
        if image_field and not detected_image_field:
            detected_image_field = image_field
        if text_field and not detected_text_field:
            detected_text_field = text_field
        candidates.append(candidate)
        corpus_id_to_page_id[corpus_id] = int(candidate["page_id"])
        corpus_id_to_candidate[corpus_id] = candidate
    if not candidates:
        return [], {}, {}, detected_image_field, detected_text_field, image_parse_summary, image_parse_diagnostics
    logger.info("ViDoRe corpus image parse summary: %s", image_parse_summary)
    logger.info("ViDoRe corpus image parse diagnostics: %s", image_parse_diagnostics)
    if not any(candidate.get("image") is not None or candidate.get("image_path") for candidate in candidates):
        diagnostics = {
            "dataset_path": str(root),
            "image_parse_summary": image_parse_summary,
            "image_parse_diagnostics": image_parse_diagnostics,
        }
        raise RuntimeError(
            "ViDoRe Energy corpus loaded, but no usable page images were found in corpus.image or image path fields. "
            f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False, default=str)}"
        )
    return (
        candidates,
        corpus_id_to_page_id,
        corpus_id_to_candidate,
        detected_image_field or "corpus.image",
        detected_text_field or "corpus.markdown",
        image_parse_summary,
        image_parse_diagnostics,
    )


def _candidate_from_corpus_record(record: dict[str, Any], root: Path, page_id: int) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    corpus_id = str(record.get("corpus_id", "")).strip()
    markdown, text_field = _first_value_with_key(record, PAGE_TEXT_KEYS)
    image_value = record.get("image")
    image_field = "corpus.image" if image_value is not None else ""
    image_obj, image_path, source_type, image_meta = _resolve_hf_image_field(image_value, root)
    if image_obj is None and image_path is None:
        path_value, path_key = _first_value_with_key(record, IMAGE_PATH_KEYS)
        if path_value is not None:
            image_obj, image_path, source_type, direct_meta = _resolve_direct_image_path(str(path_value), root)
            image_meta = {**image_meta, "direct_path_meta": direct_meta}
            image_field = f"corpus.{path_key}"
    candidate: dict[str, Any] = {
        "page_id": int(page_id),
        "corpus_id": corpus_id,
        "doc_id": str(record.get("doc_id", "")),
        "page_number_in_doc": record.get("page_number_in_doc"),
        "markdown": str(markdown or ""),
        "ocr_text": str(markdown or ""),
    }
    if image_obj is not None:
        candidate["image"] = image_obj
    if image_path is not None:
        candidate["image_path"] = str(image_path)
    candidate["image_source_type"] = source_type
    image_meta["source_type"] = source_type
    image_meta["resolved_image_path"] = str(image_path) if image_path is not None else ""
    return candidate, image_field, f"corpus.{text_field or 'markdown'}", image_meta


def _resolve_hf_image_field(value: Any, root: Path) -> tuple[Any | None, Path | None, str, dict[str, Any]]:
    """Resolve a Hugging Face Image-style field into a PIL image or verified image path."""
    meta: dict[str, Any] = {
        "image_field_exists": value is not None,
        "image_is_dict": isinstance(value, dict),
        "image_has_bytes": False,
        "image_bytes_non_empty": False,
        "image_has_path": False,
        "image_path_value": "",
        "bytes_decode_succeeded": False,
        "bytes_decode_error": "",
        "path_load_succeeded": False,
        "path_load_error": "",
        "path_candidates": [],
        "path_candidates_exist": [],
        "pil_image_succeeded": False,
    }
    if value is None:
        return None, None, "failed", meta
    if _is_pil_image(value):
        try:
            meta["pil_image_succeeded"] = True
            return value.convert("RGB"), None, "pil", meta
        except Exception as exc:
            meta["path_load_error"] = f"PIL convert failed: {exc}"
            return None, None, "failed", meta
    if isinstance(value, dict):
        if value.get("bytes"):
            meta["image_has_bytes"] = True
            meta["image_bytes_non_empty"] = True
            try:
                from PIL import Image

                image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
                meta["bytes_decode_succeeded"] = True
                return image, None, "bytes", meta
            except Exception as exc:
                meta["bytes_decode_error"] = str(exc)
        elif "bytes" in value:
            meta["image_has_bytes"] = True
            meta["image_bytes_non_empty"] = False
        if value.get("path"):
            meta["image_has_path"] = True
            meta["image_path_value"] = str(value["path"])
            image_path, path_meta = _resolve_image_path(str(value["path"]), root, include_corpus_dir=True)
            meta.update(path_meta)
            if image_path is not None:
                return None, image_path, "path", meta
        elif "path" in value:
            meta["image_has_path"] = True
            meta["image_path_value"] = str(value.get("path") or "")
    if isinstance(value, (str, Path)):
        meta["image_path_value"] = str(value)
        image_path, path_meta = _resolve_image_path(str(value), root, include_corpus_dir=True)
        meta.update(path_meta)
        if image_path is not None:
            return None, image_path, "direct_path", meta
    return None, None, "failed", meta


def _resolve_direct_image_path(value: str, root: Path) -> tuple[Any | None, Path | None, str, dict[str, Any]]:
    image_path, meta = _resolve_image_path(value, root, include_corpus_dir=True)
    if image_path is None:
        return None, None, "failed", meta
    return None, image_path, "direct_path", meta


def _resolve_image_path(value: str, root: Path, include_corpus_dir: bool = True) -> tuple[Path | None, dict[str, Any]]:
    raw_path = Path(value)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([Path(value), root / raw_path])
        if include_corpus_dir:
            candidates.append(root / "corpus" / raw_path)
    normalized_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        normalized_candidates.append(candidate)
    exists_flags = [candidate.exists() for candidate in normalized_candidates]
    meta: dict[str, Any] = {
        "path_candidates": [str(candidate) for candidate in normalized_candidates],
        "path_candidates_exist": exists_flags,
        "path_exists_with_supported_extension": False,
        "path_load_succeeded": False,
        "path_load_error": "",
    }
    for candidate in normalized_candidates:
        if not candidate.exists() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        meta["path_exists_with_supported_extension"] = True
        try:
            from PIL import Image

            with Image.open(str(candidate)) as image:
                image.convert("RGB")
            meta["path_load_succeeded"] = True
            return candidate, meta
        except Exception as exc:
            meta["path_load_error"] = str(exc)
            return candidate, meta
    return None, meta


def _build_positive_qrels(qrels: list[dict[str, Any]], corpus_id_to_page_id: dict[str, int]) -> dict[str, set[str]]:
    positives_by_query: dict[str, set[str]] = {}
    for record in qrels:
        query_id = str(record.get("query_id", "")).strip()
        corpus_id = str(record.get("corpus_id", "")).strip()
        if not query_id or not corpus_id:
            continue
        if corpus_id not in corpus_id_to_page_id:
            continue
        score = _safe_float(record.get("score", record.get("relevance", record.get("label", 1.0))), default=1.0)
        if score <= 0:
            continue
        positives_by_query.setdefault(query_id, set()).add(corpus_id)
    return positives_by_query


def _raise_if_empty_component(
    root: Path,
    sources: dict[str, str],
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    examples: dict[str, Any],
) -> None:
    if queries and corpus and qrels:
        return
    diagnostics = {
        "dataset_path": str(root),
        "sources": sources,
        "num_queries": len(queries),
        "num_candidate_pages": len(corpus),
        "num_qrels": len(qrels),
        "example_records": examples,
    }
    raise RuntimeError(
        "ViDoRe Energy dataset is empty or missing required queries/corpus/qrels components. "
        f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False, default=str)}"
    )


def _raise_empty_join(
    root: Path,
    sources: dict[str, str],
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    examples: dict[str, Any],
    join_summary: dict[str, Any],
) -> None:
    diagnostics = {
        "dataset_path": str(root),
        "sources": sources,
        "num_queries": len(queries),
        "num_candidate_pages": len(corpus),
        "num_qrels": len(qrels),
        "join_summary": join_summary,
        "example_records": examples,
    }
    raise RuntimeError(
        "ViDoRe Energy join produced zero train/eval samples. "
        "Check query_id/corpus_id/qrels.score alignment. "
        f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False, default=str)}"
    )


def _looks_like_hf_dataset(root: Path) -> bool:
    return (root / "dataset_info.json").exists() or (root / "state.json").exists()


def _first_value_with_key(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any | None, str]:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value, key
    return None, ""


def _is_pil_image(value: Any) -> bool:
    return value is not None and value.__class__.__module__.startswith("PIL.") and value.__class__.__name__ == "Image"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compact_record_for_log(record: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in record.items():
        if _is_pil_image(value):
            compact[key] = f"<PIL.Image size={getattr(value, 'size', None)}>"
        elif isinstance(value, dict) and "bytes" in value:
            compact[key] = {
                child_key: ("<bytes>" if child_key == "bytes" else child_value)
                for child_key, child_value in value.items()
            }
        elif isinstance(value, bytes):
            compact[key] = "<bytes>"
        else:
            text = str(value)
            compact[key] = text[:240] + ("..." if len(text) > 240 else "")
    return compact
