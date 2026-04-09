"""MP-DocVQA IMDB conversion into the project's canonical processed JSONL format."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.utils.io_utils import ensure_dir, load_jsonl, save_json, save_jsonl
from src.utils.logger import get_logger

PAGE_PATTERN = re.compile(r"_p(\d+)\.(jpg|json)$")


def extract_page_idx(path_or_name: str) -> int:
    """Extract the numeric page index from an MP-DocVQA image or OCR filename."""
    match = PAGE_PATTERN.search(Path(path_or_name).name)
    if match is None:
        raise ValueError(f"Failed to extract page index from: {path_or_name}")
    return int(match.group(1))


def _extract_doc_id(path_or_name: str) -> str:
    """Extract the document id from an MP-DocVQA page filename."""
    filename = Path(path_or_name).name
    match = PAGE_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Failed to extract doc id from: {path_or_name}")
    return filename[: match.start()]


def _build_document_path_index(root: str, suffix: str) -> dict[str, list[str]]:
    """Scan a flat directory once and build a doc_id -> sorted paths index."""
    directory = Path(root)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    index: dict[str, list[str]] = defaultdict(list)
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        try:
            doc_id = _extract_doc_id(str(path))
        except ValueError:
            continue
        index[doc_id].append(str(path))

    for doc_id, paths in index.items():
        paths.sort(key=extract_page_idx)
    return dict(index)


def get_document_page_images(
    doc_id: str,
    image_root: str,
    image_index: dict[str, list[str]] | None = None,
) -> list[str]:
    """Scan the image root and return all document page images sorted by page index."""
    if image_index is not None:
        return list(image_index.get(doc_id, []))

    image_dir = Path(image_root)
    image_paths = sorted(image_dir.glob(f"{doc_id}_p*.jpg"), key=lambda path: extract_page_idx(str(path)))
    return [str(path) for path in image_paths]


def get_document_ocr_paths(
    doc_id: str,
    ocr_root: str,
    ocr_index: dict[str, list[str]] | None = None,
) -> list[str]:
    """Scan the OCR root and return all document OCR files sorted by page index."""
    if ocr_index is not None:
        return list(ocr_index.get(doc_id, []))

    ocr_dir = Path(ocr_root)
    ocr_paths = sorted(ocr_dir.glob(f"{doc_id}_p*.json"), key=lambda path: extract_page_idx(str(path)))
    return [str(path) for path in ocr_paths]


def load_mpdocvqa_imdb(imdb_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load an MP-DocVQA IMDB file and split global metadata from real samples."""
    imdb = np.load(imdb_path, allow_pickle=True)
    normalized_entries: list[dict[str, Any]] = []
    for entry in imdb:
        if isinstance(entry, dict):
            normalized_entries.append(entry)
        elif hasattr(entry, "item"):
            normalized_entries.append(entry.item())
        else:
            raise ValueError(f"Unsupported IMDB entry type: {type(entry)!r}")

    if not normalized_entries:
        return {}, []
    meta = normalized_entries[0]
    samples = normalized_entries[1:]
    return meta, samples


def convert_mpdocvqa_sample(
    sample: dict[str, Any],
    image_root: str,
    ocr_root: str,
    image_index: dict[str, list[str]] | None = None,
    ocr_index: dict[str, list[str]] | None = None,
    fallback_idx: int | None = None,
) -> dict[str, Any]:
    """Convert one raw MP-DocVQA sample into the project's canonical sample format."""
    split = str(sample.get("set_name", "")).strip()
    raw_question_id = sample.get("question_id", fallback_idx)
    qid = f"{split}_{raw_question_id}"
    question = str(sample.get("question", "")).strip()

    answers = _deduplicate_answers(sample.get("valid_answers") or sample.get("answers") or [])
    answer = answers[0] if answers else ""
    doc_id = str(sample.get("image_id", "")).strip()
    answer_page = sample.get("answer_page")
    answer_page_idx = sample.get("answer_page_idx")
    if answer_page is not None:
        evidence_pages = [int(answer_page)]
    elif answer_page_idx is not None:
        evidence_pages = [int(answer_page_idx)]
    else:
        evidence_pages = []
    image_paths = get_document_page_images(doc_id, image_root, image_index=image_index)
    ocr_paths = get_document_ocr_paths(doc_id, ocr_root, ocr_index=ocr_index)
    page_count = len(image_paths)

    return {
        "qid": qid,
        "question": question,
        "answers": answers,
        "answer": answer,
        "doc_id": doc_id,
        "evidence_pages": evidence_pages,
        "page_count": page_count,
        "image_paths": image_paths,
        "ocr_paths": ocr_paths,
        "meta": {
            "dataset": "mp-docvqa",
            "split": split,
            "raw_question_id": raw_question_id,
            "raw_doc_id": sample.get("image_id"),
            "answer_page": sample.get("answer_page"),
            "answer_page_idx": sample.get("answer_page_idx"),
            "imdb_doc_pages": sample.get("imdb_doc_pages"),
            "total_doc_pages": sample.get("total_doc_pages"),
            "pages": sample.get("pages"),
            "extra_info": sample.get("extra_info", {}),
        },
    }


def build_mpdocvqa_split(
    imdb_path: str,
    image_root: str,
    ocr_root: str,
    output_path: str,
    skip_missing_answer_for_train_val: bool = True,
) -> dict[str, Any]:
    """Convert one MP-DocVQA IMDB split into processed JSONL and return build statistics."""
    logger = get_logger("mpdocvqa_sample_builder")
    meta, raw_samples = load_mpdocvqa_imdb(imdb_path)
    split = str(meta.get("dataset_type", "") or Path(imdb_path).stem.replace("imdb_", "")).strip()
    image_index = _build_document_path_index(image_root, ".jpg")
    ocr_index = _build_document_path_index(ocr_root, ".json")

    report = {
        "split": split,
        "raw_count": len(raw_samples),
        "valid_count": 0,
        "skipped_missing_question": 0,
        "skipped_missing_images": 0,
        "skipped_missing_ocr": 0,
        "skipped_mismatched_assets": 0,
        "skipped_invalid_evidence_page": 0,
        "skipped_missing_answer": 0,
    }

    valid_samples: list[dict[str, Any]] = []
    for idx, raw_sample in enumerate(tqdm(raw_samples, desc=f"Building {split}", unit="sample")):
        converted = convert_mpdocvqa_sample(
            raw_sample,
            image_root=image_root,
            ocr_root=ocr_root,
            image_index=image_index,
            ocr_index=ocr_index,
            fallback_idx=idx,
        )

        if not converted["question"]:
            report["skipped_missing_question"] += 1
            continue
        if len(converted["image_paths"]) == 0:
            report["skipped_missing_images"] += 1
            continue
        if len(converted["ocr_paths"]) == 0:
            report["skipped_missing_ocr"] += 1
            continue
        if not _paths_have_consistent_page_indices(converted["image_paths"], converted["ocr_paths"]):
            report["skipped_mismatched_assets"] += 1
            continue
        if split in {"train", "val"} and (
            not converted["evidence_pages"]
            or converted["evidence_pages"][0] < 0
            or converted["evidence_pages"][0] >= len(converted["image_paths"])
        ):
            report["skipped_invalid_evidence_page"] += 1
            continue
        if not converted["answers"] and split in {"train", "val"} and skip_missing_answer_for_train_val:
            report["skipped_missing_answer"] += 1
            continue

        valid_samples.append(converted)

    report["valid_count"] = len(valid_samples)

    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    save_jsonl(valid_samples, output_file)

    report_path = Path("outputs/logs") / f"mpdocvqa_build_report_{split}.json"
    ensure_dir(report_path.parent)
    save_json(report, report_path)
    logger.info("Built MP-DocVQA split %s: %s", split, report)
    return report


class SampleBuilder:
    """Project-integrated builder that converts configured MP-DocVQA splits."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def build_split(self, split: str) -> list[dict[str, Any]]:
        """Build one processed split from the configured MP-DocVQA IMDB file."""
        output_path = str(self.cfg.dataset.processed_files[split])
        build_mpdocvqa_split(
            imdb_path=str(self.cfg.dataset.split_files[split]),
            image_root=str(self.cfg.dataset.raw_image_dir),
            ocr_root=str(self.cfg.dataset.raw_ocr_dir),
            output_path=output_path,
            skip_missing_answer_for_train_val=True,
        )
        return load_jsonl(output_path)


def _deduplicate_answers(answers: list[Any]) -> list[str]:
    """Deduplicate answers while preserving order."""
    deduplicated: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        normalized = str(answer).strip()
        if not normalized or normalized in seen:
            continue
        deduplicated.append(normalized)
        seen.add(normalized)
    return deduplicated


def _paths_have_consistent_page_indices(image_paths: list[str], ocr_paths: list[str]) -> bool:
    """Check whether image and OCR files cover the same ordered page indices."""
    image_indices = [extract_page_idx(path) for path in image_paths]
    ocr_indices = [extract_page_idx(path) for path in ocr_paths]
    return image_indices == ocr_indices


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone split conversion."""
    parser = argparse.ArgumentParser(description="Convert one MP-DocVQA IMDB split into processed JSONL.")
    parser.add_argument("--imdb_path", required=True, help="Path to imdb_{split}.npy")
    parser.add_argument("--image_root", required=True, help="Root directory for flattened page images")
    parser.add_argument("--ocr_root", required=True, help="Root directory for flattened OCR JSON files")
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_mpdocvqa_split(
        imdb_path=args.imdb_path,
        image_root=args.image_root,
        ocr_root=args.ocr_root,
        output_path=args.output_path,
    )
