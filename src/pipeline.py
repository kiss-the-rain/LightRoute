"""Top-level pipeline orchestration with cache-aware phase execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.dataset import DatasetManager
from data.doc_dataset import DocVQADataset
from data.ocr_engine import OCREngine
from data.pdf_renderer import render_pdf_to_images
from evaluation.evaluator import Evaluator
from inference.infer_docvqa import infer_docvqa_sample
from retrieval.colpali_retriever import ColPaliRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.index_builder import build_text_index, build_visual_index, collect_document_pages
from retrieval.retriever_manager import RetrieverManager
from training.train_fusion import train_fusion
from utils.io_utils import ensure_dir, save_json, save_jsonl
from utils.logger import setup_logger
from utils.seed_utils import set_seed


def prepare_data_pipeline(cfg: Any) -> None:
    """Prepare processed samples, page images, and OCR caches."""
    logger = _build_logger(cfg, "prepare_data")
    set_seed(int(cfg.seed))
    _ensure_runtime_dirs(cfg)
    dataset_manager = DatasetManager(cfg)
    ocr_engine = OCREngine(backend="dummy")

    for split in cfg.dataset.processed_files.keys():
        processed_path = Path(cfg.dataset.processed_files[split])
        if _should_skip(cfg, processed_path):
            logger.info("Skipping split %s because processed file exists: %s", split, processed_path)
            continue

        logger.info("Building processed split: %s", split)
        samples = dataset_manager.prepare(split)
        if not samples:
            logger.info("No raw annotations found for split %s. Skipping OCR/render stage.", split)
            continue

        for sample in samples:
            _ensure_sample_page_images(cfg, sample, logger)
            _ensure_sample_ocr(cfg, sample, ocr_engine, logger)

        save_jsonl(samples, processed_path)

    logger.info("Prepare-data pipeline finished.")


def build_indexes_pipeline(cfg: Any) -> None:
    """Build visual and BM25 text retrieval indexes."""
    logger = _build_logger(cfg, "build_indexes")
    _ensure_runtime_dirs(cfg)
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    processed_samples = list(dataset)
    documents = collect_document_pages(cfg, processed_samples)
    visual_retriever = ColPaliRetriever(cfg)
    text_retriever = BM25Retriever(cfg)
    logger.info("Building visual index for %d documents.", len(documents))
    build_visual_index(visual_retriever, documents)
    logger.info("Building text index for %d documents.", len(documents))
    build_text_index(text_retriever, documents)
    logger.info("Index build completed.")


def eval_retrieval_pipeline(cfg: Any) -> None:
    """Run Phase 1 retrieval baselines and compute retrieval metrics."""
    logger = _build_logger(cfg, "eval_retrieval")
    _ensure_runtime_dirs(cfg)
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    processed_samples = list(dataset)
    documents = collect_document_pages(cfg, processed_samples)
    visual_retriever = ColPaliRetriever(cfg)
    text_retriever = BM25Retriever(cfg)
    build_visual_index(visual_retriever, documents)
    build_text_index(text_retriever, documents)
    retriever_manager = RetrieverManager(cfg, visual_retriever, text_retriever)
    evaluator = Evaluator(cfg, retriever_manager)
    output_prefix = f"{cfg.experiment.name}_{cfg.experiment.split}"
    metrics = evaluator.evaluate_retrieval(processed_samples, output_prefix=output_prefix)
    logger.info("Retrieval evaluation completed: %s", metrics)


def train_fusion_pipeline(cfg: Any) -> None:
    """Placeholder training pipeline for later adaptive fusion phases."""
    logger = _build_logger(cfg, "train_fusion")
    _ensure_runtime_dirs(cfg)
    logger.info("Fusion training is currently a placeholder beyond Phase 1.")
    train_fusion(cfg)


def infer_docvqa_pipeline(cfg: Any) -> None:
    """Placeholder end-to-end DocVQA inference pipeline."""
    logger = _build_logger(cfg, "infer_docvqa")
    dataset = _load_split_dataset(cfg, str(cfg.experiment.split))
    predictions = [infer_docvqa_sample(sample) for sample in dataset]
    save_json(predictions, Path(cfg.paths.prediction_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_docvqa_predictions.json")
    logger.info("Saved placeholder DocVQA predictions for %d samples.", len(predictions))


def eval_docvqa_pipeline(cfg: Any) -> None:
    """Placeholder end-to-end DocVQA evaluation pipeline."""
    logger = _build_logger(cfg, "eval_docvqa")
    save_json({"status": "not_implemented"}, Path(cfg.paths.metric_dir) / f"{cfg.experiment.name}_{cfg.experiment.split}_docvqa_metrics.json")
    logger.info("DocVQA evaluation is reserved for Phase 4.")


def _ensure_runtime_dirs(cfg: Any) -> None:
    for key, value in cfg.paths.items():
        if key.endswith("_dir"):
            ensure_dir(value)


def _should_skip(cfg: Any, target_path: Path) -> bool:
    return bool(cfg.runtime.use_cache) and not bool(cfg.runtime.overwrite_cache) and target_path.exists()


def _build_logger(cfg: Any, stage_name: str):
    experiment_name = f"{cfg.experiment.name}_{stage_name}"
    return setup_logger(stage_name, cfg.paths.log_dir, experiment_name)


def _load_split_dataset(cfg: Any, split: str) -> DocVQADataset:
    dataset = DocVQADataset(cfg, split)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No processed samples found for split '{split}'. Run prepare_data first or place files under {cfg.dataset.processed_files[split]}."
        )
    return dataset


def _ensure_sample_page_images(cfg: Any, sample: dict[str, Any], logger: Any) -> None:
    if sample.get("image_paths"):
        return
    pdf_path = Path(cfg.dataset.raw_pdf_dir) / f"{sample['doc_id']}.pdf"
    if not pdf_path.exists():
        logger.info("No PDF found for doc %s at %s", sample["doc_id"], pdf_path)
        return
    output_dir = Path(cfg.dataset.page_image_dir) / sample["doc_id"]
    render_result = render_pdf_to_images(pdf_path, output_dir)
    sample["image_paths"] = render_result["image_paths"]
    sample["page_ids"] = list(render_result["page_id_map"].keys())


def _ensure_sample_ocr(cfg: Any, sample: dict[str, Any], ocr_engine: OCREngine, logger: Any) -> None:
    ocr_path = Path(sample["ocr_results_path"])
    if _should_skip(cfg, ocr_path):
        return
    image_paths = sample.get("image_paths", [])
    if not image_paths:
        logger.info("No page images available for doc %s. OCR skipped.", sample["doc_id"])
        save_json([], ocr_path)
        return
    ocr_results = ocr_engine.run_ocr_on_document(image_paths)
    save_json(ocr_results, ocr_path)
