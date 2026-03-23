"""Unified CLI entrypoint for LightRoute phase-based pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pipeline import (  # noqa: E402
    build_indexes_pipeline,
    eval_docvqa_pipeline,
    eval_retrieval_pipeline,
    infer_docvqa_pipeline,
    prepare_data_pipeline,
    train_fusion_pipeline,
)
from utils.config_utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the unified project entrypoint."""
    parser = argparse.ArgumentParser(description="LightRoute DocVQA pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--mode",
        default="eval_retrieval",
        choices=[
            "prepare_data",
            "build_indexes",
            "eval_retrieval",
            "train_fusion",
            "infer_docvqa",
            "eval_docvqa",
        ],
        help="Pipeline mode to execute.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration and dispatch to the selected pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    dispatch = {
        "prepare_data": prepare_data_pipeline,
        "build_indexes": build_indexes_pipeline,
        "eval_retrieval": eval_retrieval_pipeline,
        "train_fusion": train_fusion_pipeline,
        "infer_docvqa": infer_docvqa_pipeline,
        "eval_docvqa": eval_docvqa_pipeline,
    }
    dispatch[args.mode](cfg)


if __name__ == "__main__":
    main()
