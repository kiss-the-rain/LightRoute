#!/usr/bin/env python
"""Standalone Nemotron sanity check on local ViDoRe V3 Energy.

This script intentionally bypasses the project train/eval dispatch. It loads a
few real ViDoRe queries, builds tiny 1-positive + N-negative visual candidate
pools, scores them with the local Nemotron visual model, and prints ranked
results in corpus_id space.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.vidore_energy_loader import load_vidore_energy_dataset  # noqa: E402
from src.retrieval.nemotron_visual_retriever import NemotronVisualRetriever  # noqa: E402


DEFAULT_MODEL_PATH = "/home/cuizhibin/projects/Models/nemotron-colembed-vl-8b-v2"
DEFAULT_DATASET_PATH = "/home/cuizhibin/projects/Models/vidore_dataset/vidore_v3_energy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Nemotron sanity check on ViDoRe V3 Energy.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--num-negatives", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default="outputs/cache/visual_nemotron_energy_sanity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    print(f"[setup] model_path={args.model_path}")
    print(f"[setup] dataset_path={args.dataset_path}")
    print(f"[setup] device={args.device} seed={args.seed} num_queries={args.num_queries} num_negatives={args.num_negatives}")

    samples, summary = load_vidore_energy_dataset(args.dataset_path, max_samples=0)
    print(
        "[dataset] "
        f"num_queries={summary.get('num_queries')} "
        f"num_corpus_pages={summary.get('num_corpus_pages')} "
        f"num_candidate_pages={summary.get('num_candidate_pages')} "
        f"num_qrels={summary.get('num_qrels')} "
        f"num_final_samples={summary.get('num_final_samples')}"
    )
    print(f"[dataset] image_parse_summary={summary.get('image_parse_summary')}")
    if not samples:
        raise RuntimeError("No joined ViDoRe Energy samples were loaded.")

    retriever = NemotronVisualRetriever(
        model_path=args.model_path,
        device=args.device,
        local_files_only=True,
        cache_dir=args.cache_dir,
    )
    retriever.load()

    tested = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    positive_ranks: list[int] = []

    for sample in samples:
        if tested >= args.num_queries:
            break
        query_id = str(sample.get("query_id", sample.get("qid", "")))
        query_text = str(sample.get("query", "") or "").strip()
        if not query_text:
            raise RuntimeError(f"Empty query text for query_id={query_id}")
        positives = [str(corpus_id) for corpus_id in sample.get("positive_corpus_ids", [])]
        if not positives:
            raise RuntimeError(f"No positive_corpus_ids for query_id={query_id}")

        candidates = list(sample.get("candidates", []))
        positive_corpus_id = positives[0]
        positive_candidates = [item for item in candidates if str(item.get("corpus_id")) == positive_corpus_id]
        if not positive_candidates:
            raise RuntimeError(f"Positive corpus_id={positive_corpus_id} is not in candidate pool for query_id={query_id}")

        negative_pool = [item for item in candidates if str(item.get("corpus_id")) not in set(positives)]
        if len(negative_pool) < args.num_negatives:
            raise RuntimeError(
                f"Not enough negatives for query_id={query_id}: "
                f"needed={args.num_negatives} available={len(negative_pool)}"
            )
        negatives = random.sample(negative_pool, args.num_negatives)
        tiny_candidates = positive_candidates[:1] + negatives
        random.shuffle(tiny_candidates)

        scored_rows = score_candidate_pool(retriever, query_text, tiny_candidates)
        ranked_rows = sorted(scored_rows, key=lambda row: row["score"], reverse=True)
        gold_rank = next(
            (rank for rank, row in enumerate(ranked_rows, start=1) if row["corpus_id"] == positive_corpus_id),
            None,
        )
        if gold_rank is None:
            raise RuntimeError(f"Gold positive disappeared from ranked rows for query_id={query_id}")
        positive_ranks.append(gold_rank)
        top1_hits += int(gold_rank == 1)
        top3_hits += int(gold_rank <= 3)
        top5_hits += int(gold_rank <= 5)
        tested += 1

        print("\n" + "=" * 100)
        print(f"[query {tested}] query_id={query_id}")
        print(f"[query {tested}] query={query_text}")
        print(f"[query {tested}] positive_corpus_id={positive_corpus_id}")
        print(f"[query {tested}] negative_corpus_ids={[str(item.get('corpus_id')) for item in negatives]}")
        print(f"[query {tested}] all_candidate_corpus_ids={[str(item.get('corpus_id')) for item in tiny_candidates]}")
        print(f"[query {tested}] gold_rank={gold_rank} top1={gold_rank == 1} top3={gold_rank <= 3} top5={gold_rank <= 5}")
        print(f"[query {tested}] ranked_results_desc_score:")
        for rank, row in enumerate(ranked_rows, start=1):
            marker = "<-- GOLD" if row["corpus_id"] == positive_corpus_id else ""
            print(
                f"  rank={rank:02d} corpus_id={row['corpus_id']} "
                f"score={row['score']:.6f} source={row['image_source']} "
                f"size={row['image_size']} {marker}"
            )

    if tested == 0:
        raise RuntimeError("No samples were tested.")
    avg_rank = sum(positive_ranks) / len(positive_ranks)
    print("\n" + "=" * 100)
    print("[summary]")
    print(f"tested_queries={tested}")
    print(f"top1_hits={top1_hits}/{tested} ({top1_hits / tested:.4f})")
    print(f"top3_hits={top3_hits}/{tested} ({top3_hits / tested:.4f})")
    print(f"top5_hits={top5_hits}/{tested} ({top5_hits / tested:.4f})")
    print(f"average_positive_rank={avg_rank:.4f}")
    print(f"cache_stats={retriever.cache_stats()}")
    retriever.save_cache()


def score_candidate_pool(
    retriever: NemotronVisualRetriever,
    query_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    query_embedding = retriever.encode_query(query_text)
    image_embeddings: list[Any] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        image_summary = describe_candidate_image(candidate)
        embedding = retriever.encode_image_candidate(candidate)
        image_embeddings.append(embedding)
        rows.append(
            {
                "corpus_id": str(candidate.get("corpus_id")),
                "page_id": int(candidate.get("page_id", -1)),
                "image_source": image_summary["source"],
                "image_size": image_summary["size"],
            }
        )
    scores = retriever.score(query_embedding, image_embeddings)
    for row, score in zip(rows, scores, strict=False):
        row["score"] = float(score)
    return rows


def describe_candidate_image(candidate: dict[str, Any]) -> dict[str, Any]:
    image = candidate.get("image")
    if image is not None:
        return {
            "source": str(candidate.get("image_source_type", "bytes_or_pil")),
            "size": tuple(getattr(image, "size", ())),
        }
    image_path = candidate.get("image_path")
    if not image_path:
        raise RuntimeError(f"Candidate has neither image nor image_path: corpus_id={candidate.get('corpus_id')}")
    from PIL import Image

    with Image.open(str(image_path)) as loaded:
        return {
            "source": str(candidate.get("image_source_type", "path")),
            "size": tuple(loaded.size),
        }


if __name__ == "__main__":
    main()
