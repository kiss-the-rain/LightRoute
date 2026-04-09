#!/usr/bin/env python3
"""Minimal offline self-check for loading and encoding with NV-Embed-v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QUERY = "What is the dividend payout in 2012?"
DEFAULT_PASSAGE = "The dividend payout in 2012 was 1.25 dollars per share."
DEFAULT_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the NV-Embed-v2 self-check."""
    parser = argparse.ArgumentParser(description="Offline NV-Embed-v2 self-check")
    parser.add_argument("--model-path", required=True, help="Local NV-Embed-v2 directory")
    parser.add_argument("--device", default="cuda:0", help="Target device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length")
    parser.add_argument("--batch-size", type=int, default=2, help="Encoding batch size")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Sample retrieval query")
    parser.add_argument("--passage", default=DEFAULT_PASSAGE, help="Sample passage text")
    return parser.parse_args()


def resolve_local_model_path(model_path: str) -> Path:
    """Resolve and validate the local NV-Embed-v2 directory."""
    candidate = Path(model_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Local model directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {candidate}")
    return candidate


def patch_config_to_local_path(config: Any, local_path: str) -> None:
    """Force nested config references to stay local-only."""
    for attr_name in ("_name_or_path", "name_or_path"):
        if hasattr(config, attr_name):
            setattr(config, attr_name, local_path)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr_name in ("_name_or_path", "name_or_path"):
            if hasattr(text_config, attr_name):
                setattr(text_config, attr_name, local_path)
        if hasattr(text_config, "tokenizer_name"):
            setattr(text_config, "tokenizer_name", local_path)

    latent_attention_config = getattr(config, "latent_attention_config", None)
    if latent_attention_config is not None:
        for attr_name in ("_name_or_path", "name_or_path"):
            if hasattr(latent_attention_config, attr_name):
                setattr(latent_attention_config, attr_name, local_path)


def patch_transformers_tied_weights_compat(modeling_utils: Any) -> None:
    """Provide a writable all_tied_weights_keys compatibility shim when needed."""
    pre_trained_model = modeling_utils.PreTrainedModel
    existing_attr = getattr(pre_trained_model, "all_tied_weights_keys", None)
    if isinstance(existing_attr, property) and existing_attr.fset is not None:
        return

    storage_name = "_codex_all_tied_weights_keys"

    def _expand_default_tied_keys(self: Any) -> dict[str, bool]:
        tied = getattr(self, "_tied_weights_keys", None)
        if tied is None:
            return {}
        if isinstance(tied, dict):
            return {str(key): bool(value) for key, value in tied.items()}
        if isinstance(tied, (list, tuple, set)):
            return {str(key): True for key in tied}
        return {}

    def _getter(self: Any) -> dict[str, bool]:
        stored = getattr(self, storage_name, None)
        if stored is not None:
            return stored
        expanded = _expand_default_tied_keys(self)
        setattr(self, storage_name, expanded)
        return expanded

    def _setter(self: Any, value: Any) -> None:
        if value is None:
            setattr(self, storage_name, {})
            return
        if isinstance(value, dict):
            normalized = {str(key): bool(flag) for key, flag in value.items()}
        elif isinstance(value, (list, tuple, set)):
            normalized = {str(key): True for key in value}
        else:
            normalized = {}
        setattr(self, storage_name, normalized)

    pre_trained_model.all_tied_weights_keys = property(_getter, _setter)


def patch_transformers_dynamic_cache_compat(dynamic_cache_cls: Any) -> None:
    """Restore DynamicCache.from_legacy_cache for custom model code expecting it."""
    if hasattr(dynamic_cache_cls, "from_legacy_cache"):
        return

    @classmethod
    def _from_legacy_cache(cls, past_key_values: Any) -> Any:
        if past_key_values is None:
            return cls()
        return past_key_values

    dynamic_cache_cls.from_legacy_cache = _from_legacy_cache


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Apply row-wise L2 normalization."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return array / norms


def encode_texts(
    model: Any,
    tokenizer: Any,
    torch: Any,
    texts: list[str],
    *,
    instruction: str,
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode texts with NV-Embed-v2, preferring model.encode when available."""
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            if hasattr(model, "encode"):
                output = model.encode(
                    batch_texts,
                    instruction=instruction,
                    max_length=max_length,
                    batch_size=len(batch_texts),
                )
                if hasattr(output, "detach"):
                    output = output.detach().cpu()
                array = np.asarray(output, dtype=float)
                if array.ndim == 1:
                    array = np.expand_dims(array, axis=0)
                embeddings.append(l2_normalize(array))
                continue

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().numpy().astype(float))
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    """Run the offline load + encode sanity check."""
    args = parse_args()
    model_path = resolve_local_model_path(args.model_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, modeling_utils
    from transformers.cache_utils import DynamicCache

    patch_transformers_tied_weights_compat(modeling_utils)
    patch_transformers_dynamic_cache_compat(DynamicCache)

    config = AutoConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    patch_config_to_local_path(config, str(model_path))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
        trust_remote_code=True,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device {device} unavailable, falling back to cpu")
        device = "cpu"

    model.to(device)
    model.eval()

    query = f"{DEFAULT_INSTRUCTION}{args.query}"
    query_emb = encode_texts(
        model,
        tokenizer,
        torch,
        [query],
        instruction=DEFAULT_INSTRUCTION,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    passage_emb = encode_texts(
        model,
        tokenizer,
        torch,
        [args.passage],
        instruction="",
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    score = float(query_emb[0] @ passage_emb[0])

    print("NV-Embed-v2 offline self-check succeeded")
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"query_embedding_shape: {tuple(query_emb.shape)}")
    print(f"passage_embedding_shape: {tuple(passage_emb.shape)}")
    print(f"similarity: {score:.6f}")


if __name__ == "__main__":
    main()
