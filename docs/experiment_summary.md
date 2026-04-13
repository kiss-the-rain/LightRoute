# Experiment Summary

本页汇总当前项目已经完成的主要 retrieval / fusion 实验，并补充训练命令、评估命令、checkpoint、metrics 文件和 predictions 文件。所有结果均基于当前已完成的 val 实验记录。

## Current Best Snapshot

- 当前最强 visual-only 是 `visual_colqwen`
- 当前最强 OCR-only 是 `BGE chunk OCR-only`
- 当前旧主线最强 fusion 是 `adaptive_fusion_mlp_ocrq_chunk`

## Main Experiment Table

| Model | Type | Recall@1 | Recall@5 | Recall@10 | MRR | Status | Train Command | Eval Command | Checkpoint | Metrics File | Predictions File |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| BM25 OCR-only | OCR-only baseline | 0.3902 | 0.6933 | 0.8043 | 0.5209 | baseline | `N/A` | `See [BM25 OCR-only Commands](#bm25-ocr-only-commands)` | 无 | `outputs/metrics/bm25_val_metrics.json` | `outputs/predictions/bm25_val_predictions.jsonl` |
| old visual-only | visual-only baseline | 0.6428 | 0.8764 | 0.9208 | 0.7420 | old baseline | `N/A` | `See [Old Visual-only Commands](#old-visual-only-commands)` | 无 | `outputs/metrics/visual_val_metrics.json` | `outputs/predictions/visual_val_predictions.jsonl` |
| adaptive_fusion_ablate_mlp_ocrq | old fusion baseline | 0.6424 | 0.8747 | 0.9202 | 0.7415 | old best fusion | `See [adaptive_fusion_ablate_mlp_ocrq Commands](#adaptive_fusion_ablate_mlp_ocrq-commands)` | `See [adaptive_fusion_ablate_mlp_ocrq Commands](#adaptive_fusion_ablate_mlp_ocrq-commands)` | `outputs/checkpoints/adaptive_fusion_ablate_mlp_ocrq/adaptive_fusion_ablate_mlp_ocrq_best.pkl` | `outputs/metrics/adaptive_fusion_ablate_mlp_ocrq_val_metrics.json` | `outputs/predictions/adaptive_fusion_ablate_mlp_ocrq_val_predictions.jsonl` |
| BGE chunk OCR-only | OCR-only strongest | 0.4712 | 0.7295 | 0.8024 | 0.5810 | strongest OCR-only | `N/A` | `See [BGE Chunk OCR-only Commands](#bge-chunk-ocr-only-commands)` | 无 | `outputs/metrics/ocr_bge_chunk_rerank_val_metrics.json` | `outputs/predictions/ocr_bge_chunk_rerank_val_predictions.jsonl` |
| adaptive_fusion_mlp_ocrq_chunk | best old fusion | 0.6435 | 0.8751 | 0.9202 | 0.7423 | best old fusion | `See [adaptive_fusion_mlp_ocrq_chunk Commands](#adaptive_fusion_mlp_ocrq_chunk-commands)` | `See [adaptive_fusion_mlp_ocrq_chunk Commands](#adaptive_fusion_mlp_ocrq_chunk-commands)` | `outputs/checkpoints/adaptive_fusion_mlp_ocrq_chunk/adaptive_fusion_mlp_ocrq_chunk_best.pkl` | `outputs/metrics/adaptive_fusion_mlp_ocrq_chunk_val_metrics.json` | `outputs/predictions/adaptive_fusion_mlp_ocrq_chunk_val_predictions.jsonl` |
| adaptive_fusion_mlp_ocrq_hybrid | fusion ablation | 0.6430 | 0.8741 | 0.9190 | 0.7415 | no gain | `See [adaptive_fusion_mlp_ocrq_hybrid Commands](#adaptive_fusion_mlp_ocrq_hybrid-commands)` | `See [adaptive_fusion_mlp_ocrq_hybrid Commands](#adaptive_fusion_mlp_ocrq_hybrid-commands)` | `outputs/checkpoints/adaptive_fusion_mlp_ocrq_hybrid/adaptive_fusion_mlp_ocrq_hybrid_best.pkl` | `outputs/metrics/adaptive_fusion_mlp_ocrq_hybrid_val_metrics.json` | `outputs/predictions/adaptive_fusion_mlp_ocrq_hybrid_val_predictions.jsonl` |
| adaptive_fusion_mlp_ocrq_chunkplus | fusion enhancement | 0.6430 | 0.8757 | 0.9206 | 0.7420 | no gain | `See [adaptive_fusion_mlp_ocrq_chunkplus Commands](#adaptive_fusion_mlp_ocrq_chunkplus-commands)` | `See [adaptive_fusion_mlp_ocrq_chunkplus Commands](#adaptive_fusion_mlp_ocrq_chunkplus-commands)` | `outputs/checkpoints/adaptive_fusion_mlp_ocrq_chunkplus/adaptive_fusion_mlp_ocrq_chunkplus_best.pkl` | `outputs/metrics/adaptive_fusion_mlp_ocrq_chunkplus_val_metrics.json` | `outputs/predictions/adaptive_fusion_mlp_ocrq_chunkplus_val_predictions.jsonl` |
| ocr_nv_chunk | OCR-only comparison | 0.4288 | 0.7366 | 0.8342 | 0.5582 | inferior OCR branch | `N/A` | `See [ocr_nv_chunk Commands](#ocr_nv_chunk-commands)` | 无 | `outputs/metrics/ocr_nv_chunk_val_metrics.json` | `outputs/predictions/ocr_nv_chunk_val_predictions.jsonl` |
| visual_colqwen | current best visual-only | 0.6979 | 0.8938 | 0.9333 | 0.7825 | current best visual-only | `N/A` | `See [visual_colqwen Commands](#visual_colqwen-commands)` | 无 | `outputs/metrics/visual_colqwen_val_metrics.json` | `outputs/predictions/visual_colqwen_val_predictions.jsonl` |

说明：

- 当前 strongest OCR-only 是 chunk-level OCR retrieval + rerank，对应的正式统计文件使用 `chunk_rerank` 命名。

## Command Blocks

### BM25 OCR-only Commands

Train Command

```bash
N/A
```

Eval Command

```bash
python -m src.main --mode eval_bm25_val --device cuda:0
```

### Old Visual-only Commands

Train Command

```bash
N/A
```

Eval Command

```bash
python -m src.main --mode eval_visual_val --device cuda:0
```

### adaptive_fusion_ablate_mlp_ocrq Commands

Train Command

```bash
python -m src.main --mode train_adaptive_fusion_ablate_mlp_ocrq --device cuda:0
```
    
Eval Command

```bash
python -m src.main --mode eval_adaptive_fusion_ablate_mlp_ocrq_val --device cuda:0
```

### BGE Chunk OCR-only Commands

Train Command

```bash
N/A
```

Eval Command

```bash
python -m src.main --mode eval_ocr_bge_chunk_rerank_val --device cuda:0
```

### adaptive_fusion_mlp_ocrq_chunk Commands

Train Command

```bash
python -m src.main --mode train_adaptive_fusion_mlp_ocrq_chunk --device cuda:0
```

Eval Command

```bash
python -m src.main --mode eval_adaptive_fusion_mlp_ocrq_chunk_val --device cuda:0
```

### adaptive_fusion_mlp_ocrq_hybrid Commands

Train Command

```bash
python -m src.main --mode train_adaptive_fusion_mlp_ocrq_hybrid --device cuda:0
```

Eval Command

```bash
python -m src.main --mode eval_adaptive_fusion_mlp_ocrq_hybrid_val --device cuda:0
```

### adaptive_fusion_mlp_ocrq_chunkplus Commands

Train Command

```bash
python -m src.main --mode train_adaptive_fusion_mlp_ocrq_chunkplus --device cuda:0
```

Eval Command

```bash
python -m src.main --mode eval_adaptive_fusion_mlp_ocrq_chunkplus_val --device cuda:0
```

### ocr_nv_chunk Commands

Train Command

```bash
N/A
```

Eval Command

```bash
python -m src.main --mode eval_ocr_nv_chunk_val --device cuda:0
```

### visual_colqwen Commands

Train Command

```bash
N/A
```

Eval Command

```bash
python -m src.main --mode eval_visual_colqwen_val --device cuda:0
```

## Short Conclusions

- `adaptive_fusion_mlp_ocrq_hybrid` 未超过 `adaptive_fusion_mlp_ocrq_chunk`
- `adaptive_fusion_mlp_ocrq_chunkplus` 未超过 `adaptive_fusion_mlp_ocrq_chunk`
- `ocr_nv_chunk` 未超过当前 `BGE chunk OCR-only`
- `visual_colqwen` 已成为当前最强单路模型
