# Current Best Models

## Current Best Visual-only

- Model: `visual_colqwen`
- Metric summary:
  - Recall@1 = 0.6979
  - Recall@5 = 0.8938
  - Recall@10 = 0.9333
  - MRR = 0.7825

Eval Command

```bash
python -m src.main --mode eval_visual_colqwen_val --device cuda:0
```

Metrics File

- `outputs/metrics/visual_colqwen_val_metrics.json`

## Current Best OCR-only

- Model: `BGE chunk OCR-only`
- Metric summary:
  - Recall@1 = 0.4712
  - Recall@5 = 0.7295
  - Recall@10 = 0.8024
  - MRR = 0.5810

Eval Command

```bash
python -m src.main --mode eval_ocr_bge_chunk_rerank_val --device cuda:0
```

Metrics File

- `outputs/metrics/ocr_bge_chunk_rerank_val_metrics.json`

## Current Best Old-Fusion

- Model: `adaptive_fusion_mlp_ocrq_chunk`
- Metric summary:
  - Recall@1 = 0.6435
  - Recall@5 = 0.8751
  - Recall@10 = 0.9202
  - MRR = 0.7423

Train Command

```bash
python -m src.main --mode train_adaptive_fusion_mlp_ocrq_chunk --device cuda:0
```

Eval Command

```bash
python -m src.main --mode eval_adaptive_fusion_mlp_ocrq_chunk_val --device cuda:0
```

Checkpoint

- `outputs/checkpoints/adaptive_fusion_mlp_ocrq_chunk/adaptive_fusion_mlp_ocrq_chunk_best.pkl`

Metrics File

- `outputs/metrics/adaptive_fusion_mlp_ocrq_chunk_val_metrics.json`

## Current Research Question

在更强 OCR-free visual retriever 已经很强的情况下，OCR chunk route 是否仍能继续带来页级检索增益？
