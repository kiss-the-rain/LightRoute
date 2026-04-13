# Research Direction

## Stage-1

- 在旧 visual backbone 下，chunk OCR + adaptive fusion 是有效的。
- `adaptive_fusion_mlp_ocrq_chunk` 超过了旧版 `adaptive_fusion_ablate_mlp_ocrq`，说明 OCR 路增强的关键在于 chunk-level 建模。

## Stage-2

- 更强的 `visual_colqwen` 显著提升了单路页级检索性能。
- `visual_colqwen` 已明显超过 old visual-only，也明显超过当前所有旧 fusion 结果。

## Current Stage

- 当前核心问题不再是“超过 old visual-only”。
- 当前阶段的重点是研究 `visual_colqwen + OCR chunk` 是否仍有互补性。
- 若该新主线不能超过 `visual_colqwen-only`，则说明 OCR 路在强 visual backbone 下的边际收益可能已经有限。

## Next Step

Next priority:

1. `train_adaptive_fusion_visual_colqwen_ocr_chunk`
2. `eval_adaptive_fusion_visual_colqwen_ocr_chunk_val`

对应命令：

```bash
python -m src.main --mode train_adaptive_fusion_visual_colqwen_ocr_chunk --device cuda:0
python -m src.main --mode eval_adaptive_fusion_visual_colqwen_ocr_chunk_val --device cuda:0
```
