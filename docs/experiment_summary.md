# Experiment Summary

本页汇总当前项目已完成的主要 retrieval / fusion 实验、核心指标与阶段性结论。所有结果均基于当前项目已完成的 val 实验记录；未完成的分支不写入已完成结论。

## Overall Table

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 OCR-only | 0.3902 | 0.6933 | 0.8043 | 0.5209 | baseline |
| old visual-only | 0.6428 | 0.8764 | 0.9208 | 0.7420 | old baseline |
| adaptive_fusion_ablate_mlp_ocrq | 0.6424 | 0.8747 | 0.9202 | 0.7415 | old best fusion |
| BGE chunk OCR-only | 0.4712 | 0.7295 | 0.8024 | 0.5810 | strongest OCR-only |
| adaptive_fusion_mlp_ocrq_chunk | 0.6435 | 0.8751 | 0.9202 | 0.7423 | best old fusion |
| adaptive_fusion_mlp_ocrq_hybrid | 0.6430 | 0.8741 | 0.9190 | 0.7415 | no gain |
| adaptive_fusion_mlp_ocrq_chunkplus | 0.6430 | 0.8757 | 0.9206 | 0.7420 | no gain |
| ocr_nv_chunk | 0.4288 | 0.7366 | 0.8342 | 0.5582 | inferior OCR branch |
| visual_colqwen | 0.6979 | 0.8938 | 0.9333 | 0.7825 | current best visual-only |

## Branch-by-Branch Notes

### OCR baselines

- `BM25 OCR-only` 是基础 OCR baseline，应继续保留用于最小对照。
- page-level BGE OCR retrieval 效果不理想。
- `BGE chunk OCR-only` 明显优于 BM25，也优于 page-level dense OCR，说明 OCR 路的有效增强关键在于 chunk-level 建模。
- `ocr_nv_chunk` 的 Recall@5/10 略高，但 Recall@1 与 MRR 明显低于 `BGE chunk OCR-only`，因此不作为新的主 OCR 分支。

### Visual baselines

- old `visual-only`（旧 ColPali 系列主线）曾是强 baseline，但已不是当前最强视觉模型。
- `visual_colqwen` 目前是 current best visual-only，也是当前全项目最强单路页级检索模型。

### Fusion baselines

- Fixed Fusion 与 RRF 已完成，但没有成为最终最佳方案。
- 它们可作为简单融合 baseline，但不足以充分利用双路互补信息。

### Adaptive fusion evolution

- `adaptive_fusion_ablate_mlp_ocrq` 曾是旧版最优 fusion。
- `adaptive_fusion_mlp_ocrq_chunk` 已正式超过 `adaptive_fusion_ablate_mlp_ocrq`，说明 chunk-level OCR route 接回 fusion 是有效的。
- `adaptive_fusion_mlp_ocrq_hybrid` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `adaptive_fusion_mlp_ocrq_chunkplus` 也未超过 `adaptive_fusion_mlp_ocrq_chunk`，说明在当前设置下，进一步复杂化 fusion feature 的收益已经有限。

## Current Best Models

- Current best visual-only: `visual_colqwen`
- Current best OCR-only: `BGE chunk OCR-only`
- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`

需要明确的是：

- `visual_colqwen` 已明显超过当前所有旧 fusion 结果。
- `adaptive_fusion_mlp_ocrq_chunk` 仍然是旧 visual backbone 条件下最优 fusion 版本。
- 但 `adaptive_fusion_mlp_ocrq_chunk` 已不再是当前全项目最佳模型。

## Current Research Question

当前项目的核心问题已经从“是否能超过旧 visual-only”更新为：

- 在更强 OCR-free visual backbone 已经很强的情况下，OCR chunk route 是否还能继续带来额外页级检索收益？
- `adaptive_fusion_visual_colqwen_ocr_chunk` 是否能够超过 `visual_colqwen-only`？

## Next Step

- 第一优先：新增并验证 `adaptive_fusion_visual_colqwen_ocr_chunk`
- 第二优先：如果它不能超过 `visual_colqwen-only`，则说明 OCR 路在强视觉 backbone 下的边际价值可能接近饱和
- 后续研究可转向：
  - evidence chunk selection
  - lightweight reader
  - retrieval-then-read 两阶段 DocVQA
