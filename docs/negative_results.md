# Negative Results

本页记录当前项目中已经完成、但未成为主线的实验结果。目的不是否定这些分支，而是明确它们在当前设置下未超过更强基线。

## OCR Route

- page-level BGE OCR retrieval 不理想。
  - 结论：OCR 路的主要收益不是来自整页 dense 表示，而是来自 chunk-level 建模。

- `ocr_nv_chunk` 未超过 `BGE chunk OCR-only`。
  - Recall@5/10 略高，但 Recall@1 与 MRR 明显更低。
  - 结论：更通用的 embedding 不等于在 DocVQA OCR chunk 页级检索上更强。

## Fusion Route

- Fixed Fusion 未成为最终最佳方案。
- RRF 未成为最终最佳方案。
  - 结论：简单融合可作为 baseline，但不足以充分利用双路互补信息。

- `adaptive_fusion_mlp_ocrq_hybrid` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
  - 结论：当前最有效的 OCR 增强主线仍然是 chunk-level OCR retrieval / rerank。

- `adaptive_fusion_mlp_ocrq_chunkplus` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
  - 结论：chunk-aware + question-aware feature 增强没有继续提升 top1 / MRR。
  - 这说明 fusion 特征层面的进一步复杂化收益已经有限，当前瓶颈更可能来自 OCR 路本身的信息增益上限。
