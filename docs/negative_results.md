# Negative Results

本页只记录已经完成、但未成为当前主线的实验分支。所有结论均保持客观，不把未超过写成有效提升。

## Model / Branch: page-level BGE OCR retrieval

- short description:
  - 以整页 OCR 文本为表示的 dense OCR retrieval 路线。
- result summary:
  - 相比 chunk-level OCR retrieval / rerank，没有形成当前 strongest OCR-only 结果。
- conclusion:
  - page-level BGE OCR retrieval 不理想，OCR 路更有效的增强方向是 chunk-level 建模。

## Model / Branch: adaptive_fusion_mlp_ocrq_hybrid

- short description:
  - 在旧 fusion 主线下测试 hybrid OCR route 的融合版本。
- result summary:
  - Recall@1 = 0.6430
  - Recall@5 = 0.8741
  - Recall@10 = 0.9190
  - MRR = 0.7415
- conclusion:
  - 未超过 `adaptive_fusion_mlp_ocrq_chunk`，当前最有效的 OCR 增强主线仍然是 chunk-level OCR retrieval / rerank。

## Model / Branch: adaptive_fusion_mlp_ocrq_chunkplus

- short description:
  - 在 chunk fusion 上继续加入 chunk-aware 与 question-aware feature 增强。
- result summary:
  - Recall@1 = 0.6430
  - Recall@5 = 0.8757
  - Recall@10 = 0.9206
  - MRR = 0.7420
- conclusion:
  - 未超过 `adaptive_fusion_mlp_ocrq_chunk`，说明 fusion 特征层面的进一步复杂化收益已经有限。

## Model / Branch: ocr_nv_chunk

- short description:
  - 使用 NV-Embed-v2 的 OCR chunk-only 对比分支。
- result summary:
  - Recall@1 = 0.4288
  - Recall@5 = 0.7366
  - Recall@10 = 0.8342
  - MRR = 0.5582
- conclusion:
  - 未超过当前 strongest OCR-only `BGE chunk OCR-only`。Recall@5/10 略高，但 top1 与 MRR 明显更低，因此不适合作为新的主 OCR 分支。
