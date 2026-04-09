# Current Best Models

本页只记录当前项目在各条主线上的 current best 模型，以及不建议继续推进的分支。

## Current Best

- Current best visual-only: `visual_colqwen`
  - Recall@1 = 0.6978985926354347
  - Recall@5 = 0.8937728937728938
  - Recall@10 = 0.9332947754000386
  - MRR = 0.7825204800768722
  - PageAcc = 0.6978985926354347

- Current best OCR-only: `BGE chunk OCR-only`
  - Recall@1 ≈ 0.4712
  - Recall@5 ≈ 0.7295
  - Recall@10 ≈ 0.8024
  - MRR ≈ 0.5810

- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`
  - Recall@1 = 0.6435319066898014
  - Recall@5 = 0.8750722961249277
  - Recall@10 = 0.9201850780798149
  - MRR = 0.7423099874227729
  - PageAcc = 0.6435319066898014

## Interpretation

- `visual_colqwen` 是当前全项目最强单路页级检索模型。
- `adaptive_fusion_mlp_ocrq_chunk` 是旧 visual backbone 条件下的最优 fusion 版本。
- 但 `adaptive_fusion_mlp_ocrq_chunk` 已不再是当前全项目最佳模型，因为 `visual_colqwen` 已经明显更强。

## Not Recommended to Prioritize

- page-level BGE OCR retrieval
  - 效果不理想，不是当前主线。
- `adaptive_fusion_mlp_ocrq_hybrid`
  - 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `adaptive_fusion_mlp_ocrq_chunkplus`
  - 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `ocr_nv_chunk`
  - Recall@5/10 略高，但 top1 与 MRR 明显更低，不适合作为新的主 OCR 路。
- Fixed Fusion / RRF
  - 作为 baseline 有价值，但不是当前最优方案。

## Next Recommendation

- 第一优先：验证 `adaptive_fusion_visual_colqwen_ocr_chunk`
- 目标：直接回答在更强 visual backbone 下，OCR chunk 路是否仍有互补增益
- 若该新 fusion 不能超过 `visual_colqwen-only`，则后续更合理的方向是：
  - evidence chunk selection
  - lightweight reader
  - retrieval-then-read 两阶段 DocVQA
