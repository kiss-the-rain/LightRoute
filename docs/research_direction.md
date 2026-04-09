# Research Direction

## Stage-1 Conclusion

- 在旧 visual backbone 下，chunk-level OCR route 接入 adaptive fusion 是有效的。
- `adaptive_fusion_mlp_ocrq_chunk` 超过了旧版 `adaptive_fusion_ablate_mlp_ocrq`，说明 OCR 路增强的关键在于 chunk-level 建模。

## Stage-2 Conclusion

- 更强的 OCR-free visual retriever `visual_colqwen` 已显著提高页级检索性能。
- `visual_colqwen` 不仅超过旧 visual-only，也明显超过当前所有旧 fusion 结果。

## Current Core Question

当前项目不再把“超过旧 visual-only”作为核心目标。新的核心问题是：

- 更强 OCR-free visual retriever 已经显著提高页级检索性能；
- 下一步重点是研究：
  - `visual_colqwen + OCR chunk` 是否仍然具有互补性
  - 若无明显增益，是否说明 OCR 路在强视觉 backbone 下的价值已接近饱和

## Immediate Priority

- 新增并测试 `adaptive_fusion_visual_colqwen_ocr_chunk`
- 用它与以下三类结果直接比较：
  - `visual_colqwen-only`
  - `adaptive_fusion_mlp_ocrq_chunk`
  - `BGE chunk OCR-only`

## Decision Rule

- 若 `adaptive_fusion_visual_colqwen_ocr_chunk` 超过 `visual_colqwen-only`：
  - 说明 OCR chunk 路在强视觉 backbone 下仍有稳定互补价值
  - 可继续沿该双路主线细化

- 若 `adaptive_fusion_visual_colqwen_ocr_chunk` 未超过 `visual_colqwen-only`：
  - 说明 OCR 路在强视觉 backbone 下的边际增益有限
  - 后续研究更适合转向：
    - evidence chunk selection
    - lightweight reader
    - retrieval-then-read 两阶段 DocVQA 系统
