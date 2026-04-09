LightRoute 当前面向的任务是轻量双路检索增强 DocVQA，目标是在资源受限场景下完成多页文档中的证据页检索，并为后续问答生成提供稳定的 evidence ranking。

当前主数据集是 MP-DocVQA。项目统一采用“给定问题 + 给定文档全部页面，在文档内部做 page ranking”的任务定义，不做跨文档全库检索。

## 主线变化

旧阶段目标：

- 建立 OCR-only、visual-only、simple fusion、adaptive fusion 的可运行 baseline
- 在旧 visual backbone 下验证 OCR 路增强是否有效

当前阶段目标：

- 已经确认 chunk-level OCR route 在旧 visual backbone 下是有效增强方向
- 已经确认更强的 OCR-free visual backbone `visual_colqwen` 显著提升了页级检索性能
- 当前最重要的问题不再是“能否超过旧 visual-only”，而是：
  - 在更强 visual backbone 下，OCR chunk 路是否仍然有额外价值？
  - `visual_colqwen + OCR chunk` 是否还能进一步提升？

## 当前 best 表述

- Current best visual-only: `visual_colqwen`
- Current best OCR-only: `BGE chunk OCR-only`
- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`

需要明确：

- `visual_colqwen` 是当前全项目最强单路页级检索模型
- `adaptive_fusion_mlp_ocrq_chunk` 是旧 visual backbone 条件下最优 fusion 版本
- 但它已不再是当前全项目最佳模型

## 当前下一步

- 第一优先：验证 `adaptive_fusion_visual_colqwen_ocr_chunk`
- 第二优先：如果该新 fusion 不能超过 `visual_colqwen-only`，则将后续工作重点转向：
  - evidence chunk selection
  - lightweight reader
  - retrieval-then-read 两阶段 DocVQA
