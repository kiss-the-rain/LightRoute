当前项目是一个面向资源受限场景的轻量双路 DocVQA 页级检索系统，主数据集为 MP-DocVQA。当前 retrieval 任务定义是：

- 给定一个问题和该文档的全部页面
- 在当前文档内部对页面排序
- 不做跨文档全库检索

## 当前项目阶段

项目已经完成从基础 baseline 到 OCR chunk / visual_colqwen 主线的实验推进，当前状态如下：

- BM25 OCR-only 已完成
- old visual-only 已完成
- Fixed Fusion 与 RRF 已完成
- 旧版 adaptive fusion 与回滚消融已完成
- BGE chunk OCR-only 已完成并验证有效
- `adaptive_fusion_mlp_ocrq_chunk` 已完成并成为旧 visual backbone 下最优 fusion
- `adaptive_fusion_mlp_ocrq_hybrid` 与 `adaptive_fusion_mlp_ocrq_chunkplus` 已完成，但都未超过 `adaptive_fusion_mlp_ocrq_chunk`
- `visual_colqwen` 已完成并成为 current best visual-only
- 新主线 `adaptive_fusion_visual_colqwen_ocr_chunk` 已接线，当前的核心问题是它是否能超过 `visual_colqwen-only`

## Current Best

- Current best visual-only: `visual_colqwen`
- Current best OCR-only: `BGE chunk OCR-only`
- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`

## Current Research Question

当前不再把“超过旧 visual-only”当成主目标。新的核心问题是：

- 在更强 OCR-free visual retriever 已显著变强的情况下，OCR chunk route 是否仍然有额外价值？
- `visual_colqwen + OCR chunk` 是否还能进一步提升？

## 建议在新会话中的工作方式

```text
请基于当前项目继续实现或分析 visual_colqwen + OCR chunk 的融合实验。
要求：
- 不覆盖已有 adaptive_fusion_mlp_ocrq_chunk、visual-only、visual_colqwen-only
- 当前最强单路是 visual_colqwen
- 当前最强旧主线 fusion 是 adaptive_fusion_mlp_ocrq_chunk
- 重点回答在更强 visual backbone 下 OCR chunk 路是否仍有增益
- val 上优先看 Recall@1/5/10、MRR、PageAcc
```
