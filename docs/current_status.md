当前项目围绕 MP-DocVQA 的文档内页级证据检索展开，现阶段已经从“建立可运行 baseline”进入“确认更强 visual backbone 下 OCR 路是否仍有增益”的阶段。

## 当前状态

已完成的主要实验主线包括：

- OCR-only BM25 baseline
- old visual-only baseline（旧 ColPali 系列主线）
- Fixed Fusion baseline
- RRF Fusion baseline
- `adaptive_fusion_v1`
- `adaptive_fusion_v2`
- rollback ablations
  - `adaptive_fusion_ablate_mlp`
  - `adaptive_fusion_ablate_q`
  - `adaptive_fusion_ablate_ocrq`
  - `adaptive_fusion_ablate_lex`
- pairwise combination experiments
  - `adaptive_fusion_ablate_mlp_q`
  - `adaptive_fusion_ablate_mlp_ocrq`
  - `adaptive_fusion_ablate_mlp_lex`
- OCR route upgrades
  - BGE page-level OCR retrieval
  - BGE chunk OCR retrieval / rerank
  - hybrid OCR retrieval
  - NV-Embed-v2 OCR chunk retrieval
- new visual backbone
  - `visual_colqwen`
- new fusion branch wiring
  - `adaptive_fusion_visual_colqwen_ocr_chunk`

## 当前 best 结论

- Current best visual-only: `visual_colqwen`
- Current best OCR-only: `BGE chunk OCR-only`
- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`
- Current best overall single-route page retriever: `visual_colqwen`

## 关键实验结论

- old visual-only baseline 曾经是较强 baseline，但已不再是当前最强视觉主干。
- BM25 OCR-only 应保留为基础 OCR baseline。
- Fixed Fusion 与 RRF 可以作为简单融合 baseline，但不足以充分利用双路互补信息。
- `adaptive_fusion_ablate_mlp_ocrq` 曾是旧版最优 fusion，但已被后续 chunk 主线超过。
- page-level BGE OCR retrieval 效果不理想。
- chunk-level OCR retrieval / rerank 明显更有效，说明 OCR 路的关键增益来自局部块建模，而不是整页表示。
- `adaptive_fusion_mlp_ocrq_chunk` 超过了旧版 best fusion，说明 chunk-level OCR route 接回 fusion 是有效的。
- `adaptive_fusion_mlp_ocrq_hybrid` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `adaptive_fusion_mlp_ocrq_chunkplus` 未超过 `adaptive_fusion_mlp_ocrq_chunk`，说明进一步复杂化 feature 层收益有限。
- `ocr_nv_chunk` 在 Recall@5/10 上略高于 BGE chunk OCR-only，但 top1 与 MRR 更低，因此不适合作为新的主 OCR 分支。
- `visual_colqwen` 已显著超过旧 visual-only 与所有旧 fusion 结果。

## 当前主线变化

第一阶段结论：

- 在旧 visual backbone 下，chunk-level OCR route + adaptive fusion 是有效的。
- 这证明 OCR 路增强的关键在于 chunk-level 建模。

第二阶段结论：

- 更强的 OCR-free visual retriever `visual_colqwen` 显著提升了页级检索性能。
- 当前最强 visual-only 已明显超过旧双路 fusion 主线。

当前核心研究问题：

- 在更强 visual backbone 下，OCR chunk route 是否仍然具有额外价值？
- `visual_colqwen + OCR chunk` 是否还能进一步超过 `visual_colqwen-only`？

## 下一步

- 第一优先：训练并验证 `adaptive_fusion_visual_colqwen_ocr_chunk`
- 第二优先：如果该新 fusion 不能超过 `visual_colqwen-only`，则说明 OCR 路在强视觉 backbone 下的增益可能已经接近饱和
- 后续可转向：
  - evidence chunk selection
  - lightweight reader
  - retrieval-then-read 两阶段 DocVQA

## 文档索引

- 结果总表与结论汇总见 [experiment_summary.md](/Users/Lawrence/Code/PycharmProjects/LightRoute/docs/experiment_summary.md)
- 当前 best 模型列表见 [current_best_models.md](/Users/Lawrence/Code/PycharmProjects/LightRoute/docs/current_best_models.md)
- 当前研究方向见 [research_direction.md](/Users/Lawrence/Code/PycharmProjects/LightRoute/docs/research_direction.md)
- 负结果整理见 [negative_results.md](/Users/Lawrence/Code/PycharmProjects/LightRoute/docs/negative_results.md)
