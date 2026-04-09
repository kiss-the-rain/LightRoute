当前项目的实验协议围绕“文档内页级 evidence retrieval”展开，重点是保证各条主线在 val 上可以被直接比较，并且不把未完成实验写成既成事实。

## Data and Evaluation Scope

- 训练集：`data/processed/mpdocvqa/train.jsonl`
- 验证集：`data/processed/mpdocvqa/val.jsonl`
- 测试集：`data/processed/mpdocvqa/test.jsonl`

评估原则：

- 所有主要比较优先在 val 上完成
- test 只用于推理输出，不做监督指标汇报
- retrieval 范围固定为 document-internal page ranking

## Current Main Comparison Set

当前最重要的对比对象是：

1. `visual_colqwen-only`
2. `adaptive_fusion_mlp_ocrq_chunk`
3. `adaptive_fusion_visual_colqwen_ocr_chunk`

此外保留以下对照：

- `BM25 OCR-only`
- old `visual-only`
- `adaptive_fusion_ablate_mlp_ocrq`
- `BGE chunk OCR-only`
- Fixed Fusion
- RRF
- `adaptive_fusion_mlp_ocrq_hybrid`
- `adaptive_fusion_mlp_ocrq_chunkplus`
- `ocr_nv_chunk`

## Current Experimental Conclusions

- old visual-only 曾是较强 baseline，但已不是当前最强视觉主干。
- chunk-level OCR retrieval / rerank 明显优于 page-level OCR retrieval。
- `adaptive_fusion_mlp_ocrq_chunk` 是旧 visual backbone 条件下的最优 fusion。
- `adaptive_fusion_mlp_ocrq_hybrid` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `adaptive_fusion_mlp_ocrq_chunkplus` 未超过 `adaptive_fusion_mlp_ocrq_chunk`。
- `visual_colqwen` 已明显超过所有旧 fusion 结果。

## Current Best Strategy

- Current best visual-only: `visual_colqwen`
- Current best OCR-only: `BGE chunk OCR-only`
- Current best old-backbone fusion: `adaptive_fusion_mlp_ocrq_chunk`

## Next-Step Protocol

第一优先：

- 训练与评估 `adaptive_fusion_visual_colqwen_ocr_chunk`

判定标准：

- 若其超过 `visual_colqwen-only`，说明 OCR chunk 路在强视觉 backbone 下仍有互补性
- 若其未超过 `visual_colqwen-only`，说明 OCR 路在当前强视觉 backbone 下的增益可能接近饱和

第二优先：

- 若未超过，则将项目重点转向：
  - evidence chunk selection
  - lightweight reader
  - retrieval-then-read 两阶段 DocVQA

## Reporting Discipline

- 对未超过的分支明确写“未超过”
- 对当前最强模型明确写“current best”
- 不虚构 test 指标
- 不删除旧结果，只更新其状态与结论
