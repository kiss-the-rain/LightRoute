这份文档记录当前项目中高频、可复用的 Codex prompt 模式。目标不是保存花哨模板，而是保留真正适合当前工程状态的工作指令。

**OCR-only**

适用场景：
- 实现或调试 BM25 baseline
- 检查 OCR JSON 解析
- 验证 `evidence_pages` 与预测页号是否一致

推荐 prompt：

```text
请基于当前 MP-DocVQA processed 数据，实现或修复 OCR-only（BM25）页级检索。
要求：
- retrieval 范围只在当前文档内部
- 使用 evidence_pages 作为全局页号监督
- 输出 val prediction 与 metrics
- 不引入 dense retrieval
- 优先保证最小可运行
```

```text
请检查 OCR-only baseline 的页号映射是否正确，重点核对：
- evidence_pages
- ocr_paths 顺序
- pred_page_ids
- retrieval metrics 计算
如果有错误，直接修改代码。
```

**Visual-only**

适用场景：
- 接入 visual retriever stub
- 替换当前 surrogate visual encoding
- 构建 visual-only validation baseline

推荐 prompt：

```text
请为当前项目实现 visual-only 页级检索 baseline。
要求：
- 保持 MP-DocVQA 当前 processed 数据格式不变
- 检索仍然是文档内页级排序
- 先实现最小可运行版本
- 输出 val prediction 和 retrieval metrics
```

```text
请检查当前 visual retriever 接口是否适合后续替换成真实 ColPali 风格模块。
如果接口写死了，请重构，但不要提前加入复杂训练逻辑。
```

**Fusion**

适用场景：
- Fixed Fusion
- RRF Fusion
- Adaptive Fusion
- question-aware / quality-aware 特征扩展

推荐 prompt：

```text
请在当前 OCR-only 和 visual-only baseline 已存在的前提下，实现 fixed fusion 与 RRF fusion。
要求：
- 输入是文档内页级候选
- 输出统一 page ranking
- 保持现有 prediction 和 metrics 输出格式
- 不改动 processed 数据格式
```

```text
请扩展当前 feature_builder 和 adaptive fusion，使其支持：
- question-aware features
- quality-aware features
- routing weights
但不要破坏现有 OCR-only / visual-only baseline。
```

**数据构建**

适用场景：
- 修改 `sample_builder.py`
- 调整 MP-DocVQA 样本格式
- 增加 build report 或性能优化

推荐 prompt：

```text
请只修改 src/data/sample_builder.py，使其严格符合 MP-DocVQA 当前字段映射规范。
要求：
- evidence_pages 使用全局页号
- 输出格式不变
- 不改动其它模块
```

**实验核查**

适用场景：
- 改完代码后快速确认当前阶段状态
- 核对 docs 与代码是否一致

推荐 prompt：

```text
请根据当前代码和配置，核查项目当前实验状态。
要求：
- 不虚构结果
- 明确哪些 baseline 已实现
- 明确哪些文件已接通 main.py
- 明确下一步应该做什么
```

**Pairwise Combination Experiments**

适用场景：
- 在 `ablate_mlp` 底座上继续做两两组合
- 验证哪一类增强与升级后的 MLP 兼容
- 保持其它实验版本不变

推荐 prompt：

```text
请基于当前 adaptive fusion 回滚消融结果，继续实现两两组合实验。
要求：
- 以 ablate_mlp 为底座
- 只新增一类增强：question-aware / OCR-quality / lexical-overlap 三选一
- 不破坏已有 v1、v2 和单项 ablation
- 保持 feature order 固定
- 保持现有训练、推理、评估输出格式不变
```

```text
请为当前项目补充 adaptive fusion 的 pairwise combination 实验入口：
- ablate_mlp_q
- ablate_mlp_ocrq
- ablate_mlp_lex
要求：
- 每个版本单独 checkpoint 和 metrics 文件
- 不把尚未运行的实验写进 docs 的完成结论
```

**Current Fusion Context**

当前文档中的 prompt 需要基于下面的实验现实来写：

- 当前 best visual-only：`visual_colqwen`
- 当前 best OCR-only：`BGE chunk OCR-only`
- 当前 best old-backbone fusion：`adaptive_fusion_mlp_ocrq_chunk`
- `adaptive_fusion_ablate_mlp_ocrq` 是历史上的 old best fusion，不再是当前 fusion 主线

说明：

- `adaptive_fusion_mlp_ocrq_chunk` 代表“旧 visual backbone + chunk-level OCR route”的最优融合版本
- `visual_colqwen` 已明显强于当前所有旧 fusion 结果
- 因此后续扩展融合模型时，优先问题不再是继续堆叠旧 `mlp_ocrq` 特征，而是验证 `visual_colqwen + OCR chunk` 是否仍有增益

**Final Model Selection Rule**

当多个版本指标接近时，优先选择：

1. Retrieval metrics 更强者
2. 训练更稳定者
3. 结构更简洁者

补充约定：

- 不要把 old best fusion 写成 current best overall
- 不要再把“超过旧 visual-only”当成当前核心目标
- 当前更合理的 prompt 应直接围绕 `visual_colqwen + OCR chunk` 的互补性验证展开
