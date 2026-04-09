当前项目的主数据集是 MP-DocVQA，processed 数据位于：
- `data/processed/mpdocvqa/train.jsonl`
- `data/processed/mpdocvqa/val.jsonl`
- `data/processed/mpdocvqa/test.jsonl`

每条样本当前统一为下面的结构：

```json
{
  "qid": "val_49153",
  "question": "What is the 'actual' value per 1000, during the year 1975?",
  "answers": ["12.3"],
  "answer": "12.3",
  "doc_id": "pybv0228",
  "evidence_pages": [80],
  "page_count": 95,
  "image_paths": ["data/raw/mpdocvqa/images/pybv0228_p0.jpg"],
  "ocr_paths": ["data/raw/mpdocvqa/ocr/pybv0228_p0.json"],
  "meta": {}
}
```

关键字段说明：

- `qid`
  - 当前格式为 `{split}_{question_id}`，例如 `train_337`
  - 用于 prediction、metrics、case study 和错误分析的主键

- `question`
  - 原始问题文本，已做基本 `strip()`
  - OCR-only、visual-only、fusion 检索都直接使用该字段

- `answers`
  - 答案候选列表
  - 优先来自 `valid_answers`
  - 去重但保持顺序

- `answer`
  - 当前取 `answers[0]` 作为主答案
  - 用于后续 DocVQA EM/F1/ANLS 计算

- `doc_id`
  - 文档 id，例如 `xnbl0037`
  - 用于收集整份文档的所有页面与 OCR 文件

- `evidence_pages`
  - 当前使用整份文档的全局页号，即 `answer_page`
  - 不是 `answer_page_idx`
  - 这是当前项目最重要的数据约定之一
  - 所有 retrieval metrics 都按这个全局页号评估

- `page_count`
  - 等于 `len(image_paths)`
  - 当前不直接信任 `imdb_doc_pages` 或 `total_doc_pages`

- `image_paths`
  - 整份文档的全部页面路径
  - 文件名规则为 `{doc_id}_p{page_idx}.jpg`
  - 当前按整数页号排序，而不是字符串排序

- `ocr_paths`
  - 整份文档的全部 OCR 文件路径
  - 文件名规则为 `{doc_id}_p{page_idx}.json`
  - 与 `image_paths` 一一对应，并按页号排序

- `meta`
  - 保留原始 MP-DocVQA 中的重要字段
  - 至少包括：
    - `dataset`
    - `split`
    - `raw_question_id`
    - `raw_doc_id`
    - `answer_page`
    - `answer_page_idx`
    - `imdb_doc_pages`
    - `total_doc_pages`
    - `pages`
    - `extra_info`

当前 retrieval 设定是“文档内页级检索”：
- 给定问题和该文档的全部页面
- 对 `0 ... page_count-1` 或等价全局页号进行排序
- 不做全库跨文档 BM25

因此：
- `evidence_pages` 必须解释为当前文档内部目标页号
- `ocr_paths` 的顺序必须和页面页号严格对齐
- 检索预测中的 `pred_page_ids` 应该输出页号，而不是 `doc_id_p80` 这类字符串
