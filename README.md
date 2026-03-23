# LightRoute

LightRoute is a lightweight dual-route retrieval-augmented DocVQA project for
resource-constrained settings. The repository root is the project root for the
`dual_docvqa` system.

## Scope

- Visual page retrieval branch with a reserved ColPali-style interface
- OCR text retrieval branch with OCR + BM25 page retrieval
- Fusion and adaptive routing for lightweight experimentation
- Retrieval and downstream DocVQA evaluation
- YAML-driven configuration, disk caching, and extensible module boundaries

## Layout

```text
.
├── configs/
├── data/
├── outputs/
├── scripts/
└── src/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --config configs/base.yaml
```
