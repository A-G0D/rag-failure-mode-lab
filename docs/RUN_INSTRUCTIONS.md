# Run Instructions

Python 3.10+. Dependencies: `numpy`, `scikit-learn`, `pyyaml`, `pytest` (all
available offline). No network, no GPU, no API key.

## Setup

```powershell
cd P2-rag
$env:PYTHONPATH = "$PWD"          # bash: export PYTHONPATH="$PWD"
```

## Regenerate the synthetic dataset (optional, already shipped)

```powershell
python scripts/build_dataset.py
# -> docs/datasets/erp_corpus.jsonl, erp_queries.jsonl  (mirrored to tests/fixtures/)
```

## Single query

```powershell
python -m src.cli run "What triggers a backorder during allocation?" `
    --retriever hybrid --generator template
```

`--retriever` ∈ {bm25, embedding, hybrid}. `--generator` ∈ {extractive, template}.

## Evaluate all combinations

```powershell
python -m src.cli eval-suite          # prints per-combo aggregate metrics as JSON
```

## Comparative report

```powershell
python -m src.cli report              # writes eval/ artifacts and prints the report
python scripts/run_comparative_report.py   # standalone equivalent
```

Outputs:
- `eval/COMPARISON.md`: summary table, latency analysis, failure breakdown,
  baseline-vs-improved headline, recommendation.
- `eval/stress_results.json`: full per-query + aggregate results for every combo.
- `eval/comparison.json`: the baseline-vs-improved delta object.

## Tests

```powershell
python -m pytest tests -q
```

Covers each module (unit), the pipeline (integration), metric correctness
(evaluation), and reproducibility (determinism, 3 runs).

## Inspect a trace as HTML

```python
from src.logging_layer.debug_hooks import export_query_trace_html
# events come from a JSONLFileEventLog / InMemoryEventLog used by RAGPipeline
export_query_trace_html("Q_N_03", events, "logs/Q_N_03.html")
```
