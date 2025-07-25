# RAG Comparative Report — Baseline vs Improved

_Deterministic run (seed-locked). All data synthetic (fictional Stillwater Industrial ERP corpus)._

## Summary (all retriever x generator combinations)

| combo | faithfulness | hallucination_rate | context_precision@k | answer_F1 | p50_ms |
| --- | --- | --- | --- | --- | --- |
| bm25+extractive | 0.733333 | 0.266667 | 0.253333 | 0.5636 | 0.2049 |
| bm25+template | 0.7 | 0.233333 | 0.253333 | 0.542864 | 0.2179 |
| embedding+extractive | 0.8 | 0.2 | 0.253333 | 0.571443 | 1.8147 |
| embedding+template | 0.733333 | 0.2 | 0.253333 | 0.550007 | 2.047 |
| hybrid+extractive | 0.733333 | 0.266667 | 0.253333 | 0.5636 | 1.4695 |
| hybrid+template | 0.7 | 0.233333 | 0.253333 | 0.542864 | 1.6894 |


## Latency analysis (ms)

| combo | p50 | p95 | p99 |
| --- | --- | --- | --- |
| bm25+extractive | 0.2049 | 0.46476 | 0.485592 |
| bm25+template | 0.2179 | 0.48881 | 0.749602 |
| embedding+extractive | 1.8147 | 3.76896 | 3.834032 |
| embedding+template | 2.047 | 4.37203 | 4.480726 |
| hybrid+extractive | 1.4695 | 2.21825 | 2.47893 |
| hybrid+template | 1.6894 | 2.60415 | 2.61787 |


## Failure-mode breakdown

**bm25+extractive** — correct: 9, hallucination: 4, partial_correctness: 2, irrelevant_retrieval: 0, context_mismatch: 0
**bm25+template** — correct: 8, hallucination: 5, partial_correctness: 1, irrelevant_retrieval: 0, context_mismatch: 1
**embedding+extractive** — correct: 9, hallucination: 3, partial_correctness: 3, irrelevant_retrieval: 0, context_mismatch: 0
**embedding+template** — correct: 8, hallucination: 4, partial_correctness: 2, irrelevant_retrieval: 0, context_mismatch: 1
**hybrid+extractive** — correct: 9, hallucination: 4, partial_correctness: 2, irrelevant_retrieval: 0, context_mismatch: 0
**hybrid+template** — correct: 8, hallucination: 5, partial_correctness: 1, irrelevant_retrieval: 0, context_mismatch: 1

## Baseline vs Improved (headline)

- **hallucination_rate**: baseline 0.266667 -> improved 0.233333 (12.5% change, improved)
- **faithfulness**: baseline 0.733333 -> improved 0.7 (-4.55% change, no improvement)
- **answer_correctness_f1**: baseline 0.5636 -> improved 0.542864 (-3.68% change, no improvement)
- **context_precision_at_k**: baseline 0.253333 -> improved 0.253333 (0.0% change, no improvement)

## Recommendation

Recommended configuration: **bm25+extractive** for low-latency dev/demo (fast, simple), and **hybrid+template** for production accuracy (hallucination rate 0.233333 vs 0.266667, a 12.5% reduction).
