# Metrics

All metrics are pure, deterministic functions in `src/evaluation/metrics.py`.
A "claim" is one sentence of the generated answer. The **grounding** is the set
of content tokens (stopwords removed) from the gold answer plus the gold
(expected) context documents.

## Faithfulness  (higher is better, range [0, 1])

```
faithfulness = grounded_claims / total_claims
```

A claim is *grounded* if at least 60% of its content tokens appear in the
grounding. An abstention scores 0.0 (it makes no grounded claim).

## Hallucination Rate  (lower is better, range [0, 1])

```
hallucination_rate = ungrounded_claims / total_claims
```

The complement of faithfulness over non-abstaining answers. An abstention is
**not** a hallucination and scores 0.0.

## Context Precision@k  (higher is better, range [0, 1])

```
context_precision@k = |relevant_docs in top-k| / k
```

where the relevant set is the query's `expected_context_ids`.

## Answer Correctness — token-overlap F1  (higher is better, range [0, 1])

```
precision = |A ∩ G| / |A|        recall = |A ∩ G| / |G|
F1        = 2 * precision * recall / (precision + recall)
```

`A` = answer content tokens, `G` = gold-answer content tokens. An abstention or
empty answer scores 0.0.

## Latency percentiles  (lower is better, milliseconds)

`p50 / p95 / p99` of per-query wall-clock pipeline latency, via linear
interpolation between order statistics. Reported for trade-off analysis;
excluded from the determinism contract because wall-clock time is not
reproducible.

## Failure taxonomy (counts per configuration)

| Mode | Meaning |
|------|---------|
| `correct` | matches gold answer and not fabricating |
| `hallucination` | confident answer with ungrounded claims |
| `partial_correctness` | grounded but only partially matches gold |
| `irrelevant_retrieval` | no expected context document was retrieved |
| `context_mismatch` | relevant context retrieved but answer fails to use it (e.g. abstention) |

The report computes `bm25+extractive` (baseline) vs `hybrid+template` (improved)
deltas. The metric that moves is the hallucination rate, which drops.
