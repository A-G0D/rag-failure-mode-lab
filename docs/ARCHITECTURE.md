# Architecture

Five packages under `src/`, each with a narrow contract; the evaluation package
composes the rest. Data flows top to bottom.

## Data flow

```
erp_corpus.jsonl (corpus)       erp_queries.jsonl (queries + gold answers)
        |                                   |
        v                                   v
+------------------- input_layer -------------------+
|  schema (Document/Query/RetrievedDoc)             |
|  validation (length/empty/enum constraints)       |
|  stress_taxonomy (adversarial/ambiguous/long-ctx) |
|  dataset_loader (deterministic, id-sorted)        |
+---------------------------------------------------+
        |                                   |
        |  Document[]                        |  Query[]
        v                                   |
+---------------- abstraction ----------------+     |
|  Retriever ABC                              |     |
|   - BM25Retriever     (pure Python Okapi)   |     |
|   - EmbeddingRetriever (TF-IDF + cosine)    |     |
|   - HybridRetriever   (weighted fusion)     |     |
|  Generator ABC                              |     |
|   - ExtractiveGenerator (baseline)          |     |
|   - TemplateGenerator   (grounding-aware)   |     |
+---------------------------------------------+     |
        |                                           |
        v                                           v
+------------------ core_logic: RAGPipeline --------------------+
|  set_seed -> retrieve (retry) -> generate -> trace each stage |
+---------------------------------------------------------------+
        |                                   |
        |  PipelineResult                   |  trace events
        v                                   v
+-------------- evaluation --------------+  +------- logging --------+
|  failure_taxonomy (4 modes)           |  |  event_schema          |
|  metrics (faithfulness, ctxP@k,       |  |  event_log (mem+JSONL) |
|           hallucination, F1)          |  |  debug_hooks (HTML)    |
|  benchmark (per-query + aggregate)    |  +------------------------+
|  stress_test_runner (all combos)      |
|  report_generator (COMPARISON.md)     |
+---------------------------------------+
```

## Module contracts

- **`Retriever.index(documents) -> self`**, then **`retrieve(query, k) -> list[RetrievedDoc]`**.
  Ranking is by descending score with ties broken by `doc_id` for determinism.
- **`Generator.generate(query, contexts, *, seed) -> str`**. Generators may only
  emit content grounded in `contexts`; with empty contexts they return a fixed
  abstention string.
- **`RAGPipeline.run(query) -> PipelineResult`**. Seeds the run, retries
  retrieval up to `max_retries`, and records one trace event per stage.
- **`classify_failure(...) -> FailureMode`**. Deterministic precedence:
  correct → hallucination → irrelevant_retrieval → context_mismatch →
  partial_correctness.
- **Metrics** are pure functions; degenerate inputs return `0.0`.

## Grounding

Faithfulness and hallucination are scored against the gold answer plus the gold
context texts, not against whatever the retriever surfaced. That's what lets a
confident answer drawn from an off-topic passage register as a hallucination,
which is the gap between the extractive baseline and the template generator.

## Determinism

`set_seed` runs per pipeline run. Retrievers and generators are pure functions
of (query, corpus); the only non-deterministic thing is wall-clock latency,
which is reported but kept out of the determinism test.
