# Architecture

Five packages under `src/`, each with a narrow contract. The evaluation package
composes the rest. Data flows top to bottom.

## Data flow

```mermaid
flowchart TD
    corpus["erp_corpus.jsonl (corpus)"] --> input
    queries["erp_queries.jsonl (queries + gold answers)"] --> input

    subgraph input["input_layer"]
        s1["schema (Document/Query/RetrievedDoc)"]
        s2["validation (length/empty/enum constraints)"]
        s3["stress_taxonomy (adversarial/ambiguous/long-ctx)"]
        s4["dataset_loader (deterministic, id-sorted)"]
    end

    input -->|"Document[]"| abstraction
    input -->|"Query[]"| core

    subgraph abstraction["abstraction"]
        r["Retriever ABC\nBM25Retriever (pure Python Okapi)\nEmbeddingRetriever (TF-IDF + cosine)\nHybridRetriever (weighted fusion)"]
        g["Generator ABC\nExtractiveGenerator (baseline)\nTemplateGenerator (grounding-aware)"]
    end

    abstraction --> core["core_logic: RAGPipeline\nset_seed -> retrieve (retry) -> generate -> trace each stage"]

    core -->|"PipelineResult"| evaluation
    core -->|"trace events"| logging

    subgraph evaluation["evaluation"]
        e1["failure_taxonomy (4 modes)"]
        e2["metrics (faithfulness, ctxP@k, hallucination, F1)"]
        e3["benchmark (per-query + aggregate)"]
        e4["stress_test_runner (all combos)"]
        e5["report_generator (COMPARISON.md)"]
    end

    subgraph logging["logging_layer"]
        l1["event_schema"]
        l2["event_log (mem+JSONL)"]
        l3["debug_hooks (HTML)"]
    end
```

## Module contracts

- **`Retriever.index(documents) -> self`**, then **`retrieve(query, k) -> list[RetrievedDoc]`**.
  Ranking is by descending score with ties broken by `doc_id` for determinism.
- **`Generator.generate(query, contexts, *, seed) -> str`**. Generators may only
  emit content grounded in `contexts`. With empty contexts they return a fixed
  abstention string.
- **`RAGPipeline.run(query) -> PipelineResult`**. Seeds the run, retries
  retrieval up to `max_retries`, and records one trace event per stage.
- **`classify_failure(...) -> FailureMode`**. Deterministic precedence:
  correct → hallucination → irrelevant_retrieval → context_mismatch →
  partial_correctness.
- **Metrics** are pure functions. Degenerate inputs return `0.0`.

## Grounding

Faithfulness and hallucination are scored against the gold answer plus the gold
context texts, not against whatever the retriever surfaced. That's what lets a
confident answer drawn from an off-topic passage register as a hallucination,
which is the gap between the extractive baseline and the template generator.

## Determinism

`set_seed` runs per pipeline run. Retrievers and generators are pure functions
of (query, corpus). The only non-deterministic thing is wall-clock latency,
which is reported but kept out of the determinism test.
