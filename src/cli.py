"""CLI: run / eval-suite / report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from shared.determinism import DEFAULT_SEED, set_seed

from .abstraction.bm25_retriever import BM25Retriever
from .abstraction.embedding_retriever import EmbeddingRetriever
from .abstraction.hybrid_retriever import HybridRetriever
from .abstraction.mock_generator import ExtractiveGenerator
from .abstraction.template_generator import TemplateGenerator
from .core_logic.rag_pipeline import RAGPipeline
from .evaluation.report_generator import render_markdown, write_reports
from .evaluation.stress_test_runner import run_stress_suite
from .input_layer.dataset_loader import load_dataset, load_queries

_ROOT = Path(__file__).resolve().parent.parent

_RETRIEVERS = {
    "bm25": lambda cfg, seed: BM25Retriever(
        k1=cfg["bm25_k1"], b=cfg["bm25_b"], seed=seed),
    "embedding": lambda cfg, seed: EmbeddingRetriever(seed=seed),
    "hybrid": lambda cfg, seed: HybridRetriever(
        alpha=cfg["hybrid_alpha"], k1=cfg["bm25_k1"], b=cfg["bm25_b"], seed=seed),
}
_GENERATORS = {
    "extractive": lambda: ExtractiveGenerator(),
    "template": lambda: TemplateGenerator(),
}


def _load_config(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"seed": DEFAULT_SEED, "retrieval": {"k": 5, "hybrid_alpha": 0.5,
            "bm25_k1": 1.5, "bm25_b": 0.75}}


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    rc = cfg.get("retrieval", {})
    set_seed(seed)
    docs = load_dataset()
    retriever = _RETRIEVERS[args.retriever](
        {"hybrid_alpha": rc.get("hybrid_alpha", 0.5),
         "bm25_k1": rc.get("bm25_k1", 1.5), "bm25_b": rc.get("bm25_b", 0.75)}, seed)
    retriever.index(docs)
    generator = _GENERATORS[args.generator]()
    pipeline = RAGPipeline(retriever, generator, seed=seed, k=rc.get("k", 5))
    result = pipeline.run(args.query)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_eval_suite(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    rc = cfg.get("retrieval", {})
    set_seed(seed)
    docs = load_dataset()
    queries = load_queries()
    suite = run_stress_suite(
        docs, queries, seed=seed, k=rc.get("k", 5),
        hybrid_alpha=rc.get("hybrid_alpha", 0.5),
        k1=rc.get("bm25_k1", 1.5), b=rc.get("bm25_b", 0.75))
    summary = {combo: bench["aggregate"] for combo, bench in suite.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    seed = int(cfg.get("seed", DEFAULT_SEED))
    rc = cfg.get("retrieval", {})
    set_seed(seed)
    docs = load_dataset()
    queries = load_queries()
    suite = run_stress_suite(
        docs, queries, seed=seed, k=rc.get("k", 5),
        hybrid_alpha=rc.get("hybrid_alpha", 0.5),
        k1=rc.get("bm25_k1", 1.5), b=rc.get("bm25_b", 0.75))
    out_dir = Path(args.out) if args.out else _ROOT / "eval"
    paths = write_reports(suite, out_dir)
    print(render_markdown(suite))
    print("\nwrote:", ", ".join(str(p) for p in paths.values()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="p2-rag", description="RAG Evaluation & Failure Mode Lab")
    ap.add_argument("--config", default=str(_ROOT / "config.json"))
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one query")
    p_run.add_argument("query")
    p_run.add_argument("--retriever", choices=list(_RETRIEVERS), default="hybrid")
    p_run.add_argument("--generator", choices=list(_GENERATORS), default="template")
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser("eval-suite", help="evaluate all combinations")
    p_eval.set_defaults(func=_cmd_eval_suite)

    p_rep = sub.add_parser("report", help="write comparative report to eval/")
    p_rep.add_argument("--out", default=None)
    p_rep.set_defaults(func=_cmd_report)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
