"""Standalone runner for the full report (no CLI args):

    python scripts/run_comparative_report.py

Writes eval/COMPARISON.md, eval/stress_results.json, eval/comparison.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.determinism import DEFAULT_SEED, set_seed  # noqa: E402
from src.evaluation.report_generator import build_report, write_reports  # noqa: E402
from src.evaluation.stress_test_runner import run_stress_suite  # noqa: E402
from src.input_layer.dataset_loader import load_dataset, load_queries  # noqa: E402


def main() -> int:
    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    seed = int(cfg.get("seed", DEFAULT_SEED))
    rc = cfg.get("retrieval", {})
    set_seed(seed)

    docs = load_dataset()
    queries = load_queries()
    suite = run_stress_suite(
        docs, queries, seed=seed, k=rc.get("k", 5),
        hybrid_alpha=rc.get("hybrid_alpha", 0.5),
        k1=rc.get("bm25_k1", 1.5), b=rc.get("bm25_b", 0.75))
    paths = write_reports(suite, ROOT / "eval")
    comparison = build_report(suite)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("wrote:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
