"""Render the suite result as markdown + JSON: a summary table, latency, the
failure-mode breakdown, the baseline-vs-improved headline, a recommendation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_KEY = "bm25+extractive"
IMPROVED_KEY = "hybrid+template"


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    return head + sep + body


def _summary_rows(suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for combo, bench in sorted(suite.items()):
        agg = bench["aggregate"]
        rows.append({
            "combo": combo,
            "faithfulness": agg["faithfulness"],
            "hallucination_rate": agg["hallucination_rate"],
            "context_precision@k": agg["context_precision_at_k"],
            "answer_F1": agg["answer_correctness_f1"],
            "p50_ms": agg["latency_p50_ms"],
        })
    return rows


def _recommendation(suite: dict[str, Any]) -> str:
    base = suite.get(BASELINE_KEY)
    impr = suite.get(IMPROVED_KEY)
    if not base or not impr:
        best = min(suite.items(), key=lambda kv: kv[1]["aggregate"]["hallucination_rate"])
        return (f"Recommended configuration: **{best[0]}** "
                f"(lowest hallucination rate = {best[1]['aggregate']['hallucination_rate']}).")
    b_h = base["aggregate"]["hallucination_rate"]
    i_h = impr["aggregate"]["hallucination_rate"]
    reduction = round((b_h - i_h) / b_h * 100.0, 1) if b_h else 0.0
    return (
        f"Recommended configuration: **{BASELINE_KEY}** for low-latency dev/demo "
        f"(fast, simple), and **{IMPROVED_KEY}** for production accuracy "
        f"(hallucination rate {i_h} vs {b_h}, a {reduction}% reduction)."
    )


def build_report(suite: dict[str, Any]) -> dict[str, Any]:
    base = suite.get(BASELINE_KEY, {}).get("aggregate", {})
    impr = suite.get(IMPROVED_KEY, {}).get("aggregate", {})

    def _delta(metric: str, lower_is_better: bool) -> dict[str, Any]:
        b = float(base.get(metric, 0.0))
        i = float(impr.get(metric, 0.0))
        improved = (i < b) if lower_is_better else (i > b)
        if lower_is_better:
            pct = round((b - i) / b * 100.0, 2) if b else 0.0
        else:
            pct = round((i - b) / b * 100.0, 2) if b else 0.0
        return {"baseline": b, "improved": i, "improved_pct": pct, "is_improvement": improved}

    return {
        "baseline_combo": BASELINE_KEY,
        "improved_combo": IMPROVED_KEY,
        "deltas": {
            "hallucination_rate": _delta("hallucination_rate", lower_is_better=True),
            "faithfulness": _delta("faithfulness", lower_is_better=False),
            "answer_correctness_f1": _delta("answer_correctness_f1", lower_is_better=False),
            "context_precision_at_k": _delta("context_precision_at_k", lower_is_better=False),
        },
    }


def render_markdown(suite: dict[str, Any]) -> str:
    comparison = build_report(suite)
    lines: list[str] = [
        "# RAG Comparative Report: Baseline vs Improved",
        "",
        "_Deterministic run (seed-locked). All data synthetic "
        "(fictional Stillwater Industrial ERP corpus)._",
        "",
        "## Summary (all retriever x generator combinations)",
        "",
        _md_table(_summary_rows(suite),
                  ["combo", "faithfulness", "hallucination_rate",
                   "context_precision@k", "answer_F1", "p50_ms"]),
        "",
        "## Latency analysis (ms)",
        "",
        _md_table(
            [{"combo": c,
              "p50": b["aggregate"]["latency_p50_ms"],
              "p95": b["aggregate"]["latency_p95_ms"],
              "p99": b["aggregate"]["latency_p99_ms"]}
             for c, b in sorted(suite.items())],
            ["combo", "p50", "p95", "p99"]),
        "",
        "## Failure-mode breakdown",
        "",
    ]
    for combo, bench in sorted(suite.items()):
        modes = bench["aggregate"]["failure_modes"]
        lines.append(f"**{combo}**: " + ", ".join(f"{m}: {n}" for m, n in modes.items()))
    lines += ["", "## Baseline vs Improved (headline)", ""]
    for metric, d in comparison["deltas"].items():
        arrow = "improved" if d["is_improvement"] else "no improvement"
        lines.append(f"- **{metric}**: baseline {d['baseline']} -> improved "
                     f"{d['improved']} ({d['improved_pct']}% change, {arrow})")
    lines += ["", "## Recommendation", "", _recommendation(suite), ""]
    return "\n".join(lines)


def write_reports(suite: dict[str, Any], eval_dir: str | Path) -> dict[str, Path]:
    d = Path(eval_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_json": d / "stress_results.json",
        "comparison_json": d / "comparison.json",
        "report_md": d / "COMPARISON.md",
    }
    paths["results_json"].write_text(json.dumps(suite, indent=2, sort_keys=True),
                                     encoding="utf-8")
    paths["comparison_json"].write_text(
        json.dumps(build_report(suite), indent=2, sort_keys=True), encoding="utf-8")
    paths["report_md"].write_text(render_markdown(suite), encoding="utf-8")
    return paths
