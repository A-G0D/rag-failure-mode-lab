"""Stress-suite + comparative-report tests."""
from src.evaluation.report_generator import (
    BASELINE_KEY, IMPROVED_KEY, build_report, render_markdown, write_reports,
)
from src.evaluation.stress_test_runner import run_stress_suite
from src.input_layer.dataset_loader import load_dataset, load_queries


def _suite():
    return run_stress_suite(load_dataset(), load_queries(), k=5)


def test_suite_runs_all_combos():
    suite = _suite()
    # 3 retrievers x 2 generators = 6 combinations.
    assert len(suite) == 6
    assert BASELINE_KEY in suite and IMPROVED_KEY in suite


def test_suite_has_per_stress_type_slices():
    suite = _suite()
    by_stress = suite[BASELINE_KEY]["by_stress_type"]
    for st in ("adversarial", "ambiguous", "long_context_overload"):
        assert st in by_stress


def test_improved_beats_baseline_on_hallucination():
    suite = _suite()
    report = build_report(suite)
    delta = report["deltas"]["hallucination_rate"]
    assert delta["is_improvement"], (
        f"expected hallucination improvement: {delta}")
    assert delta["improved_pct"] > 0.0


def test_render_markdown_has_tables_and_recommendation():
    md = render_markdown(_suite())
    assert "# RAG Comparative Report" in md
    assert "Recommendation" in md
    assert "| combo |" in md


def test_write_reports_creates_files(tmp_path):
    paths = write_reports(_suite(), tmp_path)
    for p in paths.values():
        assert p.exists()
