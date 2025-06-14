"""Evaluation layer: failure taxonomy, metrics, benchmark, stress tests, reports."""
from .failure_taxonomy import FailureMode, classify_failure
from .metrics import (
    faithfulness, context_precision_at_k, hallucination_rate, answer_correctness_f1,
)
from .benchmark import run_benchmark
from .stress_test_runner import run_stress_suite
from .report_generator import build_report

__all__ = [
    "FailureMode", "classify_failure",
    "faithfulness", "context_precision_at_k", "hallucination_rate", "answer_correctness_f1",
    "run_benchmark", "run_stress_suite", "build_report",
]
