"""Same seed should give the same suite output across runs."""
import json

from src.evaluation.stress_test_runner import run_stress_suite
from src.input_layer.dataset_loader import load_dataset, load_queries


def _signature():
    suite = run_stress_suite(load_dataset(), load_queries(), k=5)
    # latency is wall-clock, so drop it; everything else must repeat
    sig = {}
    for combo, bench in suite.items():
        sig[combo] = {k: v for k, v in bench["aggregate"].items()
                      if not k.startswith("latency_")}
    return json.dumps(sig, sort_keys=True)


def test_suite_is_reproducible():
    a, b, c = (_signature() for _ in range(3))
    assert a == b == c


def test_answers_dont_drift():
    docs, queries = load_dataset(), load_queries()
    answers = [
        tuple(r["answer"] for r in run_stress_suite(docs, queries, k=5)
              ["hybrid+template"]["per_query"])
        for _ in range(3)
    ]
    assert answers[0] == answers[1] == answers[2]
