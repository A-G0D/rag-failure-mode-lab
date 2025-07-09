from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_src_packages_present():
    for pkg in ("input_layer", "core_logic", "abstraction", "evaluation", "logging_layer"):
        assert (ROOT / "src" / pkg).is_dir(), f"missing package: {pkg}"


def test_shipped_files_present():
    for rel in ("README.md", "config.json", "shared/obs.py", "shared/determinism.py",
                "docs/datasets/erp_corpus.jsonl", "docs/datasets/erp_queries.jsonl"):
        assert (ROOT / rel).exists(), f"missing: {rel}"


def test_readme_covers_the_basics():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "architecture" in text or "layout" in text
    assert "metric" in text
