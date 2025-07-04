"""CLI smoke tests for the three subcommands."""
import json

from src.cli import main


def test_cli_run(capsys):
    rc = main(["run", "What triggers a backorder?", "--retriever", "hybrid",
               "--generator", "template"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "generated_answer" in out


def test_cli_eval_suite(capsys):
    rc = main(["eval-suite"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert "bm25+extractive" in summary and "hybrid+template" in summary


def test_cli_report_writes_artifacts(tmp_path, capsys):
    rc = main(["report", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "COMPARISON.md").exists()
    assert (tmp_path / "stress_results.json").exists()
    assert (tmp_path / "comparison.json").exists()
