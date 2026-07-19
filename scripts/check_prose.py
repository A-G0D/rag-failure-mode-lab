"""Scan tracked markdown/comments for AI-writing tells and leaked artifacts.

Checks every .md file and every .py file's comments/docstrings for:
  - em-dashes (the single biggest AI-writing tell)
  - leaked tool-call artifacts (</invoke>, <function>, <parameter>, tool_use, etc.)
  - employer/internal-tool references that don't belong in a public repo

Stdlib only, no dependencies. Exits non-zero and prints file:line for every hit.

    python scripts/check_prose.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EM_DASH = "—"

ARTIFACT_MARKERS = [
    "</invoke>", "<function>", "<parameter>", "</content>",
    "tool_use", "antml:invoke", "antml:parameter",
]

# Employer / internal-tool references that shouldn't leak into a public repo.
BANNED_TERMS = [
    "gss", "global shop", "gssmail", "fact graph", "brain vault", "livefire",
]

CHECKED_SUFFIXES = {".md", ".py"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}


_SELF = Path(__file__).resolve()


def _iter_files():
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if path.resolve() == _SELF:
            continue  # this file legitimately contains the patterns it looks for
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in CHECKED_SUFFIXES:
            yield path


def _check_file(path: Path) -> list[str]:
    problems = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return problems

    for lineno, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            problems.append(f"{path}:{lineno}: em-dash found")

        lowered = line.lower()
        for marker in ARTIFACT_MARKERS:
            if marker.lower() in lowered:
                problems.append(f"{path}:{lineno}: leaked tool-call artifact ({marker!r})")

        for term in BANNED_TERMS:
            if re.search(r"\b" + re.escape(term) + r"\b", lowered):
                problems.append(f"{path}:{lineno}: banned term ({term!r})")

    return problems


def main() -> int:
    all_problems = []
    for path in sorted(_iter_files()):
        all_problems.extend(_check_file(path))

    if all_problems:
        print("check_prose: found issues:\n")
        for problem in all_problems:
            print(f"  {problem}")
        print(f"\n{len(all_problems)} issue(s) found.")
        return 1

    print("check_prose: clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
