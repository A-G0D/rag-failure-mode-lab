"""Read the corpus + query set from JSONL, sorted by id and validated."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Document, Query
from .validation import validate_document, validate_query

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "datasets"
DEFAULT_CORPUS_PATH = _DEFAULT_DIR / "erp_corpus.jsonl"
DEFAULT_QUERIES_PATH = _DEFAULT_DIR / "erp_queries.jsonl"


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(path: str | Path | None = None) -> list[Document]:
    rows = _read_jsonl(path or DEFAULT_CORPUS_PATH)
    docs = [validate_document(Document.from_dict(r)) for r in rows]
    return sorted(docs, key=lambda d: d.doc_id)


def load_queries(path: str | Path | None = None) -> list[Query]:
    rows = _read_jsonl(path or DEFAULT_QUERIES_PATH)
    queries = [validate_query(Query.from_dict(r)) for r in rows]
    return sorted(queries, key=lambda q: q.query_id)


def documents_to_jsonl(docs: Iterable[Document], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.as_dict(), sort_keys=True) + "\n")
