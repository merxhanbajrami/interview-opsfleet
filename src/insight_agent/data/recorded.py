"""Fixture-replaying executor, used by the eval harness and the tests.

Evaluation has to be deterministic and free.  Running the eval suite against
BigQuery would make results depend on the dataset's current contents, cost
money on every run, and make the suite unusable in CI without cloud
credentials.

Fixtures are recorded from real BigQuery responses (see
``evals/record_fixtures.py``), so what is replayed is genuine warehouse output,
not invented data.  Queries are matched on a normalised form of the SQL, so
formatting differences do not cause a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

from insight_agent.data.executor import (
    ColumnInfo,
    DryRunEstimate,
    QueryResult,
    QuerySyntaxError,
    TableInfo,
)

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def fingerprint(sql: str) -> str:
    """Stable key for a query, insensitive to whitespace and case."""
    # Order matters: collapse whitespace first, then strip trailing semicolons
    # and any space they leave behind, otherwise "SELECT 1" and "select 1 ;"
    # hash differently and every fixture lookup misses.
    normalised = _WS.sub(" ", sql.strip().lower()).rstrip("; ").strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


class RecordedExecutor:
    """Replays recorded query results."""

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir
        self._fixtures: dict[str, dict] = {}
        self._schemas: dict[str, TableInfo] = {}
        self.misses: list[str] = []
        self.load()

    def load(self) -> None:
        if not self.fixtures_dir.is_dir():
            log.warning("no fixtures directory at %s", self.fixtures_dir)
            return
        for path in sorted(self.fixtures_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                log.warning("skipping fixture %s: %s", path.name, exc)
                continue
            for record in data if isinstance(data, list) else [data]:
                key = record.get("fingerprint") or fingerprint(record.get("sql", ""))
                self._fixtures[key] = record

    # --- QueryExecutor protocol -------------------------------------------

    def dry_run(self, sql: str) -> DryRunEstimate:
        record = self._fixtures.get(fingerprint(sql))
        if record is None:
            return DryRunEstimate(0, valid=False, error=_miss_message(sql))
        if record.get("error"):
            return DryRunEstimate(0, valid=False, error=record["error"])
        return DryRunEstimate(int(record.get("bytes_processed", 0)), valid=True)

    def execute(self, sql: str) -> QueryResult:
        key = fingerprint(sql)
        record = self._fixtures.get(key)
        if record is None:
            self.misses.append(sql)
            raise QuerySyntaxError(_miss_message(sql))
        if record.get("error"):
            raise QuerySyntaxError(record["error"])
        return QueryResult(
            rows=pd.DataFrame(record.get("rows", [])),
            sql=sql,
            bytes_processed=int(record.get("bytes_processed", 0)),
            bytes_billed=int(record.get("bytes_billed", 0)),
            duration_ms=float(record.get("duration_ms", 0.0)),
            cache_hit=True,
            job_id=f"recorded-{key}",
        )

    def get_table_schema(self, table_name: str) -> TableInfo:
        """Fall back to the curated catalog, which is the same shape."""
        if table_name in self._schemas:
            return self._schemas[table_name]
        from insight_agent.data.catalog import TABLES

        table = TABLES.get(table_name)
        if table is None:
            raise QuerySyntaxError(f"Unknown table: {table_name}")
        return TableInfo(
            name=table.name,
            description=table.description,
            columns=[
                ColumnInfo(name=c.name, type=c.type, description=c.description)
                for c in table.columns
            ],
        )

    def health_check(self) -> bool:
        return True


def _miss_message(sql: str) -> str:
    return (
        f"No recorded fixture for this query (fingerprint {fingerprint(sql)}). "
        "Record one with `python evals/record_fixtures.py`, or set "
        "INSIGHT_EXECUTOR=bigquery to run against the warehouse."
    )
