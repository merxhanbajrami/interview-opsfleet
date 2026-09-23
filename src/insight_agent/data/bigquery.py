"""BigQuery implementation of QueryExecutor.

Two things distinguish it from a thin SDK wrapper. Every query is dry-run
first, which is free, validates syntax and returns the exact byte count, so
queries over budget are refused before they cost anything. And failures are
classified into the taxonomy in executor.py rather than propagated, because
the graph needs to know whether a failure is worth repairing, retrying, or
neither.
"""

from __future__ import annotations

import logging
import re
import time

from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exc
from google.cloud import bigquery

from insight_agent.config import Settings, get_settings
from insight_agent.data.executor import (
    ColumnInfo,
    DryRunEstimate,
    QueryCostError,
    QueryPermissionError,
    QueryResult,
    QuerySyntaxError,
    QueryTransientError,
    TableInfo,
)

log = logging.getLogger(__name__)

# BigQuery reports an over-budget query as a plain 400.  Only the message
# distinguishes it from a syntax error, and the two need different handling.
_BYTES_BILLED_RE = re.compile(r"exceed(s|ed)?\s+limit\s+for\s+bytes\s+billed", re.I)


class BigQueryExecutor:
    """Read-only BigQuery access with cost control and error classification."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dataset_id = self.settings.bq_dataset
        try:
            self._client = bigquery.Client(
                project=self.settings.gcp_project or None,
                location=self.settings.bq_location,
            )
        except auth_exc.DefaultCredentialsError as exc:
            raise QueryPermissionError(
                "No Google Cloud credentials found. Run `gcloud auth application-default "
                "login` and set GOOGLE_CLOUD_PROJECT in .env."
            ) from exc
        log.info("BigQuery executor ready (project=%s)", self._client.project)

    # --- Cost gate --------------------------------------------------------

    def dry_run(self, sql: str) -> DryRunEstimate:
        """Validate and price a query without reading a byte.

        Dry runs are free and synchronous.  They catch syntax errors, unknown
        columns and unknown tables, which means most bad SQL is rejected
        before it can cost anything.
        """
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self._client.query(sql, job_config=config)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
            return DryRunEstimate(bytes_processed=0, valid=False, error=_message(exc))
        return DryRunEstimate(bytes_processed=job.total_bytes_processed or 0, valid=True)

    # --- Execution --------------------------------------------------------

    def execute(self, sql: str) -> QueryResult:
        """Run a query that has already passed the guard and the cost gate."""
        estimate = self.dry_run(sql)
        if not estimate.valid:
            raise _classify(estimate.error or "unknown dry-run failure")
        if estimate.bytes_processed > self.settings.bq_max_bytes_billed:
            raise QueryCostError(
                f"Query would scan {estimate.gigabytes:.2f} GB, over the "
                f"{self.settings.bq_max_bytes_billed / 1e9:.2f} GB limit. "
                "Narrow the date range or add a filter."
            )

        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.settings.bq_max_bytes_billed,
            use_query_cache=True,
            labels={"app": "insight-agent"},
        )
        started = time.perf_counter()
        try:
            job = self._client.query(sql, job_config=config)
            frame = job.result(timeout=self.settings.bq_timeout_seconds).to_dataframe()
        except Exception as exc:  # noqa: BLE001 - normalised into the taxonomy
            raise _classify(_message(exc)) from exc

        duration_ms = (time.perf_counter() - started) * 1000
        truncated = len(frame) > self.settings.max_row_limit
        if truncated:
            frame = frame.head(self.settings.max_row_limit)

        return QueryResult(
            rows=frame,
            sql=sql,
            bytes_processed=job.total_bytes_processed or 0,
            bytes_billed=job.total_bytes_billed or 0,
            duration_ms=duration_ms,
            cache_hit=bool(job.cache_hit),
            truncated=truncated,
            job_id=job.job_id,
        )

    # --- Introspection ----------------------------------------------------

    def get_table_schema(self, table_name: str) -> TableInfo:
        try:
            table = self._client.get_table(f"{self.dataset_id}.{table_name}")
        except Exception as exc:  # noqa: BLE001
            raise _classify(_message(exc)) from exc
        return TableInfo(
            name=table_name,
            columns=[
                ColumnInfo(
                    name=f.name,
                    type=f.field_type,
                    mode=f.mode or "NULLABLE",
                    description=f.description or "",
                )
                for f in table.schema
            ],
            row_count=table.num_rows,
            description=table.description or "",
        )

    def health_check(self) -> bool:
        """Cheapest possible round trip, for the circuit breaker."""
        try:
            estimate = self.dry_run("SELECT 1")
            return estimate.valid
        except Exception:  # noqa: BLE001 - a health check never raises
            return False


# --- Error classification ---------------------------------------------------


def _message(exc: Exception) -> str:
    """Best available human-readable text for an exception."""
    for attr in ("message", "errors"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and first.get("message"):
                return str(first["message"])
    return str(exc)


def _classify(message: str):
    """Map a raw failure to the taxonomy the graph reasons about."""
    lowered = message.lower()

    if _BYTES_BILLED_RE.search(message) or "bytes billed" in lowered:
        return QueryCostError(message)
    if any(k in lowered for k in ("permission", "access denied", "forbidden", "credential")):
        return QueryPermissionError(message)
    if any(k in lowered for k in ("rate limit", "quota exceeded", "too many requests")):
        return QueryTransientError(message)
    if any(k in lowered for k in ("deadline", "timeout", "unavailable", "internal error")):
        return QueryTransientError(message)
    # Syntax, unknown column, unknown table and type mismatch are all things
    # the model has a real chance of repairing.
    return QuerySyntaxError(message)


def classify_exception(exc: Exception):
    """Public entry point used by tests and by the retry wrapper."""
    if isinstance(exc, (gexc.Forbidden, gexc.Unauthorized)):
        return QueryPermissionError(_message(exc))
    if isinstance(exc, (gexc.TooManyRequests, gexc.ServiceUnavailable, gexc.InternalServerError)):
        return QueryTransientError(_message(exc))
    if isinstance(exc, gexc.DeadlineExceeded):
        return QueryTransientError(_message(exc))
    return _classify(_message(exc))
