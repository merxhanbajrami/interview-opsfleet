"""The data-plane contract.

Every component above this line talks to a ``QueryExecutor``, never to
BigQuery directly.  That buys three things:

* the eval harness replays recorded fixtures, so evaluation is deterministic
  and costs nothing;
* a second warehouse (Snowflake, Postgres, DuckDB) is a new implementation
  rather than a change to the agent;
* failure modes are normalised, so the graph reasons about
  ``QuerySyntaxError`` instead of a vendor-specific exception hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

# --- Normalised failure taxonomy -------------------------------------------
# The self-correction node branches on these.  The distinction matters:
# a syntax error is worth repairing, a permission error never is.


class QueryError(Exception):
    """Base class for all data-plane failures."""

    retryable: bool = False
    repairable: bool = False


class QuerySyntaxError(QueryError):
    """Malformed SQL, unknown column, type mismatch.

    Repairable: the model can fix it if shown the error text.
    """

    repairable = True


class QueryCostError(QueryError):
    """The query would scan more data than policy allows.

    Repairable: usually fixed by adding a filter or narrowing the date range.
    """

    repairable = True


class QueryPermissionError(QueryError):
    """Authentication or authorisation failure.

    Neither retryable nor repairable — retrying wastes time and money.
    """


class QueryTransientError(QueryError):
    """Timeout, rate limit, or backend unavailability.

    Retryable with backoff, but not repairable: the SQL itself is fine.
    """

    retryable = True


@dataclass(slots=True)
class QueryResult:
    """Outcome of a successful query, plus the metadata observability needs."""

    rows: pd.DataFrame
    sql: str
    bytes_processed: int = 0
    bytes_billed: int = 0
    duration_ms: float = 0.0
    cache_hit: bool = False
    truncated: bool = False
    job_id: str | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """Whether the query found nothing.

        Zero rows is the obvious case. The one that matters in practice is an
        aggregate over no matching rows: ``SELECT SUM(sale_price) FROM
        order_items WHERE status = 'NoSuchStatus'`` returns exactly one row
        containing NULL, not an empty result set. Treating that as data hands
        the model a null to explain, and a model asked to explain a null will
        usually invent a reason. It means "nothing matched", so it is reported
        that way.
        """
        if self.rows.empty:
            return True
        if len(self.rows) != 1:
            return False

        # A single row of aggregates needs care, because SQL reports "no rows
        # matched" inconsistently across aggregate functions: SUM returns
        # NULL, COUNT returns 0. So `SELECT SUM(x), COUNT(*) WHERE <no match>`
        # yields {x: NULL, n: 0} -- neither empty nor all-null.
        #
        # The discriminator is the NULL. SUM over an empty set is NULL; SUM
        # over rows that happen to total zero is 0.0, never NULL. So a row
        # that contains at least one NULL and no non-zero value means nothing
        # matched, while a row of plain zeros is a real answer.
        try:
            row = self.rows.iloc[0]
            if not bool(row.isna().any()):
                return False
            for value in row:
                # BigQuery hands back pandas nullable dtypes, so a missing
                # value is pd.NA rather than float('nan'). The usual
                # `value != value` NaN check returns pd.NA for those, which
                # raises when used in a boolean context. pd.isna handles
                # None, nan and pd.NA alike.
                if pd.isna(value):
                    continue
                try:
                    if float(value) != 0.0:
                        return False
                except (TypeError, ValueError):
                    return False  # a non-numeric value is real content
            return True
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def estimated_cost_usd(self) -> float:
        """BigQuery on-demand pricing, $6.25 per TiB scanned."""
        return (self.bytes_billed / 1_099_511_627_776) * 6.25

    def to_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """Rows as plain dicts, capped, for handing to the model."""
        return self.rows.head(limit).to_dict(orient="records")


@dataclass(slots=True)
class DryRunEstimate:
    """Result of a cost check made *before* any data is read."""

    bytes_processed: int
    valid: bool
    error: str | None = None

    @property
    def gigabytes(self) -> float:
        return self.bytes_processed / 1_000_000_000


@dataclass(slots=True)
class ColumnInfo:
    name: str
    type: str
    mode: str = "NULLABLE"
    description: str = ""


@dataclass(slots=True)
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int | None = None
    description: str = ""


@runtime_checkable
class QueryExecutor(Protocol):
    """What the agent requires of a data warehouse."""

    def dry_run(self, sql: str) -> DryRunEstimate:
        """Validate syntax and estimate scan size without reading data."""
        ...

    def execute(self, sql: str) -> QueryResult:
        """Run the query. Raises a ``QueryError`` subclass on failure."""
        ...

    def get_table_schema(self, table_name: str) -> TableInfo:
        """Describe one table."""
        ...

    def health_check(self) -> bool:
        """Cheap liveness probe, used by the circuit breaker."""
        ...
