"""Reconcile the curated catalog against the live warehouse schema.

Drift has two failure modes and they are not equally bad. A catalogued column
the warehouse lacks breaks queries, loudly. A live column absent from the
catalog is silent, but harmless: the guard is an allowlist, so an
unclassified column is unreachable until someone classifies it.

So a missing column fails the check and an unknown column is only reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from insight_agent.data.catalog import TABLES, Sensitivity
from insight_agent.data.executor import QueryExecutor


@dataclass
class TableDiff:
    """Differences for one table."""

    name: str
    missing_in_warehouse: list[str] = field(default_factory=list)
    unclassified_in_catalog: list[tuple[str, str]] = field(default_factory=list)
    type_mismatches: list[tuple[str, str, str]] = field(default_factory=list)
    error: str = ""

    @property
    def breaks_queries(self) -> bool:
        """Only a missing column actually breaks anything."""
        return bool(self.missing_in_warehouse) or bool(self.error)

    @property
    def clean(self) -> bool:
        return not (
            self.missing_in_warehouse
            or self.unclassified_in_catalog
            or self.type_mismatches
            or self.error
        )


@dataclass
class SchemaReport:
    tables: list[TableDiff] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing would break a query."""
        return not any(t.breaks_queries for t in self.tables)

    @property
    def clean(self) -> bool:
        return all(t.clean for t in self.tables)

    def blocked_column_count(self) -> int:
        """How many live columns are implicitly denied for want of a class."""
        return sum(len(t.unclassified_in_catalog) for t in self.tables)


#: BigQuery reports several names for what the catalog calls one type.
_TYPE_ALIASES: dict[str, set[str]] = {
    "INTEGER": {"INTEGER", "INT64"},
    "FLOAT": {"FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"},
    "STRING": {"STRING"},
    "TIMESTAMP": {"TIMESTAMP", "DATETIME"},
    "GEOGRAPHY": {"GEOGRAPHY"},
    "BOOLEAN": {"BOOLEAN", "BOOL"},
}


def _types_agree(catalog_type: str, live_type: str) -> bool:
    accepted = _TYPE_ALIASES.get(catalog_type.upper(), {catalog_type.upper()})
    return live_type.upper() in accepted


def verify(executor: QueryExecutor) -> SchemaReport:
    """Compare every catalogued table against the warehouse."""
    report = SchemaReport()

    for name, table in TABLES.items():
        diff = TableDiff(name=name)
        try:
            live = executor.get_table_schema(name)
        except Exception as exc:  # noqa: BLE001
            diff.error = str(exc)[:200]
            report.tables.append(diff)
            continue

        live_columns = {c.name.lower(): c for c in live.columns}
        catalog_columns = {c.name.lower(): c for c in table.columns}

        for lowered, column in catalog_columns.items():
            live_column = live_columns.get(lowered)
            if live_column is None:
                diff.missing_in_warehouse.append(column.name)
            elif not _types_agree(column.type, live_column.type):
                diff.type_mismatches.append(
                    (column.name, column.type, live_column.type)
                )

        for lowered, live_column in live_columns.items():
            if lowered not in catalog_columns:
                diff.unclassified_in_catalog.append((live_column.name, live_column.type))

        report.tables.append(diff)

    return report


def summarise(report: SchemaReport) -> str:
    """Plain-text summary, for CI logs."""
    lines: list[str] = []
    for diff in report.tables:
        if diff.error:
            lines.append(f"{diff.name}: ERROR {diff.error}")
            continue
        if diff.clean:
            lines.append(f"{diff.name}: ok")
            continue
        if diff.missing_in_warehouse:
            lines.append(
                f"{diff.name}: BREAKS QUERIES, catalogued but absent: "
                + ", ".join(diff.missing_in_warehouse)
            )
        for column, expected, actual in diff.type_mismatches:
            lines.append(f"{diff.name}.{column}: type {expected} but warehouse says {actual}")
        if diff.unclassified_in_catalog:
            lines.append(
                f"{diff.name}: {len(diff.unclassified_in_catalog)} live column(s) "
                "unclassified, therefore blocked: "
                + ", ".join(n for n, _ in diff.unclassified_in_catalog)
            )
    return "\n".join(lines)


def classification_summary() -> dict[str, int]:
    """How the catalog currently classifies its columns."""
    counts = {level.value: 0 for level in Sensitivity}
    for table in TABLES.values():
        for column in table.columns:
            counts[column.sensitivity.value] += 1
    return counts
