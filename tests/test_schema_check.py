"""Catalog-vs-warehouse reconciliation.

The important assertion here is the asymmetry: a column the catalog invented
breaks queries and must fail the check, while a column the warehouse grew that
nobody has classified is safe, because it is blocked by default, and must
reported.
"""

from __future__ import annotations

from insight_agent.data.executor import ColumnInfo, TableInfo
from insight_agent.data.schema_check import classification_summary, summarise, verify


class FakeWarehouse:
    """Returns whatever schema a test hands it."""

    def __init__(self, tables: dict[str, list[tuple[str, str]]]) -> None:
        self.tables = tables

    def get_table_schema(self, table_name: str) -> TableInfo:
        if table_name not in self.tables:
            raise RuntimeError(f"no such table: {table_name}")
        return TableInfo(
            name=table_name,
            columns=[ColumnInfo(name=n, type=t) for n, t in self.tables[table_name]],
        )


def _catalog_shaped() -> dict[str, list[tuple[str, str]]]:
    """A warehouse that agrees with the catalog exactly."""
    from insight_agent.data.catalog import TABLES

    return {
        name: [(c.name, c.type) for c in table.columns]
        for name, table in TABLES.items()
    }


def test_an_exact_match_is_clean():
    report = verify(FakeWarehouse(_catalog_shaped()))
    assert report.clean and report.ok


def test_a_column_the_catalog_invented_breaks_queries():
    tables = _catalog_shaped()
    tables["users"] = [c for c in tables["users"] if c[0] != "state"]
    report = verify(FakeWarehouse(tables))

    assert not report.ok, "a missing column should fail the check"
    users = next(t for t in report.tables if t.name == "users")
    assert "state" in users.missing_in_warehouse
    assert "BREAKS QUERIES" in summarise(report)


def test_an_unclassified_live_column_is_reported_but_not_an_error():
    """The fail-closed property: unknown columns are blocked, not leaked."""
    tables = _catalog_shaped()
    tables["users"].append(("phone_number", "STRING"))
    report = verify(FakeWarehouse(tables))

    assert report.ok, "an unclassified column must not be treated as breakage"
    assert not report.clean
    assert report.blocked_column_count() == 1
    users = next(t for t in report.tables if t.name == "users")
    assert ("phone_number", "STRING") in users.unclassified_in_catalog


def test_an_unclassified_column_cannot_be_queried():
    """Proves the claim rather than asserting it.

    A column absent from the catalog is rejected by the guard, so the window
    between a schema change and its classification is safe.
    """
    from insight_agent.security.identity import get_principal
    from insight_agent.security.sql_guard import build_guard

    guard = build_guard("bigquery-public-data.thelook_ecommerce")
    result = guard.check("SELECT phone_number FROM users", get_principal("ceo"))
    assert not result.ok


def test_type_aliases_are_tolerated():
    """BigQuery says INT64 where the catalog says INTEGER. Not a mismatch."""
    tables = _catalog_shaped()
    tables["users"] = [
        ("id", "INT64") if n == "id" else (n, t) for n, t in tables["users"]
    ]
    report = verify(FakeWarehouse(tables))
    users = next(t for t in report.tables if t.name == "users")
    assert not users.type_mismatches


def test_a_genuine_type_change_is_reported():
    tables = _catalog_shaped()
    tables["order_items"] = [
        (n, "STRING") if n == "sale_price" else (n, t)
        for n, t in tables["order_items"]
    ]
    report = verify(FakeWarehouse(tables))
    items = next(t for t in report.tables if t.name == "order_items")
    assert ("sale_price", "FLOAT", "STRING") in items.type_mismatches


def test_an_unreachable_table_is_an_error_not_a_crash():
    report = verify(FakeWarehouse({}))
    assert not report.ok
    assert all(t.error for t in report.tables)


def test_every_catalogued_column_has_a_classification():
    counts = classification_summary()
    from insight_agent.data.catalog import TABLES

    total = sum(len(t.columns) for t in TABLES.values())
    assert sum(counts.values()) == total
    assert counts["blocked"] >= 8, "direct identifiers must be blocked"
