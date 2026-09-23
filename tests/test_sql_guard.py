"""The SQL guard is the enforcement point for requirement 2.

These tests are written as an adversary: each one is an attempt to get
personal data or out-of-scope data past the guard.
"""

from __future__ import annotations

import pytest

from insight_agent.security.identity import get_principal
from insight_agent.security.sql_guard import build_guard

DATASET = "bigquery-public-data.thelook_ecommerce"


@pytest.fixture
def guard():
    return build_guard(DATASET)


@pytest.fixture
def ceo():
    return get_principal("ceo")


@pytest.fixture
def vp_women():
    return get_principal("vp_women")


# --- PII must not be reachable ---------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT email FROM users",
        "SELECT first_name, last_name FROM users",
        "SELECT u.email FROM users u",
        "SELECT id FROM users WHERE email = 'target@example.com'",
        "SELECT id FROM users ORDER BY last_name",
        "SELECT latitude, longitude FROM users",
        "SELECT street_address FROM users",
        "SELECT postal_code FROM users",
        "SELECT order_id FROM orders WHERE user_id IN "
        "(SELECT id FROM users WHERE email LIKE '%@vip.com')",
        "SELECT COUNT(*) c FROM users GROUP BY email",
        "SELECT CONCAT(first_name, ' ', last_name) AS n FROM users",
    ],
)
def test_blocked_columns_are_rejected(guard, ceo, sql):
    result = guard.check(sql, ceo)
    assert not result.ok, f"PII query was allowed: {sql}"


def test_select_star_rejected_because_it_expands_to_pii(guard, ceo):
    assert not guard.check("SELECT * FROM users", ceo).ok
    assert not guard.check("SELECT u.* FROM users u", ceo).ok


# --- The connection is read-only -------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE users",
        "DELETE FROM orders WHERE 1=1",
        "INSERT INTO orders (order_id) VALUES (1)",
        "UPDATE products SET cost = 0",
        "CREATE TABLE evil AS SELECT 1",
        "TRUNCATE TABLE orders",
    ],
)
def test_write_statements_are_rejected(guard, ceo, sql):
    assert not guard.check(sql, ceo).ok


def test_statement_chaining_is_rejected(guard, ceo):
    result = guard.check("SELECT 1 FROM orders; DROP TABLE users", ceo)
    assert not result.ok
    assert "one statement" in result.reason()


# --- Only catalogued tables ------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM secret_salaries",
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES",
        "SELECT id FROM `other-project.other_dataset.customers`",
    ],
)
def test_unknown_tables_are_rejected(guard, ceo, sql):
    assert not guard.check(sql, ceo).ok


# --- Legitimate analysis still works ---------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DATE_TRUNC(DATE(created_at), MONTH) m, SUM(sale_price) rev "
        "FROM order_items WHERE status = 'Complete' GROUP BY m ORDER BY m",
        "SELECT user_id, SUM(sale_price) spend FROM order_items "
        "GROUP BY user_id ORDER BY spend DESC LIMIT 10",
        "SELECT u.state, AVG(oi.sale_price) avg_spend FROM order_items oi "
        "JOIN users u ON oi.user_id = u.id GROUP BY u.state",
        "WITH rev AS (SELECT product_id, SUM(sale_price) s FROM order_items GROUP BY product_id) "
        "SELECT p.name, rev.s FROM rev JOIN products p ON p.id = rev.product_id",
    ],
)
def test_analytical_queries_pass(guard, ceo, sql):
    result = guard.check(sql, ceo)
    assert result.ok, result.reason()
    assert DATASET in result.sql


# --- Per-user product scope ------------------------------------------------


def test_scope_applied_when_products_referenced(guard, vp_women):
    result = guard.check(
        "SELECT p.name, SUM(oi.sale_price) rev FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id GROUP BY p.name",
        vp_women,
    )
    assert result.ok and result.scope_applied
    assert "'Women'" in result.sql


def test_scope_survives_a_query_that_never_mentions_products(guard, vp_women):
    """The interesting case.

    A scoped user can ask for total revenue without naming a product. If scope
    were only applied to the products table, this query would return the
    company-wide figure. It is constrained at the revenue grain instead.
    """
    result = guard.check("SELECT SUM(sale_price) total FROM order_items", vp_women)
    assert result.ok and result.scope_applied
    assert "products" in result.sql and "'Women'" in result.sql


def test_unrestricted_principal_gets_no_scope_filter(guard, ceo):
    result = guard.check("SELECT SUM(sale_price) total FROM order_items", ceo)
    assert result.ok and not result.scope_applied
    assert "department IN" not in result.sql


def test_scope_values_are_escaped():
    from insight_agent.security.identity import Principal

    hostile = Principal(
        user_id="x", display_name="X", role="r",
        departments=frozenset({"Women' OR 1=1--"}),
    )
    assert "''" in hostile.scope_predicate()


# --- Result bounding -------------------------------------------------------


def test_missing_limit_is_added(guard, ceo):
    result = guard.check("SELECT id FROM products", ceo)
    assert result.ok and result.limit_applied == 1000
    assert "LIMIT" in result.sql.upper()


def test_oversized_limit_is_clamped(guard, ceo):
    result = guard.check("SELECT id FROM products LIMIT 999999", ceo)
    assert result.ok and result.limit_applied == 10_000


def test_unparseable_sql_is_reported_not_raised(guard, ceo):
    result = guard.check("SELECT FROM WHERE ORDER", ceo)
    assert not result.ok and result.violations
