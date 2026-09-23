#!/usr/bin/env python
"""Record real BigQuery responses for offline replay.

Fixtures are what make the eval suite deterministic and free, but invented
fixtures would only test the agent against my guesses about the data.  So they
are recorded from genuine queries against the public dataset once, and replayed
thereafter.

Run this after configuring BigQuery:

    python evals/record_fixtures.py

Queries are passed through the SQL guard first, so what gets recorded is
keyed on exactly the form the agent will execute — including the row limit and
any scope rewrite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insight_agent.config import get_settings  # noqa: E402
from insight_agent.data.bigquery import BigQueryExecutor  # noqa: E402
from insight_agent.data.recorded import fingerprint  # noqa: E402
from insight_agent.security.identity import get_principal  # noqa: E402
from insight_agent.security.sql_guard import build_guard  # noqa: E402

#: One representative query per capability the assignment names.
QUERIES: list[tuple[str, str, str]] = [
    (
        "monthly_revenue", "ceo",
        "SELECT DATE_TRUNC(DATE(created_at), MONTH) AS month, "
        "ROUND(SUM(sale_price), 2) AS revenue, COUNT(DISTINCT order_id) AS orders "
        "FROM order_items WHERE status IN ('Complete', 'Shipped') "
        "GROUP BY month ORDER BY month DESC LIMIT 24",
    ),
    (
        "top_customers", "ceo",
        "SELECT user_id, ROUND(SUM(sale_price), 2) AS lifetime_spend, "
        "COUNT(DISTINCT order_id) AS orders FROM order_items "
        "WHERE status IN ('Complete', 'Shipped') GROUP BY user_id "
        "ORDER BY lifetime_spend DESC LIMIT 10",
    ),
    (
        "state_comparison", "ceo",
        "SELECT u.state, COUNT(DISTINCT u.id) AS customers, "
        "ROUND(SUM(oi.sale_price), 2) AS revenue, "
        "ROUND(AVG(oi.sale_price), 2) AS avg_item_price "
        "FROM order_items oi JOIN users u ON u.id = oi.user_id "
        "WHERE oi.status IN ('Complete', 'Shipped') "
        "AND u.state IN ('California', 'New York') GROUP BY u.state",
    ),
    (
        "category_performance", "ceo",
        "SELECT p.category, COUNT(*) AS items_sold, "
        "ROUND(SUM(oi.sale_price), 2) AS revenue, "
        "ROUND(AVG(oi.sale_price - p.cost), 2) AS avg_margin "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "WHERE oi.status IN ('Complete', 'Shipped') "
        "GROUP BY p.category ORDER BY revenue DESC LIMIT 15",
    ),
    (
        "channel_revenue", "ceo",
        "SELECT u.traffic_source, COUNT(DISTINCT u.id) AS customers, "
        "ROUND(SUM(oi.sale_price), 2) AS revenue "
        "FROM order_items oi JOIN users u ON u.id = oi.user_id "
        "WHERE oi.status IN ('Complete', 'Shipped') GROUP BY u.traffic_source "
        "ORDER BY revenue DESC",
    ),
    (
        "womenswear_revenue", "vp_women",
        "SELECT ROUND(SUM(sale_price), 2) AS total_revenue FROM order_items "
        "WHERE status IN ('Complete', 'Shipped')",
    ),
    (
        # Two shapes of "nothing matched", because they behave differently:
        # the grouped form returns zero rows, the bare aggregate returns one
        # row of NULL. Both must be reported as empty.
        "empty_grouped", "ceo",
        "SELECT status, ROUND(SUM(sale_price), 2) AS revenue FROM order_items "
        "WHERE status = 'NoSuchStatus' GROUP BY status",
    ),
    (
        "empty_aggregate", "ceo",
        "SELECT ROUND(SUM(sale_price), 2) AS revenue FROM order_items "
        "WHERE status = 'NoSuchStatus'",
    ),
]


def main() -> int:
    settings = get_settings()
    if not settings.gcp_project:
        print("GOOGLE_CLOUD_PROJECT is not set. See README for BigQuery setup.")
        return 1

    output = settings.fixtures_dir
    output.mkdir(parents=True, exist_ok=True)
    guard = build_guard(
        settings.bq_dataset,
        default_limit=settings.default_row_limit,
        max_limit=settings.max_row_limit,
    )

    try:
        executor = BigQueryExecutor(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach BigQuery: {exc}")
        return 1

    recorded = 0
    for name, user, raw_sql in QUERIES:
        checked = guard.check(raw_sql, get_principal(user))
        if not checked.ok:
            print(f"  SKIP {name}: guard rejected — {checked.reason()}")
            continue

        try:
            result = executor.execute(checked.sql)
        except Exception as exc:  # noqa: BLE001
            # Record the failure too: the repair path needs a fixture that
            # fails in order to be tested.
            payload = {
                "fingerprint": fingerprint(checked.sql),
                "sql": checked.sql, "rows": [], "error": str(exc)[:400],
            }
            (output / f"{name}.json").write_text(json.dumps(payload, indent=2))
            print(f"  ERR  {name}: {str(exc)[:70]} (recorded as a failure fixture)")
            continue

        payload = {
            "fingerprint": fingerprint(checked.sql),
            "name": name,
            "sql": checked.sql,
            "rows": json.loads(result.rows.to_json(orient="records", date_format="iso")),
            "bytes_processed": result.bytes_processed,
            "bytes_billed": result.bytes_billed,
            "duration_ms": round(result.duration_ms, 1),
        }
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
        recorded += 1
        print(
            f"  ok   {name:22} {result.row_count:>5} rows  "
            f"{result.bytes_billed / 1e6:>7.1f} MB  "
            f"${result.estimated_cost_usd():.6f}"
        )

    print(f"\nRecorded {recorded}/{len(QUERIES)} fixtures into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
