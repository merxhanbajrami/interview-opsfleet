"""Curated semantic catalog for ``thelook_ecommerce``.

This is deliberately hand-written rather than pulled from ``INFORMATION_SCHEMA``.
A raw schema dump tells the model that ``order_items.status`` is a STRING; it
does not tell it that the value ``Complete`` is the one that counts as revenue,
or that ``users.email`` must never be read.  Both facts change the SQL the
model writes, so both belong in the catalog.

Each column carries a :class:`Sensitivity`, and that classification is the
single source of truth for the SQL guard and the output scrubber.  Adding a
column to the dataset therefore cannot silently widen the PII surface: an
unclassified column is treated as blocked until someone classifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Sensitivity(StrEnum):
    """How a column may be used.

    The three levels exist because a binary allow/deny cannot express the
    actual requirement.  Executives must be able to ask "who are our top
    customers", which needs a per-customer identity, while never seeing a
    name or an email address.
    """

    PUBLIC = "public"
    #: Direct identifier. Rejected wherever it appears in a query — projection,
    #: filter or join. There is no analytical question that requires it.
    BLOCKED = "blocked"
    #: Identifier needed for joins and per-customer ranking, but never shown
    #: as itself. Replaced with a stable pseudonym on the way out.
    PSEUDONYM = "pseudonym"


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: str
    description: str
    sensitivity: Sensitivity = Sensitivity.PUBLIC


@dataclass(frozen=True, slots=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]
    grain: str = ""
    joins: tuple[str, ...] = field(default_factory=tuple)

    def column(self, name: str) -> Column | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


_PUB = Sensitivity.PUBLIC
_BLOCK = Sensitivity.BLOCKED
_PSEUDO = Sensitivity.PSEUDONYM


USERS = Table(
    name="users",
    description="Customer demographics. One row per registered customer.",
    grain="one row per customer",
    joins=("users.id = orders.user_id", "users.id = order_items.user_id"),
    columns=(
        Column("id", "INTEGER", "Customer identifier. Join key.", _PSEUDO),
        Column("first_name", "STRING", "Given name.", _BLOCK),
        Column("last_name", "STRING", "Family name.", _BLOCK),
        Column("email", "STRING", "Email address.", _BLOCK),
        Column("street_address", "STRING", "Street address.", _BLOCK),
        Column("postal_code", "STRING", "Postal code. Narrow enough to re-identify.", _BLOCK),
        Column("latitude", "FLOAT", "Home latitude.", _BLOCK),
        Column("longitude", "FLOAT", "Home longitude.", _BLOCK),
        Column("user_geom", "GEOGRAPHY", "Home location as a point.", _BLOCK),
        Column("age", "INTEGER", "Age in years. Safe in aggregate.", _PUB),
        Column("gender", "STRING", "Reported gender: 'M' or 'F'.", _PUB),
        Column("city", "STRING", "City of residence.", _PUB),
        Column("state", "STRING", "State or province. The usual regional dimension.", _PUB),
        Column("country", "STRING", "Country of residence.", _PUB),
        Column("traffic_source", "STRING", "Acquisition channel: Search, Organic, Facebook…", _PUB),
        Column("created_at", "TIMESTAMP", "Account registration time. Use for cohorts.", _PUB),
    ),
)

ORDERS = Table(
    name="orders",
    description="Order headers. One row per order, no monetary value.",
    grain="one row per order",
    joins=("orders.order_id = order_items.order_id", "orders.user_id = users.id"),
    columns=(
        Column("order_id", "INTEGER", "Order identifier.", _PUB),
        Column("user_id", "INTEGER", "Customer who placed the order.", _PSEUDO),
        Column(
            "status",
            "STRING",
            "Lifecycle state: Complete, Shipped, Processing, Cancelled, Returned. "
            "'Complete' plus 'Shipped' are realised demand; 'Cancelled' is not.",
            _PUB,
        ),
        Column("gender", "STRING", "Denormalised customer gender.", _PUB),
        Column("created_at", "TIMESTAMP", "Order placement time. The default time axis.", _PUB),
        Column("returned_at", "TIMESTAMP", "Return time; NULL if not returned.", _PUB),
        Column("shipped_at", "TIMESTAMP", "Dispatch time.", _PUB),
        Column("delivered_at", "TIMESTAMP", "Delivery time.", _PUB),
        Column("num_of_item", "INTEGER", "Item count on the order.", _PUB),
    ),
)

ORDER_ITEMS = Table(
    name="order_items",
    description=(
        "Order lines. This is the revenue table: sale_price lives here, not on "
        "orders. Any revenue question starts from this table."
    ),
    grain="one row per item sold",
    joins=(
        "order_items.order_id = orders.order_id",
        "order_items.product_id = products.id",
        "order_items.user_id = users.id",
    ),
    columns=(
        Column("id", "INTEGER", "Order-line identifier.", _PUB),
        Column("order_id", "INTEGER", "Parent order.", _PUB),
        Column("user_id", "INTEGER", "Purchasing customer.", _PSEUDO),
        Column("product_id", "INTEGER", "Product sold. Join to products.id.", _PUB),
        Column("inventory_item_id", "INTEGER", "Stock unit sold.", _PUB),
        Column("status", "STRING", "Line status, mirrors orders.status.", _PUB),
        Column(
            "sale_price",
            "FLOAT",
            "Price paid for this line, in USD. SUM this for revenue. "
            "Margin is sale_price minus products.cost.",
            _PUB,
        ),
        Column("created_at", "TIMESTAMP", "Line creation time.", _PUB),
        Column("shipped_at", "TIMESTAMP", "Line dispatch time.", _PUB),
        Column("delivered_at", "TIMESTAMP", "Line delivery time.", _PUB),
        Column("returned_at", "TIMESTAMP", "Line return time; NULL if kept.", _PUB),
    ),
)

PRODUCTS = Table(
    name="products",
    description=(
        "Product catalogue. Carries the department and category dimensions that "
        "per-user access scoping is enforced on."
    ),
    grain="one row per product",
    joins=("products.id = order_items.product_id",),
    columns=(
        Column("id", "INTEGER", "Product identifier.", _PUB),
        Column("name", "STRING", "Product name.", _PUB),
        Column("brand", "STRING", "Brand name.", _PUB),
        Column("category", "STRING", "Product category, e.g. 'Jeans', 'Outerwear & Coats'.", _PUB),
        Column("department", "STRING", "Top-level department: 'Men' or 'Women'.", _PUB),
        Column("sku", "STRING", "Stock-keeping unit.", _PUB),
        Column("cost", "FLOAT", "Unit cost to the business, USD. Margin input.", _PUB),
        Column("retail_price", "FLOAT", "List price, USD. Differs from realised sale_price.", _PUB),
        Column("distribution_center_id", "INTEGER", "Fulfilling distribution centre.", _PUB),
    ),
)


TABLES: dict[str, Table] = {t.name: t for t in (USERS, ORDERS, ORDER_ITEMS, PRODUCTS)}

#: The only tables the agent may reference. Anything else is rejected.
ALLOWED_TABLES: frozenset[str] = frozenset(TABLES)

#: Tables that carry a product dimension, so per-user scoping applies to them.
SCOPED_TABLES: frozenset[str] = frozenset({"products"})


def blocked_columns() -> dict[str, frozenset[str]]:
    """Per-table names that must never appear in a query."""
    return {
        name: frozenset(
            c.name.lower() for c in table.columns if c.sensitivity is Sensitivity.BLOCKED
        )
        for name, table in TABLES.items()
    }


def pseudonym_columns() -> dict[str, frozenset[str]]:
    """Per-table names that are queryable but must be masked in output."""
    return {
        name: frozenset(
            c.name.lower() for c in table.columns if c.sensitivity is Sensitivity.PSEUDONYM
        )
        for name, table in TABLES.items()
    }


def known_column_names() -> frozenset[str]:
    """Every column the catalog knows about, blocked ones included.

    The SQL guard uses this as an allowlist. A column absent from the catalog
    is therefore unreachable, which is what makes adding a column upstream
    safe by default: it cannot be queried until someone classifies it.
    """
    return frozenset(
        c.name.lower() for table in TABLES.values() for c in table.columns
    )


def all_blocked_names() -> frozenset[str]:
    """Flat set of blocked column names, for unqualified-reference checks."""
    return frozenset().union(*blocked_columns().values())


def render_for_prompt(dataset: str, include_blocked_notice: bool = True) -> str:
    """Render the catalog as the schema context given to the model.

    Blocked columns are omitted rather than listed-and-forbidden.  A column the
    model has never seen is one it cannot be talked into selecting; listing
    them would advertise exactly what an attacker should ask for.
    """
    lines: list[str] = [f"Dataset: `{dataset}`", ""]
    for table in TABLES.values():
        lines.append(f"### {table.name} — {table.description}")
        if table.grain:
            lines.append(f"Grain: {table.grain}")
        lines.append("")
        for col in table.columns:
            if col.sensitivity is Sensitivity.BLOCKED:
                continue
            note = " [masked in output]" if col.sensitivity is Sensitivity.PSEUDONYM else ""
            lines.append(f"- `{col.name}` ({col.type}){note} — {col.description}")
        if table.joins:
            lines.append("")
            lines.append("Joins: " + "; ".join(f"`{j}`" for j in table.joins))
        lines.append("")

    if include_blocked_notice:
        lines += [
            "### Access policy",
            "The schema above is complete. Columns holding personal data are "
            "not part of it and cannot be queried; do not guess at column "
            "names that are not listed. Customer identifiers may be used for "
            "joins and ranking, and are pseudonymised before display.",
            "",
        ]
    return "\n".join(lines)
