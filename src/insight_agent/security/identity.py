"""Who is asking, and what they are allowed to analyse.

The assignment states that each user may only analyse "products related to
him".  We model that as a *product scope*: a set of departments and/or
categories attached to the principal.  A merchandising VP for Womenswear
asking "what were our top sellers last month" must get a different answer from
the CEO asking the identical question, without either of them phrasing it
differently.

The scope is never passed to the model as an instruction.  It is compiled into
a SQL predicate and applied by rewriting the query (see
``security.sql_guard``), so no prompt can talk the agent out of it.

In production this comes from the identity provider — Okta or Google Workspace
groups via OIDC claims — and is cached per session.  The file-backed directory
here keeps the prototype runnable without an IdP.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated user together with their data entitlements."""

    user_id: str
    display_name: str
    role: str
    #: Departments this user may analyse. Empty means unrestricted.
    departments: frozenset[str] = field(default_factory=frozenset)
    #: Categories this user may analyse. Empty means all within `departments`.
    categories: frozenset[str] = field(default_factory=frozenset)
    #: Default report persona; the user may switch it at runtime.
    persona: str = "default"

    @property
    def is_unrestricted(self) -> bool:
        return not self.departments and not self.categories

    def scope_description(self) -> str:
        """Plain-language scope, shown in the CLI header and in report footers."""
        if self.is_unrestricted:
            return "all products"
        parts: list[str] = []
        if self.departments:
            parts.append(" / ".join(sorted(self.departments)) + " department")
        if self.categories:
            parts.append(" / ".join(sorted(self.categories)))
        return ", ".join(parts)

    def scope_predicate(self, alias: str = "") -> str:
        """The SQL predicate enforcing this principal's product entitlement.

        Returned as a fragment to be applied to the ``products`` table. String
        values are single-quote escaped; the underlying values come from the
        identity directory rather than from user input, but escaping here keeps
        the guarantee local and auditable.
        """
        if self.is_unrestricted:
            return "TRUE"
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        if self.departments:
            clauses.append(f"{prefix}department IN ({_sql_list(self.departments)})")
        if self.categories:
            clauses.append(f"{prefix}category IN ({_sql_list(self.categories)})")
        return " AND ".join(clauses)


def _sql_list(values: frozenset[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in sorted(values))


# --- Demo directory ---------------------------------------------------------
# Three principals with deliberately different entitlements, so the same
# question can be shown returning different data. Swap for OIDC claims in
# production.

DIRECTORY: dict[str, Principal] = {
    "ceo": Principal(
        user_id="ceo",
        display_name="Dana Whitfield",
        role="Chief Executive Officer",
        persona="executive_brief",
    ),
    "vp_women": Principal(
        user_id="vp_women",
        display_name="Maya Ortiz",
        role="VP, Womenswear",
        departments=frozenset({"Women"}),
        persona="default",
    ),
    "vp_men": Principal(
        user_id="vp_men",
        display_name="Tom Becker",
        role="VP, Menswear",
        departments=frozenset({"Men"}),
        persona="default",
    ),
    "buyer_outerwear": Principal(
        user_id="buyer_outerwear",
        display_name="Priya Raman",
        role="Senior Buyer, Outerwear",
        categories=frozenset({"Outerwear & Coats", "Sweaters"}),
        persona="default",
    ),
}

DEFAULT_PRINCIPAL = "ceo"


def get_principal(user_id: str) -> Principal:
    """Resolve a principal, raising rather than silently granting wide access."""
    try:
        return DIRECTORY[user_id]
    except KeyError:
        known = ", ".join(sorted(DIRECTORY))
        raise KeyError(f"Unknown user '{user_id}'. Known users: {known}") from None


def list_principals() -> list[Principal]:
    return list(DIRECTORY.values())
