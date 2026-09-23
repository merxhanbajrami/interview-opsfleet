"""Static analysis and rewriting of model-generated SQL.

A prompt telling the model not to select users.email is a request, not a
control. This guard parses generated SQL into an AST and decides on
structure, so a query that would expose personal data never reaches BigQuery.

It both rejects and rewrites. Rewriting matters as much: the caller's
entitlement is applied by transforming the query tree, so the model cannot
omit a filter it was never responsible for writing.

At scale this control belongs in the warehouse, as authorized views or
row-level policies. See DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from insight_agent.data.catalog import (
    ALLOWED_TABLES,
    all_blocked_names,
    blocked_columns,
    known_column_names,
)
from insight_agent.security.identity import Principal

DIALECT = "bigquery"

#: Statement types that must never appear. The connection is read-only, but
#: defence in depth costs nothing here and documents the intent.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.TruncateTable,
    exp.Grant,
)

#: Functions that reach outside the dataset or leak environment detail.
_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"external_query", "session_user", "net.host", "gethostname"}
)


class SqlGuardError(Exception):
    """Raised when a query cannot be made safe."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


@dataclass(slots=True)
class GuardResult:
    """Outcome of guarding one query."""

    ok: bool
    sql: str = ""
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scope_applied: bool = False
    limit_applied: int | None = None

    def reason(self) -> str:
        return "; ".join(self.violations)


class SqlGuard:
    """Validates and rewrites one statement at a time."""

    def __init__(self, dataset: str, default_limit: int = 1000, max_limit: int = 10_000) -> None:
        self.dataset = dataset
        self.default_limit = default_limit
        self.max_limit = max_limit
        self._blocked_by_table = blocked_columns()
        self._blocked_any = all_blocked_names()
        #: Every catalogued column name. A reference outside this set is
        #: rejected, which is what makes an unclassified column unreachable.
        self._known_columns = known_column_names()

    # --- Public entry point ----------------------------------------------

    def check(self, sql: str, principal: Principal) -> GuardResult:
        """Validate and rewrite ``sql`` for ``principal``.

        Returns a :class:`GuardResult`; never raises for a merely invalid
        query, because the caller turns violations into a repair prompt.
        """
        violations: list[str] = []
        warnings: list[str] = []

        try:
            statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
        except Exception as exc:  # noqa: BLE001 - parse errors are user-visible
            return GuardResult(ok=False, violations=[f"SQL could not be parsed: {exc}"])

        if not statements:
            return GuardResult(ok=False, violations=["No SQL statement found."])
        if len(statements) > 1:
            return GuardResult(
                ok=False,
                violations=[
                    f"Expected exactly one statement, found {len(statements)}. "
                    "Statement chaining is not permitted."
                ],
            )

        tree = statements[0]

        violations += self._check_read_only(tree)
        violations += self._check_functions(tree)
        # Structural problems make later checks meaningless; stop here.
        if violations:
            return GuardResult(ok=False, violations=violations)

        cte_names = self._cte_names(tree)
        violations += self._check_tables(tree, cte_names)
        violations += self._check_star(tree)
        violations += self._check_columns(tree, cte_names)

        if violations:
            return GuardResult(ok=False, violations=violations)

        # --- Rewrites -----------------------------------------------------
        tree, scope_applied = self._apply_scope_and_qualify(tree, principal, cte_names)
        tree, limit = self._enforce_limit(tree)
        if limit != self.default_limit:
            warnings.append(f"Row limit set to {limit}.")

        return GuardResult(
            ok=True,
            sql=tree.sql(dialect=DIALECT, pretty=True),
            warnings=warnings,
            scope_applied=scope_applied,
            limit_applied=limit,
        )

    # --- Checks -----------------------------------------------------------

    def _check_read_only(self, tree: exp.Expression) -> list[str]:
        found = {
            type(node).__name__.upper()
            for node in tree.walk()
            if isinstance(node, _FORBIDDEN_NODES)
        }
        if found:
            return [
                f"Only SELECT statements are permitted; found {', '.join(sorted(found))}. "
                "The database connection is read-only."
            ]
        if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
            return [f"Expected a SELECT statement, got {type(tree).__name__.upper()}."]
        return []

    def _check_functions(self, tree: exp.Expression) -> list[str]:
        bad: set[str] = set()
        for node in tree.find_all(exp.Anonymous):
            name = (node.this or "").lower() if isinstance(node.this, str) else ""
            if name in _FORBIDDEN_FUNCTIONS:
                bad.add(name)
        return [f"Function not permitted: {n}." for n in sorted(bad)]

    def _cte_names(self, tree: exp.Expression) -> set[str]:
        """Names introduced by WITH clauses, which are legal table references."""
        return {
            cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name
        }

    def _check_tables(self, tree: exp.Expression, cte_names: set[str]) -> list[str]:
        violations: list[str] = []
        for table in tree.find_all(exp.Table):
            name = (table.name or "").lower()
            if not name or name in cte_names:
                continue
            if name not in ALLOWED_TABLES:
                violations.append(
                    f"Table `{table.name}` is not available. "
                    f"Permitted tables: {', '.join(sorted(ALLOWED_TABLES))}."
                )
        return violations

    def _check_star(self, tree: exp.Expression) -> list[str]:
        """Reject ``SELECT *``.

        A star expands to whatever the table happens to contain, which for
        ``users`` includes every blocked column.  Requiring explicit columns
        makes the projection auditable and also bounds scan cost.
        """
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star):
                    return [
                        "SELECT * is not permitted. List the columns you need explicitly."
                    ]
                if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                    return [
                        f"SELECT {projection.sql(dialect=DIALECT)} is not permitted. "
                        "List the columns you need explicitly."
                    ]
        return []

    def _check_columns(self, tree: exp.Expression, cte_names: set[str]) -> list[str]:
        """Allow only catalogued columns. Everything else is rejected.

        An allowlist, not a denylist. A denylist is correct only while the
        catalog is exhaustive, so a column added upstream would be silently
        permitted. Here it is unreachable until someone classifies it.

        Checked in filters and joins as well as the projection, because
        ``WHERE email = 'x@y.com'`` returns no personal column but reveals
        whether one named person is a customer.
        """
        aliases = self._table_aliases(tree, cte_names)
        # Names the query itself introduces: `SUM(x) AS total`, and any
        # explicit CTE column list. These are legitimate references that no
        # catalog could know about.
        local_names = self._local_names(tree)

        violations: list[str] = []
        seen: set[str] = set()

        for column in tree.find_all(exp.Column):
            name = (column.name or "").lower()
            if not name or name in seen:
                continue
            qualifier = (column.table or "").lower()

            if qualifier:
                table = aliases.get(qualifier, qualifier)
                blocked = self._blocked_by_table.get(table, frozenset())
                if name in blocked:
                    seen.add(name)
                    violations.append(
                        f"Column `{name}` holds personal data and cannot be queried."
                    )
                    continue
            elif name in self._blocked_any:
                seen.add(name)
                violations.append(
                    f"Column `{name}` holds personal data and cannot be queried."
                )
                continue

            if name not in self._known_columns and name not in local_names:
                seen.add(name)
                violations.append(
                    f"Column `{name}` is not available in this dataset. "
                    "Use only the columns listed in the schema."
                )
        return violations

    def _local_names(self, tree: exp.Expression) -> set[str]:
        """Identifiers the query defines for itself.

        Result aliases and CTE column lists are referenced exactly like real
        columns, so an allowlist that did not account for them would reject
        ``SELECT SUM(sale_price) AS revenue ... ORDER BY revenue``.
        """
        names: set[str] = set()
        for alias in tree.find_all(exp.Alias):
            if alias.alias:
                names.add(alias.alias.lower())
        # `WITH t(pid, total) AS (...)` hangs the column list off the CTE's
        # TableAlias, not off the CTE itself.
        for cte in tree.find_all(exp.CTE):
            table_alias = cte.args.get("alias")
            columns = table_alias.args.get("columns") if table_alias else None
            for column in columns or []:
                name = getattr(column, "name", None) or str(column)
                names.add(str(name).lower())
        # The same form appears on plain derived tables and on table
        # functions, so pick those up wherever they occur.
        for table_alias in tree.find_all(exp.TableAlias):
            for column in table_alias.args.get("columns") or []:
                name = getattr(column, "name", None) or str(column)
                names.add(str(name).lower())
        # Bare column references that a SELECT re-exports keep their own name,
        # which the catalog already covers, so nothing more is needed here.
        return names

    def _table_aliases(self, tree: exp.Expression, cte_names: set[str]) -> dict[str, str]:
        """Map every alias to the real table it stands for."""
        mapping: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            real = (table.name or "").lower()
            if real in cte_names:
                continue
            mapping[real] = real
            alias = (table.alias or "").lower()
            if alias:
                mapping[alias] = real
        return mapping

    # --- Rewrites ---------------------------------------------------------

    def _apply_scope_and_qualify(
        self, tree: exp.Expression, principal: Principal, cte_names: set[str]
    ) -> tuple[exp.Expression, bool]:
        """Qualify table names and inject the principal's product entitlement.

        ``products`` is filtered directly.  ``order_items`` is filtered by
        product membership, because that is the revenue grain: without it a
        scoped user could compute company-wide revenue by never mentioning
        products at all.
        """
        scoped = not principal.is_unrestricted
        predicate = principal.scope_predicate("__p")
        applied = False

        def transform(node: exp.Expression) -> exp.Expression:
            nonlocal applied
            if not isinstance(node, exp.Table):
                return node
            name = (node.name or "").lower()
            if not name or name in cte_names:
                return node

            alias = node.alias
            if scoped and name == "products":
                applied = True
                replacement = (
                    f"(SELECT __p.* FROM `{self.dataset}.products` AS __p WHERE {predicate})"
                )
            elif scoped and name == "order_items":
                applied = True
                replacement = (
                    f"(SELECT __oi.* FROM `{self.dataset}.order_items` AS __oi "
                    f"WHERE __oi.product_id IN ("
                    f"SELECT __p.id FROM `{self.dataset}.products` AS __p WHERE {predicate}))"
                )
            else:
                # Build the identifier directly rather than re-parsing, so the
                # whole path stays inside one pair of backticks. The hyphen in
                # the public project id makes quoting mandatory.
                qualified = exp.Table(
                    this=exp.Identifier(this=f"{self.dataset}.{name}", quoted=True)
                )
                return qualified.as_(alias) if alias else qualified

            new_node = sqlglot.parse_one(replacement, dialect=DIALECT)
            if isinstance(new_node, exp.Subquery) or isinstance(new_node, exp.Table):
                if alias:
                    new_node = new_node.as_(alias)
                elif isinstance(new_node, exp.Subquery):
                    # A derived table must be named; reuse the original name so
                    # existing column qualifiers keep resolving.
                    new_node = new_node.as_(name)
            return new_node

        return tree.transform(transform), applied

    def _enforce_limit(self, tree: exp.Expression) -> tuple[exp.Expression, int]:
        """Guarantee a bounded result set.

        An unbounded query can return millions of rows into memory and into the
        model's context. The limit is applied whether or not the model asked
        for one, and clamped if it asked for too many.
        """
        existing = tree.args.get("limit") if isinstance(tree, (exp.Select, exp.Union)) else None
        if existing is not None:
            try:
                value = int(existing.expression.this)
            except (AttributeError, TypeError, ValueError):
                value = self.default_limit
            clamped = min(value, self.max_limit)
            if clamped != value:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(clamped)))
            return tree, clamped

        return tree.limit(self.default_limit), self.default_limit


def build_guard(dataset: str, default_limit: int = 1000, max_limit: int = 10_000) -> SqlGuard:
    return SqlGuard(dataset=dataset, default_limit=default_limit, max_limit=max_limit)
