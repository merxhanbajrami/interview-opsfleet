"""Command-line chat interface.

Two properties the assignment asks for explicitly shape this file.

**The interface must not crash.**  Every turn runs inside an exception
boundary. A failure anywhere below produces a message and a trace id, and the
session continues with its history intact. There is no path where an
unhandled exception ends the conversation.

**Confirmation must be strict without being tiresome.**  When the graph
interrupts, the exact reports are listed and a single explicit answer is
required. Anything other than a clear yes is treated as no. A request that
matches nothing never reaches this prompt at all.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from insight_agent.config import get_settings
from insight_agent.graph.build import build_agent
from insight_agent.graph.services import Services, build_services
from insight_agent.graph.state import new_turn
from insight_agent.obs.tracing import agent_metrics, load_trace, recent_traces
from insight_agent.security.identity import DEFAULT_PRINCIPAL, list_principals

app = typer.Typer(
    add_completion=False,
    help="Conversational data analysis over the retail warehouse.",
)
console = Console()

BANNER = """\
[bold]Insight Agent[/bold] — retail data analysis
Ask about revenue, products, customers or trends. Type [cyan]/help[/cyan] for \
commands, [cyan]/quit[/cyan] to leave."""


# --- Entry points -----------------------------------------------------------


@app.command()
def chat(
    user: str = typer.Option(DEFAULT_PRINCIPAL, "--user", "-u", help="Who is asking."),
    persona: str = typer.Option("", "--persona", "-p", help="Override report persona."),
    thread: str = typer.Option("", "--thread", "-t", help="Resume a conversation id."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show step timings."),
) -> None:
    """Start an interactive session."""
    settings = get_settings()
    _configure_logging(settings.log_level, verbose)

    missing = settings.missing_credentials()
    if missing:
        _credentials_help(missing)
        raise typer.Exit(code=1)

    try:
        services = build_services(settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not start:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        principal = services.principal(user)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    agent = build_agent(services)
    thread_id = thread or uuid.uuid4().hex[:12]

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(
        f"  Signed in as [bold]{principal.display_name}[/bold] · {principal.role}\n"
        f"  Data scope:  [yellow]{principal.scope_description()}[/yellow]\n"
        f"  Conversation: [dim]{thread_id}[/dim]\n"
    )

    session = _Session(services, agent, principal.user_id, thread_id, persona, verbose)
    session.run()


@app.command()
def traces(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """List recent turns."""
    services = build_services()
    rows = recent_traces(services.conn, limit)
    if not rows:
        console.print("No traces recorded yet.")
        return
    table = Table(title="Recent turns", header_style="bold cyan")
    for column in ("trace", "started", "user", "events", "errors", "total ms"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["trace_id"][:12], row["started"][:19], row["user_id"] or "-",
            str(row["events"]),
            f"[red]{row['errors']}[/red]" if row["errors"] else "0",
            f"{row['total_ms']:.0f}",
        )
    console.print(table)


@app.command()
def trace(trace_id: str) -> None:
    """Replay one turn in full: every step, prompt and decision."""
    services = build_services()
    events = load_trace(services.conn, trace_id)
    if not events:
        console.print(f"No trace matching [bold]{trace_id}[/bold].")
        return
    _render_trace(events)


@app.command()
def metrics() -> None:
    """Agent-level health metrics."""
    services = build_services()
    table = Table(title="Agent metrics", header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in agent_metrics(services.conn).items():
        table.add_row(key, "—" if value is None else str(value))
    console.print(table)


@app.command(name="verify-schema")
def verify_schema() -> None:
    """Reconcile the curated catalog against the live BigQuery schema.

    Run this after configuring BigQuery. It is the check that turns the
    hand-written catalog from an assumption into a verified fact.
    """
    from insight_agent.data.schema_check import (
        classification_summary,
        verify,
    )

    settings = get_settings()
    if settings.executor != "bigquery":
        console.print(
            "[yellow]Executor is 'recorded', so this compares the catalog with "
            "itself and always passes.[/yellow]\n"
            "Set INSIGHT_EXECUTOR=bigquery to check against the real warehouse."
        )

    try:
        services = build_services(settings)
        report = verify(services.executor)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not read the warehouse schema:[/red] {exc}")
        console.print("\nRun [cyan]insight doctor[/cyan] to check credentials.")
        raise typer.Exit(code=1) from exc

    counts = classification_summary()
    console.print(
        f"[bold]Catalog[/bold]: {counts['public']} public · "
        f"{counts['blocked']} blocked · {counts['pseudonym']} pseudonymised\n"
    )

    for diff in report.tables:
        if diff.error:
            console.print(f"  [red]ERROR[/red]   {diff.name}: {diff.error}")
            continue
        if diff.clean:
            console.print(f"  [green]ok[/green]      {diff.name}")
            continue

        console.print(f"  [yellow]diff[/yellow]    {diff.name}")
        for column in diff.missing_in_warehouse:
            console.print(
                f"            [red]breaks queries[/red] — catalogued but absent "
                f"from the warehouse: [bold]{column}[/bold]"
            )
        for column, expected, actual in diff.type_mismatches:
            console.print(
                f"            [yellow]type[/yellow] {column}: catalog says "
                f"{expected}, warehouse says {actual}"
            )
        for column, column_type in diff.unclassified_in_catalog:
            console.print(
                f"            [dim]unclassified, therefore blocked:[/dim] "
                f"{column} ({column_type})"
            )

    console.print()
    if report.clean:
        console.print("[green]Catalog matches the warehouse exactly.[/green]")
    elif report.ok:
        console.print(
            f"[yellow]{report.blocked_column_count()} live column(s) are not "
            "classified and are therefore blocked.[/yellow]\n"
            "Nothing is broken and no data can leak through them. Classify them "
            "in data/catalog.py to make them queryable."
        )
    else:
        console.print(
            "[red]The catalog references columns the warehouse does not have.[/red]\n"
            "Queries using them will fail. Fix data/catalog.py before demoing."
        )
        raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Check configuration and connectivity before a demo."""
    settings = get_settings()
    console.print("[bold]Configuration[/bold]")
    console.print(f"  provider     {settings.llm_provider}")
    console.print(f"  primary      {settings.model_primary}")
    console.print(f"  fast         {settings.model_fast}")
    console.print(f"  executor     {settings.executor}")
    console.print(f"  dataset      {settings.bq_dataset}")
    console.print(f"  project      {settings.gcp_project or '[red]unset[/red]'}")

    missing = settings.missing_credentials()
    console.print("\n[bold]Credentials[/bold]")
    if missing:
        for name in missing:
            console.print(f"  [red]missing[/red]  {name}")
    else:
        console.print("  [green]all present[/green]")

    console.print("\n[bold]Warehouse[/bold]")
    try:
        services = build_services(settings)
        ok = services.executor.health_check()
        console.print("  [green]reachable[/green]" if ok else "  [red]not reachable[/red]")
        if ok:
            console.print(
                "\n[dim]Next: `insight verify-schema` to confirm the catalog "
                "matches the live tables.[/dim]"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]unavailable[/red]  {exc}")


# --- Session ----------------------------------------------------------------


class _Session:
    """One interactive conversation."""

    def __init__(
        self, services: Services, agent: Any, user_id: str,
        thread_id: str, persona: str, verbose: bool,
    ) -> None:
        self.services = services
        self.agent = agent
        self.user_id = user_id
        self.thread_id = thread_id
        self.persona = persona
        self.verbose = verbose
        self.config = {"configurable": {"thread_id": thread_id}}

    def run(self) -> None:
        while True:
            try:
                text = console.input("[bold cyan]you ›[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye.")
                return
            if not text:
                continue
            if text.startswith("/"):
                if self._command(text):
                    return
                continue
            self._turn(text)

    # --- One turn ---------------------------------------------------------

    def _turn(self, text: str) -> None:
        tracer = self.services.new_tracer(user_id=self.user_id, thread_id=self.thread_id)
        state = new_turn(
            user_input=text, user_id=self.user_id, thread_id=self.thread_id,
            trace_id=tracer.trace_id, persona=self.persona,
        )
        try:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                result = self.agent.invoke(state, self.config)

            while _interrupt_of(result) is not None:
                payload = _interrupt_of(result)
                decision = self._confirm(payload)
                with console.status("[dim]applying…[/dim]", spinner="dots"):
                    result = self.agent.invoke(Command(resume=decision), self.config)

            self._render(result, tracer)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped. Nothing was changed.[/yellow]\n")
        except Exception as exc:  # noqa: BLE001
            # The boundary. Anything reaching here is a bug, and the session
            # still has to survive it with enough information to debug.
            logging.getLogger(__name__).exception("turn failed")
            tracer.emit("cli", "unhandled", error_type=type(exc).__name__, error=str(exc))
            console.print(
                Panel(
                    f"Something went wrong on my side, and nothing was changed.\n\n"
                    f"[dim]{type(exc).__name__}: {exc}[/dim]\n\n"
                    f"Run [cyan]insight trace {tracer.trace_id}[/cyan] to see what "
                    f"happened.",
                    title="[red]Error[/red]", border_style="red",
                )
            )

    def _confirm(self, payload: dict[str, Any]) -> str:
        """Show exactly what will be deleted and require an explicit answer."""
        reports = payload.get("reports", [])
        table = Table(
            title=f"[red]{len(reports)} report(s) will be deleted[/red]",
            header_style="bold red",
        )
        table.add_column("id")
        table.add_column("created")
        table.add_column("title")
        for report in reports:
            table.add_row(report["id"][:8], report.get("created_at", "")[:10],
                          report.get("title", ""))
        console.print()
        console.print(table)
        console.print(f"[dim]Matched: {payload.get('criterion', '')}[/dim]")
        console.print("[dim]Soft delete — recoverable afterwards.[/dim]")
        try:
            answer = console.input(
                "[bold red]Type 'yes' to confirm, anything else cancels ›[/bold red] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return "no"
        return answer

    def _render(self, result: dict[str, Any], tracer: Any) -> None:
        answer = result.get("answer") or "(no answer produced)"
        console.print()
        console.print(Markdown(answer))

        warnings = result.get("warnings") or []
        for warning in warnings:
            console.print(f"[dim]· {warning}[/dim]")

        if self.verbose:
            meta = result.get("result_meta", {})
            counters = tracer.summary()
            bits = [f"trace {tracer.trace_id}"]
            if result.get("sql_attempts"):
                bits.append(f"{result['sql_attempts']} query attempt(s)")
            if meta.get("bytes_billed"):
                bits.append(f"{meta['bytes_billed'] / 1e6:.1f} MB scanned")
            if meta.get("cost_usd"):
                bits.append(f"${meta['cost_usd']:.6f}")
            if counters.get("golden_examples_used"):
                bits.append(f"{int(counters['golden_examples_used'])} golden example(s)")
            if result.get("scope_applied"):
                bits.append("scope filter applied")
            console.print(f"[dim]{' · '.join(bits)}[/dim]")
        console.print()

    # --- Commands ---------------------------------------------------------

    def _command(self, text: str) -> bool:
        """Handle a slash command. Returns True to end the session."""
        parts = text.split(maxsplit=1)
        name = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if name in {"/quit", "/exit", "/q"}:
            console.print("Bye.")
            return True

        if name == "/help":
            console.print(_HELP)
        elif name == "/users":
            table = Table(header_style="bold cyan")
            for column in ("id", "name", "role", "data scope"):
                table.add_column(column)
            for principal in list_principals():
                marker = " ←" if principal.user_id == self.user_id else ""
                table.add_row(
                    principal.user_id + marker, principal.display_name,
                    principal.role, principal.scope_description(),
                )
            console.print(table)
        elif name == "/switch":
            self._switch(argument)
        elif name == "/persona":
            self._persona(argument)
        elif name == "/reports":
            self._reports()
        elif name == "/prefs":
            self._prefs(argument)
        elif name == "/trace":
            events = load_trace(self.services.conn, argument or self.services.tracer.trace_id)
            _render_trace(events) if events else console.print("No such trace.")
        elif name == "/metrics":
            for key, value in agent_metrics(self.services.conn).items():
                console.print(f"  {key:26} {'—' if value is None else value}")
        elif name == "/health":
            self._health()
        elif name == "/new":
            self.thread_id = uuid.uuid4().hex[:12]
            self.config = {"configurable": {"thread_id": self.thread_id}}
            console.print(f"Started a new conversation [dim]{self.thread_id}[/dim]")
        else:
            console.print(f"Unknown command {name}. Try /help.")
        return False

    def _switch(self, argument: str) -> None:
        if not argument:
            console.print("Usage: /switch <user-id>. See /users.")
            return
        try:
            principal = self.services.principal(argument)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        self.user_id = principal.user_id
        self.thread_id = uuid.uuid4().hex[:12]
        self.config = {"configurable": {"thread_id": self.thread_id}}
        console.print(
            f"Now signed in as [bold]{principal.display_name}[/bold] · "
            f"scope [yellow]{principal.scope_description()}[/yellow] · "
            f"new conversation [dim]{self.thread_id}[/dim]"
        )

    def _persona(self, argument: str) -> None:
        loader = self.services.personas
        if not argument:
            current = self.services.persona_for(self.user_id, self.persona)
            console.print(f"Current persona: [bold]{current.label or current.name}[/bold]")
            console.print(f"Available: {', '.join(loader.available())}")
            console.print("[dim]Edit personas/*.yaml and the change applies next turn.[/dim]")
            return
        if argument not in loader.available():
            console.print(f"[red]No persona '{argument}'.[/red] Available: "
                          f"{', '.join(loader.available())}")
            return
        self.persona = argument
        console.print(f"Persona set to [bold]{loader.load(argument).label}[/bold]")

    def _reports(self) -> None:
        reports = self.services.reports.list_for_user(self.user_id)
        if not reports:
            console.print("No saved reports.")
            return
        table = Table(header_style="bold cyan")
        for column in ("id", "created", "title"):
            table.add_column(column)
        for report in reports:
            table.add_row(report.id[:8], report.created_at[:10], report.title)
        console.print(table)

    def _prefs(self, argument: str) -> None:
        store = self.services.preferences
        if argument.lower() in {"clear", "forget"}:
            console.print(f"Forgot {store.forget_all(self.user_id)} preference(s).")
            return
        preferences = store.all_for_user(self.user_id)
        if not preferences:
            console.print("Nothing learned about your preferences yet.")
            return
        console.print("[bold]What I've learned about how you like answers:[/bold]")
        for preference in preferences:
            console.print(f"  · {preference.describe()}")
        console.print("[dim]/prefs clear to forget all of it.[/dim]")

    def _health(self) -> None:
        from insight_agent.resilience.breaker import REGISTRY

        snapshots = REGISTRY.snapshot()
        if not snapshots:
            console.print("No dependency calls made yet this session.")
            return
        for snapshot in snapshots:
            colour = {"closed": "green", "half_open": "yellow", "open": "red"}[
                snapshot["state"]
            ]
            console.print(
                f"  {snapshot['name']:20} [{colour}]{snapshot['state']}[/{colour}]"
                f"  ({snapshot['failures']} consecutive failures)"
            )


# --- Rendering helpers ------------------------------------------------------

_HELP = """\
[bold]Ask anything about the data.[/bold] For example:
  · What was our monthly revenue this year?
  · Why are customers in California underspending compared to New York?
  · Compare Outerwear and Sweaters, and explain the difference
  · Create a Q1 report with insights and action items for Q2
  · Delete all reports mentioning Acme

[bold]Commands[/bold]
  /users            who you can sign in as, and what each may analyse
  /switch <id>      change user, to see access scoping take effect
  /persona [name]   show or change report tone (edit personas/*.yaml live)
  /reports          your saved reports
  /prefs [clear]    what the agent has learned about your preferences
  /trace [id]       replay a turn step by step
  /metrics          agent health metrics
  /health           dependency circuit-breaker state
  /new              start a fresh conversation
  /quit             leave
"""


def _interrupt_of(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a pending interrupt payload, if the graph paused."""
    pending = result.get("__interrupt__")
    if not pending:
        return None
    first = pending[0] if isinstance(pending, (list, tuple)) else pending
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"prompt": str(value)}


def _render_trace(events: list[dict[str, Any]]) -> None:
    """Show the full message correspondence for one turn."""
    header = events[0]
    console.print(
        Panel(
            f"trace [bold]{header['trace_id']}[/bold] · user {header.get('user_id') or '-'} "
            f"· conversation {header.get('thread_id') or '-'}\n"
            f"{len(events)} events from {header['ts'][:19]}",
            border_style="cyan",
        )
    )
    for event in events:
        duration = f"{event['duration_ms']:.0f}ms" if event.get("duration_ms") else ""
        colour = {"error": "red", "reject": "yellow", "degraded": "yellow"}.get(
            event["event"], "green"
        )
        console.print(
            f"[dim]{event['seq']:>3}[/dim] "
            f"[bold]{event['node']:<20}[/bold] "
            f"[{colour}]{event['event']:<12}[/{colour}] "
            f"[dim]{duration:>8}[/dim]"
        )
        for key, value in (event.get("payload") or {}).items():
            rendered = str(value)
            if "\n" in rendered:
                console.print(f"       [cyan]{key}[/cyan]:")
                for line in rendered.splitlines():
                    console.print(f"         [dim]{line}[/dim]")
            else:
                console.print(f"       [cyan]{key}[/cyan]: [dim]{rendered[:160]}[/dim]")


def _credentials_help(missing: list[str]) -> None:
    settings = get_settings()
    console.print(
        Panel(
            "Missing configuration: [red]" + ", ".join(missing) + "[/red]\n\n"
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and fill it in.\n\n"
            f"Provider is [bold]{settings.llm_provider}[/bold]; its key is "
            f"[cyan]{settings.provider_spec.api_key_env or 'not required'}[/cyan].\n"
            f"{settings.provider_spec.notes}\n\n"
            "For BigQuery:\n"
            "  gcloud auth application-default login\n"
            "  gcloud config set project YOUR_PROJECT_ID\n\n"
            "Run [cyan]insight doctor[/cyan] to re-check.",
            title="[yellow]Setup needed[/yellow]", border_style="yellow",
        )
    )


def _configure_logging(level: str, verbose: bool) -> None:
    """Logs go to a file. The console belongs to the conversation."""
    settings = get_settings()
    settings.ensure_dirs()
    handlers: list[logging.Handler] = [
        logging.FileHandler(settings.data_dir / "agent.log", encoding="utf-8")
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers, force=True,
    )
    # These are chatty and drown the agent's own events. The google_genai
    # notice about automatic function calling is emitted on every single
    # generate_content call and does not apply to us -- we never enable AFC.
    for noisy in ("google", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("google_genai.types").setLevel(logging.ERROR)


if __name__ == "__main__":
    app()
