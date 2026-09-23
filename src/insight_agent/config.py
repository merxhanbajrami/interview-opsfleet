"""Central configuration.

Every tunable value in the system is declared here and sourced from the
environment (or a local ``.env``).  Nothing else in the codebase reads
``os.environ`` directly, so the full operational surface of the agent is
visible in one file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, read once per process."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="INSIGHT_",
        extra="ignore",
    )

    # --- Credentials ------------------------------------------------------
    # These two keep their conventional names so the Google SDKs pick them up.
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gcp_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")

    # --- Models -----------------------------------------------------------
    # The provider decides which company answers; the three roles are the same
    # whichever it is.
    #   primary  : reasoning, SQL synthesis, report writing.
    #   fast     : routing, classification, short structured calls.
    #   fallback : used only when `primary` fails or is rate-limited.
    # Leave the three model names unset to take the provider's defaults.
    llm_provider: str = "google"
    model_primary_override: str = ""
    model_fast_override: str = ""
    model_fallback_override: str = ""

    @property
    def provider_spec(self):
        from insight_agent.llm.providers import get_spec

        return get_spec(self.llm_provider)

    @property
    def model_primary(self) -> str:
        return self.model_primary_override or self.provider_spec.primary

    @property
    def model_fast(self) -> str:
        return self.model_fast_override or self.provider_spec.fast

    @property
    def model_fallback(self) -> str:
        return self.model_fallback_override or self.provider_spec.fallback

    @property
    def llm_api_key(self) -> str:
        """The key for the selected provider, read from its conventional name."""
        import os

        env_name = self.provider_spec.api_key_env
        if not env_name:
            return ""  # local provider
        if env_name == "GOOGLE_API_KEY":
            return self.google_api_key
        return os.environ.get(env_name, "")

    llm_timeout_seconds: float = 60.0
    llm_max_attempts: int = 3
    # Deterministic by default.  SQL generation must not be creative.
    llm_temperature_precise: float = 0.0
    llm_temperature_prose: float = 0.4

    # --- Data plane -------------------------------------------------------
    executor: str = "bigquery"  # "bigquery" | "recorded"
    bq_dataset: str = "bigquery-public-data.thelook_ecommerce"
    bq_location: str = "US"
    bq_timeout_seconds: float = 90.0
    # Hard ceiling enforced by BigQuery itself: the job is killed, not billed,
    # if it would exceed this.  Our own dry-run check refuses earlier and
    # more gracefully.
    bq_max_bytes_billed: int = 2_000_000_000  # 2 GB
    bq_dry_run_warn_bytes: int = 500_000_000  # 500 MB
    # Every generated query is capped, whether or not the model remembered to.
    default_row_limit: int = 1_000
    max_row_limit: int = 10_000

    # --- Agent behaviour --------------------------------------------------
    # Bounded self-correction. The cycle in the graph cannot spin forever.
    max_sql_repair_attempts: int = 2
    max_turn_seconds: float = 180.0
    # Circuit breaker: consecutive failures before a dependency is cut off.
    breaker_failure_threshold: int = 4
    breaker_reset_seconds: float = 60.0

    # --- Storage ----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / ".insight_data"
    personas_dir: Path = PROJECT_ROOT / "personas"
    golden_dir: Path = PROJECT_ROOT / "golden_bucket" / "trios"
    fixtures_dir: Path = PROJECT_ROOT / "evals" / "fixtures"
    default_persona: str = "default"

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    # Structured events are always written to disk; the console stays clean
    # for the user unless this is on.
    console_debug: bool = False

    # --- Derived paths ----------------------------------------------------
    @property
    def checkpoint_db(self) -> Path:
        """LangGraph conversation checkpoints (durable agent state)."""
        return self.data_dir / "checkpoints.db"

    @property
    def app_db(self) -> Path:
        """Saved reports, user preferences, audit log, traces."""
        return self.data_dir / "app.db"

    @property
    def trace_log(self) -> Path:
        """Newline-delimited JSON event stream."""
        return self.data_dir / "traces.jsonl"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- Readiness --------------------------------------------------------
    def missing_credentials(self) -> list[str]:
        """Return the names of credentials needed but absent.

        Checked at startup so the CLI can explain what to do instead of
        failing deep inside a node with an opaque SDK error.
        """
        missing: list[str] = []
        spec = self.provider_spec
        if spec.needs_key and not self.llm_api_key:
            missing.append(spec.api_key_env)
        if self.executor == "bigquery" and not self.gcp_project:
            missing.append("GOOGLE_CLOUD_PROJECT")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
