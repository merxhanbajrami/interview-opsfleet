"""Model provider registry.

The agent depends on a chat model, not on a vendor.  Every node calls
``LLMClient``; only this module knows which company answers.  Switching
provider is an environment variable, which matters for three reasons:

* the free tiers that make this prototype runnable have tight rate limits, so
  being able to move is practical rather than theoretical;
* the same graph must be able to run on a self-hosted model where data
  residency rules forbid a third-party API;
* "reasoning for the chosen LLM" is a design question, and a design that
  cannot be changed has not really chosen anything.

Each entry declares its own default model names so that a switch does not
require editing three other settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """How to reach one provider, and what to call by default."""

    name: str
    package: str
    #: Import path of the LangChain chat class.
    module: str
    class_name: str
    api_key_env: str
    primary: str
    fast: str
    fallback: str
    #: Tried in order when `primary` is exhausted. A single fallback is not
    #: enough on a shared free tier, where several models can be rate-limited
    #: at once and the chain has to keep stepping down until one answers.
    chain: tuple[str, ...] = ()
    notes: str = ""

    def fallbacks_for(self, model: str) -> list[str]:
        """Models to try after ``model``, in order, excluding it."""
        ordered = list(self.chain) or [self.fallback, self.fast]
        seen: set[str] = {model}
        out: list[str] = []
        for candidate in ordered:
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_env)


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        name="google",
        package="langchain-google-genai",
        module="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        api_key_env="GOOGLE_API_KEY",
        # gemini-3.8-flash is the strongest model, but it has no free-tier
        # quota: a free key gets 429 RESOURCE_EXHAUSTED on the first call.
        # 3.7-flash is the newest model that a free key can actually use.
        # Override with INSIGHT_MODEL_PRIMARY_OVERRIDE on a billed project.
        primary="gemini-3.7-flash",
        fast="gemini-3.5-flash-lite",
        fallback="gemini-3.5-flash",
        chain=(
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ),
        notes=(
            "Free key from aistudio.google.com/apikey, no billing account. "
            "Note gemini-3.8-flash needs billing; the defaults here do not."
        ),
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        package="langchain-anthropic",
        module="langchain_anthropic",
        class_name="ChatAnthropic",
        api_key_env="ANTHROPIC_API_KEY",
        primary="claude-sonnet-5",
        fast="claude-haiku-4-5-20251001",
        fallback="claude-haiku-4-5-20251001",
        chain=("claude-haiku-4-5-20251001",),
        notes="Paid from the first call; no free tier.",
    ),
    "openai": ProviderSpec(
        name="openai",
        package="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        api_key_env="OPENAI_API_KEY",
        primary="gpt-4.1",
        fast="gpt-4.1-mini",
        fallback="gpt-4.1-mini",
        notes="Set INSIGHT_MODEL_PRIMARY to pin a newer model.",
    ),
    "groq": ProviderSpec(
        name="groq",
        package="langchain-groq",
        module="langchain_groq",
        class_name="ChatGroq",
        api_key_env="GROQ_API_KEY",
        primary="llama-3.3-70b-versatile",
        fast="llama-3.1-8b-instant",
        fallback="llama-3.1-8b-instant",
        notes="Free key, no billing details. Generous rate limits.",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        package="langchain-ollama",
        module="langchain_ollama",
        class_name="ChatOllama",
        api_key_env="",
        primary="llama3.1:8b",
        fast="llama3.1:8b",
        fallback="llama3.1:8b",
        notes="Runs locally. No account, no key, no data leaves the machine.",
    ),
}

DEFAULT_PROVIDER = "google"


class ProviderUnavailableError(RuntimeError):
    """The selected provider's package is not installed."""


def get_spec(name: str) -> ProviderSpec:
    try:
        return PROVIDERS[name.lower()]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise KeyError(f"Unknown LLM provider '{name}'. Available: {known}") from None


def build_chat_model(
    spec: ProviderSpec, model: str, *, temperature: float, timeout: float, api_key: str
) -> Any:
    """Instantiate the provider's chat model.

    The import is deferred so that an unused provider's package need not be
    installed. Only the selected one has to be present.
    """
    try:
        module = __import__(spec.module, fromlist=[spec.class_name])
        cls = getattr(module, spec.class_name)
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"Provider '{spec.name}' needs the '{spec.package}' package. "
            f"Install it with: uv add {spec.package}"
        ) from exc

    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}

    # Provider constructors disagree about these two argument names.
    if spec.name == "google":
        kwargs["google_api_key"] = api_key or None
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0
    elif spec.name == "anthropic":
        kwargs["api_key"] = api_key or None
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0
    elif spec.name == "openai":
        kwargs["api_key"] = api_key or None
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0
    elif spec.name == "groq":
        kwargs["api_key"] = api_key or None
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0
    # Ollama is local: no key, and it manages its own timeouts.

    return cls(**kwargs)
