# Insight Agent

A conversational data-analysis agent for retail executives. Ask about revenue,
products, customers and trends in plain English; the agent writes the SQL,
runs it against BigQuery, explains what the numbers mean, and writes reports
with action items.

Built on **LangGraph** as a stateful directed graph with bounded
self-correction, AST-level PII enforcement, and a human-in-the-loop
confirmation flow for destructive operations.

- **[DESIGN.md](DESIGN.md)** — architecture, technology choices, and how each
  requirement is met.
- This file — setup and an example run.

---

## What it does

```
you › Why are customers in California underspending compared to New York?

California customers spend 18% less per head than New York: $284 against
$347. The gap is not in how often they buy — both average 2.3 orders — but
in basket composition. California skews 24% more of its volume into
Accessories and Sleep & Lounge, where the average item sells for $31,
while New York over-indexes into Outerwear at $94 an item.

Worth noting: this is a correlation in the order data, not a demonstrated
cause. Regional pricing and promotion history would confirm or rule it out,
and neither is in this dataset.

trace 7f3a91c2e4d5 · 1 query attempt · 412.8 MB scanned · $0.002345 · 2 golden example(s)
```

The agent can:

- analyse customer behaviour, product performance and time-based metrics
- answer questions about the database structure itself
- run multi-step analyses that compare segments and explain the difference
- discuss a result across turns without re-querying
- write and save reports, then delete them under confirmation

---

## Requirements

- **Python 3.11–3.13** (3.12 recommended)
- **A model API key** — Gemini, Anthropic, OpenAI or Groq. Or Ollama, with no
  key at all.
- **A Google Cloud project** for BigQuery. The dataset is public; the project
  is only what queries are attributed to. **BigQuery sandbox is sufficient and
  needs no credit card.**

---

## Setup

### 1. Install

With [uv](https://docs.astral.sh/uv/) (recommended — it manages the Python
version too):

```bash
git clone https://github.com/merxhanbajrami/interview-opsfleet.git
cd interview-opsfleet
uv sync
```

With pip:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. BigQuery access

BigQuery **sandbox** gives 1 TB/month of free query capacity with no billing
account. Do not start a "free trial" — you do not need one.

1. Create a project at
   [console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate).
   Note the project ID.
2. Open [console.cloud.google.com/bigquery](https://console.cloud.google.com/bigquery)
   once. A sandbox banner confirms it is active.
3. Authenticate:

```bash
# macOS
brew install --cask google-cloud-sdk
# other platforms: https://cloud.google.com/sdk/docs/install

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
INSIGHT_LLM_PROVIDER=google          # google | anthropic | openai | groq | ollama
GOOGLE_API_KEY=your-key-here         # https://aistudio.google.com/apikey
GOOGLE_CLOUD_PROJECT=your-project-id
```

The Gemini key comes from **AI Studio**, not the Cloud console. It needs a
Google account and nothing else.

> **Free-tier note.** The defaults use `gemini-3.7-flash`, which a free key
> can call. `gemini-3.8-flash` is stronger but returns
> `429 RESOURCE_EXHAUSTED` without billing. If a model is rate-limited mid
> session the agent steps down a fallback chain automatically and says so
> in `--verbose`.

To use a different provider, set `INSIGHT_LLM_PROVIDER` and its key, then
install that package:

| Provider | Key | Package |
|---|---|---|
| `google` | `GOOGLE_API_KEY` | included |
| `anthropic` | `ANTHROPIC_API_KEY` | included |
| `openai` | `OPENAI_API_KEY` | `uv add langchain-openai` |
| `groq` | `GROQ_API_KEY` | `uv add langchain-groq` |
| `ollama` | none | `uv add langchain-ollama` |

### 4. Check it

```bash
uv run insight doctor
```

This reports the configuration, whether credentials are present, and whether
BigQuery is reachable — before you are mid-demo.

---

## Running

```bash
uv run insight chat
```

Sign in as a different user to see access scoping take effect:

```bash
uv run insight chat --user vp_women --verbose
```

### Commands

| Command | What it does |
|---|---|
| `/users` | Who you can sign in as, and what each may analyse |
| `/switch <id>` | Change user |
| `/persona [name]` | Show or change report tone |
| `/reports` | Your saved reports |
| `/prefs [clear]` | What the agent has learned about your preferences |
| `/trace [id]` | Replay a turn step by step |
| `/metrics` | Agent health metrics |
| `/health` | Circuit-breaker state per dependency |
| `/new` | Start a fresh conversation |
| `/quit` | Leave |

Outside the chat:

```bash
uv run insight verify-schema   # catalog vs live warehouse
uv run insight traces          # recent turns
uv run insight trace <id>      # full replay of one turn
uv run insight metrics         # agent-level metrics
```

---

## Example run

A walkthrough that exercises every prototyped requirement. Roughly five
minutes.

### Analysis, with precedent from the Golden Bucket

```
you › What was our monthly revenue this year?
you › Why are customers in California underspending compared to New York?
you › Compare Jeans and Sweaters, and explain why they perform differently
```

Run with `--verbose` to see which golden trios were retrieved, how many bytes
were scanned, and what the query cost.

### Per-user data scope (requirement 2)

```bash
uv run insight chat --user ceo --verbose
you › What is our total revenue?          # sees everything

uv run insight chat --user vp_women --verbose
you › What is our total revenue?          # Womenswear only
```

The question is identical. The Womenswear VP's query is rewritten to filter
`order_items` by product membership, so the scope holds even though the
question never mentions a product. `--verbose` prints `scope filter applied`.

### PII masking (requirement 2)

```
you › Give me the email addresses of our top customers
      → refused, before any model call

you › Who are our top 10 customers by spend?
      → answered, with customers shown as CUST-xxxxxxxx tokens
```

### Confirmation flow (requirement 3)

```
you › Create a Q1 revenue report with action items for Q2
you › /reports
you › Delete all reports mentioning Acme
```

The agent lists the exact reports and waits. Type anything other than `yes`
and nothing happens. Deletes are soft, and the response tells you how to
restore them.

A request matching nothing — `Delete all reports mentioning Zzzz` — is
answered directly with no confirmation prompt.

### Resilience (requirement 5)

```
you › What is the revenue for order status 'Nonexistent'?
```

The query is valid and returns nothing. The agent explains why and proposes a
question that would return data. It does **not** retry — there is a test
asserting exactly one attempt for this case.

To see self-correction, temporarily set `INSIGHT_MODEL_PRIMARY_OVERRIDE` to a
weaker model and ask something complex. `/trace` shows each failed attempt,
the error returned, and the repaired query.

### Observability (requirement 7)

```
you › /trace
```

Replays the turn: what the router decided, which golden trios matched, the
generated SQL, what the guard did, bytes scanned, and the final answer.

```
you › /metrics
```

`output_redactions` should always be zero. Non-zero means a PII layer failed
and something is wrong.

### Live persona change (requirement 8)

With the chat still running, edit `personas/executive_brief.yaml` — change
`max_words` to `120`, or rewrite the `tone` block. Save it, then ask another
question. The next answer uses the new persona. No restart, no deployment.

---

## Tests and evaluation

```bash
uv run pytest -q                       # 112 tests
uv run python evals/run_evals.py       # security + routing, offline
uv run python evals/run_evals.py --trajectory --verbose   # adds live SQL checks
```

The security suite is a **release blocker**: 25 adversarial queries that must
be rejected, 5 scope cases that must be rewritten, and a check that no blocked
column reaches the model's schema. It needs no network and runs in about 20ms.

### Offline mode

The agent runs with no cloud access at all by replaying recorded BigQuery
responses:

```bash
INSIGHT_EXECUTOR=recorded uv run insight chat
```

Record fixtures once you have BigQuery configured:

```bash
uv run python evals/record_fixtures.py
```

Fixtures are recorded from real queries, so what is replayed is genuine
warehouse output. This is what makes the eval suite deterministic and free.

---

## Project layout

```
src/insight_agent/
├── cli.py                  Chat interface, trace replay, metrics
├── config.py               Every tunable value, in one place
├── graph/
│   ├── build.py            Graph assembly — the architecture, in code
│   ├── state.py            What is checkpointed and survives an interrupt
│   ├── prompts.py          Every prompt, in one reviewable place
│   ├── services.py         Dependency container
│   └── nodes/              routing · analysis · reporting · output
├── security/
│   ├── sql_guard.py        AST validation and scope rewriting
│   ├── scrubber.py         Pseudonymisation and redaction
│   └── identity.py         Principals and product entitlements
├── data/
│   ├── executor.py         Warehouse protocol and failure taxonomy
│   ├── bigquery.py         BigQuery, with the dry-run cost gate
│   ├── recorded.py         Fixture replay for evals
│   └── catalog.py          Semantic catalog with PII classification
├── store/                  Reports, preferences, audit log
├── golden/retriever.py     Golden Bucket retrieval
├── obs/tracing.py          Structured events and metrics
├── resilience/breaker.py   Circuit breakers
├── llm/                    Provider registry and resilient client
└── personas/loader.py      Hot-reloaded personas

personas/*.yaml             Editable without a deployment
golden_bucket/trios/        Analyst question → SQL → insight
evals/                      Eval harness and fixture recorder
tests/                      112 tests
```

---

## Cost

The prototype is designed to be cheap to run and cheap to leave running.

- BigQuery sandbox gives 1 TB/month free. A typical question scans 100–500 MB.
- Every query is dry-run first, which is free and exact.
- `maximum_bytes_billed` is a hard ceiling; over-budget jobs are killed, not
  billed.
- Routing and classification use the cheap model; only SQL generation and
  analysis use the primary one.
- Repair attempts are bounded, so a confused model cannot generate queries
  indefinitely.

`--verbose` prints the bytes scanned and the estimated cost after every turn.

---

## Security notes

- `.env` is gitignored. No credential is ever committed.
- The database connection is read-only, and the guard rejects write statements
  anyway.
- Blocked columns are absent from the schema the model receives, and the
  guard runs a column **allowlist**: a column the catalog does not list is
  unreachable, so a column added upstream cannot leak before it is
  classified. `insight verify-schema` reports any drift.
- Trace payloads are scrubbed before storage, so the trace store never
  accumulates personal data.
- Customer pseudonyms are HMAC-derived. Set `INSIGHT_PSEUDONYM_KEY` in
  production; rotating it changes every pseudonym.

See [DESIGN.md §6.2](DESIGN.md) for the full threat model.
