"""The Golden Bucket: how analysts answered questions like this one before.

Requirement 1 says the agent cannot rely on the warehouse alone.  A schema
tells the model that ``order_items.status`` exists; it does not tell it that
this company's analysts count 'Complete' and 'Shipped' as realised demand and
exclude 'Returned'.  That interpretation lives in the trios — question, query,
report — that human analysts produced, and it is what turns a schema-correct
query into a business-correct one.

**Retrieval at query time.**  The incoming question is scored against every
stored question with BM25 and the best few are injected as few-shot examples
into the SQL prompt.  Scores below a floor are discarded: an irrelevant example
is worse than none, because the model will try to follow it.

**Why lexical and not embeddings.**  The provider is pluggable and not every
provider exposes an embeddings endpoint, so an embedding index would bind the
whole system back to one vendor. At the scale of a curated analyst library —
hundreds to low thousands of trios — BM25 over question text plus tag matching
retrieves well. The interface below is the seam: swapping in Vertex AI Vector
Search or pgvector means implementing ``score`` differently, and nothing above
this module changes. DESIGN.md covers the production topology.

**Updating the bucket.**  See ``promote_candidate``. New trios are not written
by the agent directly; they are queued for analyst review, because a bucket
that learns from its own unreviewed output drifts.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: Below this BM25 score a match is treated as unrelated.
RELEVANCE_FLOOR = 1.5

_STOPWORDS = frozenset(
    """a an and are as at be by do does for from how in is it of on or our that the
    to was were what when where which who why with you your we us""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class Trio:
    """One analyst-authored question, query and conclusion."""

    id: str
    question: str
    sql: str
    insight: str
    tags: list[str] = field(default_factory=list)
    author: str = ""
    created_at: str = ""
    #: Set when the trio came from a reviewed agent answer rather than an
    #: analyst writing from scratch.
    promoted_from: str = ""

    def as_example(self) -> dict[str, str]:
        return {"question": self.question, "sql": self.sql, "insight": self.insight}


class GoldenBucket:
    """BM25 index over the analyst trio library."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.trios: list[Trio] = []
        self._docs: list[list[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0
        self.load()

    # --- Index ------------------------------------------------------------

    def load(self) -> None:
        """Read every trio from disk and rebuild the index."""
        self.trios = []
        if self.directory.is_dir():
            for path in sorted(self.directory.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    log.warning("skipping unreadable trio %s: %s", path.name, exc)
                    continue
                for record in data if isinstance(data, list) else [data]:
                    try:
                        self.trios.append(
                            Trio(
                                id=str(record.get("id") or path.stem),
                                question=record["question"],
                                sql=record["sql"],
                                insight=record.get("insight", ""),
                                tags=list(record.get("tags", [])),
                                author=record.get("author", ""),
                                created_at=record.get("created_at", ""),
                                promoted_from=record.get("promoted_from", ""),
                            )
                        )
                    except KeyError as exc:
                        log.warning("trio in %s missing field %s", path.name, exc)
        self._build()

    def _build(self) -> None:
        # Index question text plus tags. Not the SQL: matching on SQL tokens
        # retrieves examples that look alike rather than ones that mean alike.
        self._docs = [tokenize(t.question + " " + " ".join(t.tags)) for t in self.trios]
        self._df = Counter()
        for doc in self._docs:
            self._df.update(set(doc))
        self._avg_len = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0

    # --- Retrieval --------------------------------------------------------

    def score(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, Trio]]:
        """BM25 score of every trio against ``query``, best first."""
        if not self._docs:
            return []
        terms = tokenize(query)
        total = len(self._docs)
        scored: list[tuple[float, Trio]] = []

        for doc, trio in zip(self._docs, self.trios, strict=True):
            if not doc:
                continue
            counts = Counter(doc)
            length = len(doc)
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = freq + k1 * (1 - b + b * length / (self._avg_len or 1))
                score += idf * (freq * (k1 + 1)) / denominator
            if score > 0:
                scored.append((score, trio))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def retrieve(self, query: str, limit: int = 3) -> list[Trio]:
        """Best few trios above the relevance floor.

        Returning nothing is a valid outcome. A weak match steers the model
        toward the wrong metric, which is harder to notice than no example.
        """
        return [trio for score, trio in self.score(query)[:limit] if score >= RELEVANCE_FLOOR]

    def examples(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        return [trio.as_example() for trio in self.retrieve(query, limit)]

    # --- Growth -----------------------------------------------------------

    def promote_candidate(
        self,
        *,
        question: str,
        sql: str,
        insight: str,
        tags: list[str] | None = None,
        source_trace: str = "",
        review_dir: Path | None = None,
    ) -> Path:
        """Queue a successful answer for analyst review.

        Deliberately writes to a review directory, not to the live library. An
        agent that promotes its own unreviewed output into its own few-shot
        examples reinforces whatever it got wrong. A human approves the
        promotion by moving the file.
        """
        target = review_dir or (self.directory.parent / "review_queue")
        target.mkdir(parents=True, exist_ok=True)
        from insight_agent.store.db import utcnow

        slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:48].strip("-")
        path = target / f"{slug or 'candidate'}-{source_trace[:8] or 'manual'}.json"
        path.write_text(
            json.dumps(
                {
                    "question": question,
                    "sql": sql,
                    "insight": insight,
                    "tags": tags or [],
                    "created_at": utcnow(),
                    "promoted_from": source_trace,
                    "status": "awaiting_review",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def __len__(self) -> int:
        return len(self.trios)
