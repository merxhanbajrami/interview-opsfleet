"""Output scrubbing: the layer that runs after the guard has already passed."""

from __future__ import annotations

import pandas as pd
import pytest

from insight_agent.security.scrubber import (
    pseudonymize,
    pseudonymize_frame,
    scrub_records,
    scrub_text,
)


def test_pseudonyms_are_stable_and_distinct():
    assert pseudonymize(4821) == pseudonymize(4821)
    assert pseudonymize(4821) != pseudonymize(4822)
    assert pseudonymize(4821).startswith("CUST-")


def test_pseudonyms_do_not_reveal_the_identifier():
    assert "4821" not in pseudonymize(4821)


@pytest.mark.parametrize(
    "text",
    [
        "Contact maya@thelook.com about this.",
        "SSN 123-45-6789 on record.",
        "Call (415) 555-0132 today.",
        "Ships to 1420 Mission Street.",
        "Home at 37.774929, -122.419418.",
        "Card 4111 1111 1111 1111 declined.",
    ],
)
def test_identifiers_are_redacted(text):
    assert scrub_text(text).redactions


@pytest.mark.parametrize(
    "text",
    [
        "Total revenue was 1234567890123 cents.",
        "Revenue 1234567.89 and 9876543.21 combined.",
        "Q1 ended at 12345678.90 in sales, up 12%.",
        "We processed 987654321098765 events.",
        "Order 10023 shipped on 2024-03-14.",
    ],
)
def test_legitimate_metrics_are_not_redacted(text):
    """A false positive here corrupts the analysis, which is worse than the
    failure the pattern is meant to prevent. Card numbers are Luhn-checked."""
    result = scrub_text(text)
    assert result.clean, f"metric wrongly redacted: {result.text}"


def test_frame_customer_ids_become_pseudonyms():
    frame = pd.DataFrame({"user_id": [4821, 4822], "state": ["CA", "NY"]})
    out, count = pseudonymize_frame(frame)
    assert count == 1
    assert all(str(v).startswith("CUST-") for v in out["user_id"])
    assert list(out["state"]) == ["CA", "NY"]


def test_records_are_cleaned_before_reaching_the_model():
    records = scrub_records([{"user_id": 4821, "note": "reach vip@x.com", "spend": 1200.5}])
    assert records[0]["user_id"].startswith("CUST-")
    assert "vip@x.com" not in records[0]["note"]
    assert records[0]["spend"] == 1200.5


def test_empty_input_is_safe():
    assert scrub_text("").text == ""
    out, count = pseudonymize_frame(pd.DataFrame())
    assert count == 0
