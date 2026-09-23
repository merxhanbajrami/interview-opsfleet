"""Detecting "nothing matched" (requirement 5, empty returns).

Found by running against live BigQuery: an aggregate over no matching rows
returns one row of NULL rather than zero rows, so the obvious ``rows.empty``
check misses the case entirely and hands the model a null to explain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from insight_agent.data.executor import QueryResult


def result(frame: pd.DataFrame) -> QueryResult:
    return QueryResult(rows=frame, sql="SELECT 1")


def test_zero_rows_is_empty():
    assert result(pd.DataFrame()).is_empty


def test_an_aggregate_over_no_rows_is_empty():
    """SELECT SUM(x) WHERE <matches nothing> returns one NULL row."""
    assert result(pd.DataFrame([{"revenue": None}])).is_empty


def test_a_row_of_all_nulls_is_empty():
    assert result(pd.DataFrame([{"revenue": None, "orders": None}])).is_empty


@pytest.mark.parametrize(
    "rows",
    [
        [{"revenue": 5_984_748.88}],
        [{"revenue": 0.0}],                       # zero revenue is an answer
        [{"revenue": 1.0, "returned": None}],     # partially null is still data
        [{"m": 1}, {"m": 2}],
    ],
)
def test_real_results_are_not_empty(rows):
    assert not result(pd.DataFrame(rows)).is_empty


def test_zero_is_distinguished_from_nothing():
    """The distinction the fix exists for.

    "Revenue was $0" and "no orders matched your filter" are different
    answers and must not be collapsed.
    """
    assert not result(pd.DataFrame([{"revenue": 0.0}])).is_empty
    assert result(pd.DataFrame([{"revenue": None}])).is_empty


# --- Nullable dtypes --------------------------------------------------------


def test_bigquery_nullable_dtypes_are_handled():
    """The shape that actually comes back from BigQuery.

    ``to_dataframe()`` returns pandas *nullable* dtypes, so a missing value is
    ``pd.NA`` rather than ``float('nan')``. The usual ``value != value`` NaN
    check evaluates to ``pd.NA`` for those, which raises when used in a
    boolean context -- so the check silently fell through to "not empty".
    """
    frame = pd.DataFrame(
        {
            "revenue": pd.array([None], dtype="Float64"),
            "orders": pd.array([0], dtype="Int64"),
        }
    )
    assert result(frame).is_empty


def test_nullable_dtypes_with_real_data_are_not_empty():
    frame = pd.DataFrame(
        {
            "revenue": pd.array([0.0], dtype="Float64"),
            "orders": pd.array([12], dtype="Int64"),
        }
    )
    assert not result(frame).is_empty


def test_sum_null_with_count_zero_means_nothing_matched():
    """SQL is inconsistent here: SUM over an empty set is NULL, COUNT is 0.

    The NULL is the discriminator. SUM over rows that happen to total zero
    returns 0.0, never NULL, so a row carrying a NULL and no non-zero value
    means nothing matched.
    """
    assert result(pd.DataFrame([{"revenue": None, "orders": 0}])).is_empty
    assert not result(pd.DataFrame([{"revenue": 0.0, "orders": 12}])).is_empty
