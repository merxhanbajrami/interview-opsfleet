"""Saved Reports library and the destructive-operation path (requirement 3)."""

from __future__ import annotations

import pytest

from insight_agent.store.db import connect
from insight_agent.store.reports import ReportStore


@pytest.fixture
def store(tmp_path):
    return ReportStore(connect(tmp_path / "app.db"))


def _seed(store, user="ceo", thread="t1"):
    a = store.save(user_id=user, thread_id=thread, title="Q1 Revenue",
                   body="Revenue grew. Client Acme drove most of it.", summary="Q1 up")
    b = store.save(user_id=user, thread_id=thread, title="Churn Review",
                   body="Churn spiked in March.", summary="churn")
    c = store.save(user_id=user, thread_id="t2", title="Acme Deep Dive",
                   body="Acme expanded.", summary="acme")
    return a, b, c


def test_reports_are_listed_newest_first(store):
    _seed(store)
    assert len(store.list_for_user("ceo")) == 3


def test_a_user_cannot_see_another_users_reports(store):
    _seed(store, user="ceo")
    _seed(store, user="vp_women")
    assert len(store.list_for_user("ceo")) == 3
    assert all(r.user_id == "ceo" for r in store.list_for_user("ceo"))


def test_a_user_cannot_fetch_another_users_report_by_id(store):
    a, _, _ = _seed(store, user="ceo")
    assert store.get(a.id, "ceo") is not None
    assert store.get(a.id, "vp_women") is None


# --- Resolving a phrase into an explicit set -------------------------------


def test_mentions_matches_title_and_body(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    assert batch.count == 2  # one by body, one by title
    assert "mentioning 'Acme'" in batch.criterion


def test_this_conversation_only_is_scoped_to_the_thread(store):
    _seed(store)
    batch = store.find_deletion_candidates(
        user_id="ceo", thread_id="t1", this_thread_only=True
    )
    assert batch.count == 2
    assert all(r.thread_id == "t1" for r in batch.reports)


def test_candidates_never_include_another_users_reports(store):
    _seed(store, user="ceo")
    _seed(store, user="vp_women")
    batch = store.find_deletion_candidates(user_id="vp_women", thread_id="t1", mentions="Acme")
    assert all(r.user_id == "vp_women" for r in batch.reports)


def test_a_request_matching_nothing_yields_an_empty_batch(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Zzzz")
    assert batch.is_empty


# --- Confirmation and application ------------------------------------------


def test_confirmed_batch_deletes_exactly_what_was_previewed(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    count, replay = store.apply_deletion(batch.batch_id, "ceo")
    assert count == 2 and not replay
    assert len(store.list_for_user("ceo")) == 1


def test_applying_twice_is_a_no_op(store):
    """LangGraph re-executes a node when an interrupt resumes. A delete that
    were not idempotent would run a second time on replay."""
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    first, _ = store.apply_deletion(batch.batch_id, "ceo")
    second, replay = store.apply_deletion(batch.batch_id, "ceo")
    assert first == 2
    assert second == 2 and replay is True
    assert len(store.list_for_user("ceo")) == 1


def test_applying_an_unconfirmed_batch_is_refused(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    with pytest.raises(PermissionError):
        store.apply_deletion(batch.batch_id, "ceo")


def test_another_user_cannot_apply_someone_elses_confirmation(store):
    _seed(store, user="ceo")
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    with pytest.raises(PermissionError):
        store.apply_deletion(batch.batch_id, "vp_women")


def test_the_confirmed_set_is_frozen_at_preview_time(store):
    """A report created after confirmation must not be swept up by it."""
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    store.save(user_id="ceo", thread_id="t1", title="New Acme note", body="Acme again")
    count, _ = store.apply_deletion(batch.batch_id, "ceo")
    assert count == 2
    assert any("New Acme" in r.title for r in store.list_for_user("ceo"))


def test_soft_deleted_reports_can_be_restored(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    store.apply_deletion(batch.batch_id, "ceo")
    assert store.restore_batch(batch.batch_id, "ceo") == 2
    assert len(store.list_for_user("ceo")) == 3


# --- Audit -----------------------------------------------------------------


def test_every_step_is_audited(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.register_confirmation(batch)
    store.apply_deletion(batch.batch_id, "ceo")
    actions = [r[0] for r in store.conn.execute("SELECT action FROM audit_log ORDER BY id")]
    assert "report.create" in actions
    assert "report.delete.preview" in actions
    assert "report.delete.apply" in actions


def test_cancellation_is_audited_and_deletes_nothing(store):
    _seed(store)
    batch = store.find_deletion_candidates(user_id="ceo", thread_id="t1", mentions="Acme")
    store.record_cancellation(batch)
    assert len(store.list_for_user("ceo")) == 3
    actions = [r[0] for r in store.conn.execute("SELECT action FROM audit_log")]
    assert "report.delete.cancel" in actions
