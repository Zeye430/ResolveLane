import asyncio
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import MIGRATIONS, CaseStore, Receipt, load_case_store
from resolvelane.domain.case import CaseMessage, CaseStatus, Issue, MessageRole
from resolvelane.domain.knowledge import Citation
from resolvelane.recognition import KeywordRecognizer, Recognition
from resolvelane.services.support_chat import IdempotencyKeyReused, handle_message
from resolvelane.simulator.fixtures import ORDERS
from resolvelane.simulator.order_source import InMemoryOrderSource

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
CUSTOMER = "demo-customer-101"
# Invented test credentials for fictional customers.
TOKENS = {f"test-token-demo-customer-{n}-xxxxxxxx": {"subject_id": f"demo-customer-{n}", "role": "customer"}
          for n in (101, 102)}


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


# --- storage ---------------------------------------------------------------------

def test_case_survives_reopening_the_database(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    store = CaseStore(path, clock=lambda: NOW)
    case = store.new_case(CUSTOMER)
    case.order_id = "demo-order-1001"
    case.status = CaseStatus.NEEDS_REVIEW
    case.issues = [Issue.LATE_DELIVERY, Issue.MISSING_ITEM]
    case.review_notes = ["Customer reports a missing wrap."]
    citation = Citation("refund-timing", 2, "How refunds work", "arrival")
    case.messages = [CaseMessage(MessageRole.CUSTOMER, "Where is my refund?", NOW),
                     CaseMessage(MessageRole.ASSISTANT, "It takes 3 to 5 business days.", NOW, (citation,))]
    store.commit(case)
    store.close()

    reopened = CaseStore(path).get(case.case_id)
    assert reopened == case


def test_case_is_stored_only_by_commit():
    store = CaseStore()
    case = store.new_case(CUSTOMER)
    assert store.get(case.case_id) is None
    store.commit(case)
    copy = store.get(case.case_id)
    copy.status = CaseStatus.NEEDS_REVIEW
    assert store.get(case.case_id).status is CaseStatus.OPEN


def test_commit_appends_only_new_messages():
    store = CaseStore()
    case = store.new_case(CUSTOMER)
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, "First", NOW))
    store.commit(case)
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, "Second", NOW))
    store.commit(case)
    store.commit(case)
    assert [message.text for message in store.get(case.case_id).messages] == ["First", "Second"]


def test_failed_commit_stores_neither_the_case_nor_the_receipt():
    store = CaseStore(clock=lambda: NOW)
    first = store.new_case(CUSTOMER)
    store.commit(first, Receipt(CUSTOMER, "key-1", "fingerprint", "{}", NOW))
    second = store.new_case(CUSTOMER)
    with pytest.raises(sqlite3.IntegrityError):
        store.commit(second, Receipt(CUSTOMER, "key-1", "other", "{}", NOW))
    assert store.get(second.case_id) is None
    assert store.receipt(CUSTOMER, "key-1").fingerprint == "fingerprint"


def test_reopening_does_not_migrate_again(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    CaseStore(path).close()
    assert CaseStore(path).schema_version == len(MIGRATIONS) == 5


def test_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "cases.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="schema version 99"):
        CaseStore(str(path))


def test_database_location_comes_from_settings(tmp_path):
    path = tmp_path / "nested" / "cases.sqlite3"
    store = load_case_store({"RESOLVELANE_DATABASE": str(path)})
    assert store.path == str(path) and path.exists()


# --- idempotent messages -------------------------------------------------------------

def send(cases, message, case_id=None, order_id=None, key="key-1", customer=CUSTOMER, recognizer=None):
    return handle_message(cases, InMemoryOrderSource(ORDERS), recognizer or KeywordRecognizer(), customer,
                          message, case_id, order_id, idempotency_key=key)


def test_retried_message_is_answered_from_its_receipt():
    cases = CaseStore(clock=lambda: NOW)
    first = send(cases, "One of my wraps is missing.", order_id="demo-order-1001")
    retry = send(cases, "One of my wraps is missing.", order_id="demo-order-1001")
    assert not first.replayed and retry.replayed
    assert (retry.case_id, retry.reply, retry.case_status) == (first.case_id, first.reply, first.case_status)
    assert len(cases.get(first.case_id).messages) == 2


def test_replay_returns_the_original_response_after_the_case_moved_on():
    cases = CaseStore(clock=lambda: NOW)
    first = send(cases, "Where is my order?", order_id="demo-order-1001")
    send(cases, "One of my wraps is missing.", case_id=first.case_id, key="key-2")
    retry = send(cases, "Where is my order?", order_id="demo-order-1001")
    assert retry.case_status is CaseStatus.OPEN
    assert cases.get(first.case_id).status is CaseStatus.NEEDS_REVIEW


def test_key_reused_for_a_different_message_is_rejected():
    cases = CaseStore(clock=lambda: NOW)
    first = send(cases, "Where is my order?", order_id="demo-order-1001")
    with pytest.raises(IdempotencyKeyReused):
        send(cases, "It arrived late.", case_id=first.case_id)
    assert len(cases.get(first.case_id).messages) == 2


def test_keys_are_scoped_to_the_customer():
    cases = CaseStore(clock=lambda: NOW)
    mine = send(cases, "Where is my order?")
    theirs = send(cases, "Where is my order?", customer="demo-customer-102")
    assert not theirs.replayed and theirs.case_id != mine.case_id


def test_messages_without_a_key_are_each_handled():
    cases = CaseStore(clock=lambda: NOW)
    first = send(cases, "Where is my order?", order_id="demo-order-1001", key=None)
    send(cases, "Where is my order?", case_id=first.case_id, key=None)
    send(cases, "Where is my order?", case_id=first.case_id, key=None)
    assert len(cases.get(first.case_id).messages) == 6


def test_expired_receipt_lets_the_message_be_handled_again():
    clock = Clock()
    cases = CaseStore(clock=clock)
    first = send(cases, "Where is my order?")
    clock.now = NOW + CaseStore.RECEIPT_RETENTION + timedelta(seconds=1)
    again = send(cases, "Where is my order?")
    assert not again.replayed and again.case_id != first.case_id


def test_failed_message_leaves_no_receipt():
    class Failing:
        def recognize(self, message, history=()):
            raise RuntimeError("recognizer crashed")

    cases = CaseStore(clock=lambda: NOW)
    with pytest.raises(RuntimeError):
        send(cases, "Where is my order?", recognizer=Failing())
    assert cases.receipt(CUSTOMER, "key-1") is None
    assert not send(cases, "Where is my order?").replayed


def test_duplicates_sent_at_once_are_handled_once():
    class BothAtOnce:
        """Holds each request in recognition until both have passed phase 1."""
        def __init__(self):
            self.barrier = threading.Barrier(2, timeout=5)

        def recognize(self, message, history=()):
            self.barrier.wait()
            return Recognition([Issue.MISSING_ITEM], None, "keywords")

    cases = CaseStore(clock=lambda: NOW)
    first = send(cases, "Where is my order?", order_id="demo-order-1001", key="key-0")
    recognizer, turns = BothAtOnce(), []

    def duplicate():
        turns.append(send(cases, "A wrap is missing.", case_id=first.case_id, recognizer=recognizer))

    threads = [threading.Thread(target=duplicate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(turn.replayed for turn in turns) == [False, True]
    assert turns[0].reply == turns[1].reply
    assert len(cases.get(first.case_id).messages) == 4


# --- over HTTP ----------------------------------------------------------------------

def post(app, body, key=None, customer=101):
    headers = {"Authorization": f"Bearer test-token-demo-customer-{customer}-xxxxxxxx"}
    if key is not None:
        headers["Idempotency-Key"] = key

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/api/v1/chat", json=body, headers=headers)
    return asyncio.run(go())


def test_retry_after_a_restart_is_replayed_over_http(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    body = {"message": "One of my wraps is missing.", "order_id": "demo-order-1001"}
    first = post(create_app(TokenRegistry(TOKENS), case_store=CaseStore(path)), body, key="5b2f-retry")
    restarted = create_app(TokenRegistry(TOKENS), case_store=CaseStore(path))
    retry = post(restarted, body, key="5b2f-retry")
    assert first.status_code == retry.status_code == 200
    assert retry.json() == first.json()
    assert retry.headers["Idempotent-Replayed"] == "true" and "Idempotent-Replayed" not in first.headers


def test_reused_key_with_another_body_is_rejected_over_http():
    app = create_app(TokenRegistry(TOKENS))
    assert post(app, {"message": "Where is my order?"}, key="5b2f-reuse").status_code == 200
    response = post(app, {"message": "Something else"}, key="5b2f-reuse")
    assert response.status_code == 422
    assert "Idempotency-Key" in response.json()["detail"]


@pytest.mark.parametrize("key", ["", "x" * 129, "has space"])
def test_malformed_keys_are_rejected(key):
    assert post(create_app(TokenRegistry(TOKENS)), {"message": "Hello"}, key=key).status_code == 422
