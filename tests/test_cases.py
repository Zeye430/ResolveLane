import asyncio
import logging
import threading

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import InMemoryCaseStore
from resolvelane.services.support_chat import handle_message
from resolvelane.simulator.fixtures import ORDERS
from resolvelane.simulator.order_source import InMemoryOrderSource

# Invented test credentials for fictional customers.
TOKENS = {f"test-token-demo-customer-{n}-xxxxxxxx": {"subject_id": f"demo-customer-{n}", "role": "customer"}
          for n in (101, 102)}


class Api:
    """One application instance, so cases persist across requests within a test."""

    def __init__(self):
        self.app = create_app(token_registry=TokenRegistry(TOKENS))

    def _send(self, method, path, customer, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs, headers={
                    "Authorization": f"Bearer test-token-demo-customer-{customer}-xxxxxxxx"})
        return asyncio.run(send())

    def chat(self, customer, message, order_id=None, case_id=None, expect=200):
        body = {"message": message}
        if order_id is not None:
            body["order_id"] = order_id
        if case_id is not None:
            body["case_id"] = case_id
        response = self._send("POST", "/api/v1/chat", customer, json=body)
        assert response.status_code == expect, response.text
        return response.json()

    def case(self, customer, case_id, expect=200):
        response = self._send("GET", f"/api/v1/cases/{case_id}", customer)
        assert response.status_code == expect, response.text
        return response.json()


@pytest.fixture
def api():
    return Api()


def test_first_message_opens_a_case_with_the_conversation(api):
    turn = api.chat(101, "Where is my order?", "demo-order-1001")
    assert turn["case_id"].startswith("case-")
    assert turn["case_status"] == "open"

    case = api.case(101, turn["case_id"])
    assert case["order_id"] == "demo-order-1001"
    assert [message["role"] for message in case["messages"]] == ["customer", "assistant"]
    assert case["messages"][0]["text"] == "Where is my order?"
    assert case["messages"][1]["text"] == turn["reply"]
    assert "pending_issues" not in case


def test_issue_raised_before_the_order_number_is_answered_when_it_arrives(api):
    first = api.chat(101, "One of my wraps is missing.")
    assert "order number" in first["reply"]
    assert first["order_id"] is None

    second = api.chat(101, "It's this order.", "demo-order-1001", case_id=first["case_id"])
    assert second["issues"] == ["missing_item"]
    assert "missing item" in second["reply"]
    assert second["case_status"] == "needs_review"
    assert second["order_id"] == "demo-order-1001"


def test_pending_and_new_issues_are_answered_together(api):
    first = api.chat(101, "One of my wraps is missing.")
    second = api.chat(101, "It was also late.", "demo-order-1001", case_id=first["case_id"])
    assert second["issues"] == ["late_delivery", "missing_item"]
    assert "15 minutes after" in second["reply"]
    assert "missing item" in second["reply"]


def test_follow_up_reuses_the_case_order(api):
    first = api.chat(101, "Where is my order?", "demo-order-1001")
    second = api.chat(101, "It arrived late.", case_id=first["case_id"])
    assert "15 minutes after" in second["reply"]
    assert second["issues"] == ["late_delivery"]


def test_answered_issues_are_not_repeated(api):
    first = api.chat(101, "One of my wraps is missing.", "demo-order-1001")
    second = api.chat(101, "Where is my order?", case_id=first["case_id"])
    assert second["issues"] == ["delivery_status"]
    assert "missing" not in second["reply"]
    assert api.case(101, first["case_id"])["issues"] == ["missing_item", "delivery_status"]


def test_review_status_is_kept_after_later_messages(api):
    first = api.chat(101, "One of my wraps is missing.", "demo-order-1001")
    second = api.chat(101, "Where is my order?", case_id=first["case_id"])
    assert second["needs_human_review"] is False
    assert second["case_status"] == "needs_review"


def test_follow_up_on_a_case_in_review_mentions_the_review(api):
    first = api.chat(101, "One of my wraps is missing.", "demo-order-1001")
    second = api.chat(101, "When will someone get back to me?", case_id=first["case_id"])
    assert second["reply"].endswith("Your case is waiting for a support agent to review it.")
    assert api.case(101, first["case_id"])["messages"][-1]["text"] == second["reply"]


def test_open_case_follow_up_does_not_mention_a_review(api):
    first = api.chat(101, "Where is my order?", "demo-order-1001")
    second = api.chat(101, "When will it arrive?", case_id=first["case_id"])
    assert "waiting for a support agent" not in second["reply"]


def test_case_cannot_switch_to_another_order(api):
    first = api.chat(101, "Where is my order?", "demo-order-1001")
    response = api.chat(101, "And this one?", "demo-order-1003", case_id=first["case_id"], expect=409)
    assert "demo-order-1001" in response["detail"]
    assert len(api.case(101, first["case_id"])["messages"]) == 2


def test_inaccessible_order_is_not_bound_to_the_case(api):
    turn = api.chat(102, "Where is my order?", "demo-order-1001")
    assert "couldn't find" in turn["reply"]
    assert api.case(102, turn["case_id"])["order_id"] is None
    retry = api.chat(102, "Sorry, this one.", "demo-order-1002", case_id=turn["case_id"])
    assert retry["order_id"] == "demo-order-1002"


def test_another_customers_case_looks_like_an_unknown_case(api):
    case_id = api.chat(101, "Where is my order?", "demo-order-1001")["case_id"]
    other = api.chat(102, "Hello", case_id=case_id, expect=404)
    unknown = api.chat(102, "Hello", case_id="case-does-not-exist", expect=404)
    assert other == unknown == {"detail": "Case not found"}
    assert api.case(102, case_id, expect=404) == {"detail": "Case not found"}
    assert len(api.case(101, case_id)["messages"]) == 2


def test_lookup_outcome_and_review_are_logged(api, caplog):
    with caplog.at_level(logging.INFO, logger="resolvelane"):
        api.chat(102, "Where is my order?", "demo-order-1001")
        api.chat(101, "One of my wraps is missing.", "demo-order-1001")
    messages = [record.getMessage() for record in caplog.records]
    assert any("access=denied" in message for message in messages)
    assert any("status=needs_review" in message and "missing_item" in message for message in messages)


def test_concurrent_messages_to_one_case_are_all_kept():
    cases = InMemoryCaseStore()
    orders = InMemoryOrderSource(ORDERS)
    case_id = handle_message(cases, orders, "demo-customer-101", "Where is my order?", None, "demo-order-1001").case.case_id

    def send(n):
        handle_message(cases, orders, "demo-customer-101", f"Message {n}", case_id)

    threads = [threading.Thread(target=send, args=(n,)) for n in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(cases.get(case_id).messages) == 2 + 20 * 2
