import asyncio

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.assistant import Issue, detect_issues
from resolvelane.simulator.fixtures import SCENARIOS

# Invented test credentials: one token per fictional customer.
TOKENS = {f"test-token-demo-customer-{n}-xxxxxxxx": {"subject_id": f"demo-customer-{n}", "role": "customer"}
          for n in (101, 102, 103, 104)}
TOKEN_BY_CUSTOMER = {account["subject_id"]: token for token, account in TOKENS.items()}


def chat(customer_id, message, order_id=None):
    app = create_app(token_registry=TokenRegistry(TOKENS))
    body = {"message": message} if order_id is None else {"message": message, "order_id": order_id}

    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/chat", json=body,
                                     headers={"Authorization": f"Bearer {TOKEN_BY_CUSTOMER[customer_id]}"})

    response = asyncio.run(send())
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("message,expected", [
    ("Where is my order?", [Issue.DELIVERY_STATUS]),
    ("My order arrived late.", [Issue.LATE_DELIVERY]),
    ("One wrap is missing.", [Issue.MISSING_ITEM]),
    ("I received the wrong drink.", [Issue.WRONG_ITEM]),
    ("My drink arrived damaged.", [Issue.DAMAGED_ITEM]),
    ("Why was my order cancelled?", [Issue.CANCELLATION]),
    ("It was late and one wrap was missing.", [Issue.LATE_DELIVERY, Issue.MISSING_ITEM]),
    ("I ordered a chocolate cake.", [Issue.UNKNOWN]),
])
def test_detect_issues(message, expected):
    assert detect_issues(message) == expected


# Expected detected issues for each scenario's issue_type.
SCENARIO_ISSUES = {
    "multiple_issues": [Issue.LATE_DELIVERY, Issue.MISSING_ITEM],
    "order_access": [Issue.UNKNOWN],
    "order_lookup": [Issue.UNKNOWN],
}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.scenario_id)
def test_every_scenario_gets_a_reply(scenario):
    body = chat(scenario.customer_id, scenario.message, scenario.order_id)
    assert body["issues"] == SCENARIO_ISSUES.get(scenario.issue_type, [scenario.issue_type])
    assert body["reply"]


def test_late_delivery_reply_uses_order_facts():
    body = chat("demo-customer-101", "My order arrived late.", "demo-order-1001")
    assert "15 minutes after the estimated time" in body["reply"]
    assert body["needs_human_review"] is True


def test_on_time_question_is_answered_with_the_estimate():
    body = chat("demo-customer-103", "Was my order delivered on time?", "demo-order-1003")
    assert "delivered at 12:40 UTC on Jan 16, within the estimated time of 12:45 UTC on Jan 16" in body["reply"]


def test_late_claim_on_an_on_time_order_is_not_simply_accepted():
    body = chat("demo-customer-103", "My order arrived late.", "demo-order-1003")
    assert "within the estimated time" in body["reply"]
    assert body["needs_human_review"] is False


def test_missing_item_reply_lists_order_contents():
    body = chat("demo-customer-101", "One of my wraps is missing.", "demo-order-1001")
    assert "missing item" in body["reply"]
    assert "2 x Vegetable wrap, 1 x Sparkling water" in body["reply"]
    assert body["needs_human_review"] is True


def test_in_progress_order_reports_latest_update():
    body = chat("demo-customer-102", "Where is my order?", "demo-order-1002")
    assert "on its way" in body["reply"]
    assert "courier assigned at 18:42 UTC" in body["reply"]
    assert body["needs_human_review"] is False


def test_item_report_before_delivery_is_not_sent_for_review():
    body = chat("demo-customer-102", "My rice bowl is missing.", "demo-order-1002")
    assert "has not been delivered yet" in body["reply"]
    assert body["needs_human_review"] is False


def test_cancelled_order_without_reason_goes_to_review():
    body = chat("demo-customer-104", "Why was my order cancelled?", "demo-order-1004")
    assert "cancelled at 14:08 UTC" in body["reply"]
    assert body["needs_human_review"] is True


def test_missing_order_number_is_requested():
    body = chat("demo-customer-101", "One of my wraps is missing.")
    assert "order number" in body["reply"]
    assert body["order_id"] is None


def test_another_customers_order_reads_like_an_unknown_order():
    denied = chat("demo-customer-102", "Where is my order?", "demo-order-1001")
    unknown = chat("demo-customer-102", "Where is my order?", "demo-order-9999")
    assert "couldn't find order demo-order-1001" in denied["reply"]
    assert denied["reply"].replace("1001", "9999") == unknown["reply"]
    assert "Vegetable wrap" not in denied["reply"]
