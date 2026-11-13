import json
from datetime import datetime, timezone

import pytest

from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.tools import SUPPORT_TOOLS, ToolContext, ToolGateway, ToolSpec, _Input

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ORDER = next(order for order in ORDERS if order.order_id == "demo-order-1001")


def context():
    return ToolContext("demo-customer-101", ORDER, NOW, POLICIES)


def run(name, args, ctx=None, gateway=None):
    result = (gateway or ToolGateway()).execute(name, args, ctx or context())
    return json.loads(result.content), result


def test_definitions_use_plain_json_schema():
    definitions = {item["name"]: item for item in ToolGateway().definitions()}
    assert set(definitions) == {"search_help_center", "get_support_policy", "request_human_review"}
    schema = definitions["get_support_policy"]["input_schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["issue"]
    assert "unknown" not in schema["properties"]["issue"]["enum"]
    assert "late_delivery" in schema["properties"]["issue"]["enum"]
    assert "title" not in schema and "$defs" not in schema
    # No tool accepts a customer or order: whose data a tool sees is set by the gateway.
    for item in definitions.values():
        assert not {"customer_id", "order_id"} & set(item["input_schema"]["properties"])


def test_policy_in_effect_is_returned():
    payload, result = run("get_support_policy", {"issue": "missing_item"})
    assert not result.is_error
    assert payload["policy"]["title"] == "Missing item review"
    assert "support agent must review any refund" in payload["policy"]["guidance"]


def test_expired_policy_is_not_returned():
    payload, result = run("get_support_policy", {"issue": "damaged_item"})
    assert not result.is_error
    assert payload["policy"] is None
    assert "not in effect. A support agent must review" in payload["note"]


def test_issue_without_any_policy_does_not_ask_for_review():
    payload, _ = run("get_support_policy", {"issue": "delivery_status"})
    assert payload["policy"] is None
    assert "review" not in payload["note"]


@pytest.mark.parametrize("args,expected", [
    ({"issue": "free_food_forever"}, "issue"),
    ({"issue": "missing_item", "customer_id": "demo-customer-102"}, "customer_id"),
    ({}, "issue"),
    ("missing_item", "input"),
])
def test_invalid_input_is_an_error_result_without_echoing_values(args, expected):
    payload, result = run("get_support_policy", args)
    assert result.is_error and result.trace.error == "invalid_input"
    assert expected in payload["error"]
    assert "free_food_forever" not in payload["error"] and "demo-customer-102" not in payload["error"]


def test_unknown_tool_is_an_error_result():
    payload, result = run("issue_refund", {"amount": 100})
    assert result.is_error and result.trace.error == "unknown_tool"


def test_review_request_is_recorded_once_and_not_applied():
    ctx = context()
    run("request_human_review", {"reason": "Customer asked for a person"}, ctx)
    payload, result = run("request_human_review", {"reason": "Asked again"}, ctx)
    assert payload["status"] == "review_requested"
    assert [effect.reason for effect in ctx.effects] == ["Customer asked for a person"]


def test_review_reason_length_is_limited():
    _, result = run("request_human_review", {"reason": "x" * 301})
    assert result.is_error


def test_failing_tool_is_an_error_result(caplog):
    class NoInput(_Input):
        pass

    def broken(ctx, args):
        raise RuntimeError("database password is hunter2")

    payload, result = run("broken", {}, gateway=ToolGateway([ToolSpec("broken", "Fails", NoInput, broken)]))
    assert result.is_error and result.trace.error == "tool_failed"
    assert payload == {"error": "The tool failed"}


def test_support_tools_are_the_default():
    assert list(ToolGateway().specs) == [spec.name for spec in SUPPORT_TOOLS]


# --- help center search ------------------------------------------------------

def test_search_returns_citable_passages_and_records_them():
    from resolvelane.knowledge import HelpCenter
    from resolvelane.simulator.help_center import ARTICLES
    ctx = ToolContext("demo-customer-101", ORDER, NOW, POLICIES, HelpCenter(ARTICLES))
    payload, result = run("search_help_center", {"query": "refund arrival time"}, ctx)
    assert not result.is_error
    first = payload["passages"][0]
    assert first["source_id"] == "refund-timing@2#arrival"
    assert "3 to 5 business days" in first["text"]
    assert set(ctx.retrieved) == {passage["source_id"] for passage in payload["passages"]}


def test_search_with_no_match_says_not_to_guess():
    from resolvelane.knowledge import HelpCenter
    from resolvelane.simulator.help_center import ARTICLES
    ctx = ToolContext("demo-customer-101", ORDER, NOW, POLICIES, HelpCenter(ARTICLES))
    payload, _ = run("search_help_center", {"query": "pay with cash"}, ctx)
    assert payload["passages"] == [] and "Do not answer from memory" in payload["note"]
    assert ctx.retrieved == {}


def test_search_without_a_help_center():
    payload, result = run("search_help_center", {"query": "refund arrival"})
    assert payload["passages"] == [] and not result.is_error
