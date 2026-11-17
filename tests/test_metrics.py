import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from resolvelane import metrics
from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import MessageRole
from resolvelane.payments import send_due
from resolvelane.refunds import RefundProposal
from resolvelane.reply_agent import AgentDraft, AgentFailed
from resolvelane.resilience import CircuitBreaker
from resolvelane.services import support_desk
from resolvelane.simulator.fixtures import ORDERS
from resolvelane.simulator.order_source import InMemoryOrderSource
from resolvelane.simulator.payment_provider import SimulatedPaymentProvider

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ALERTS = Path(__file__).resolve().parent.parent / "ops" / "prometheus" / "alerts.yml"
# Invented test credentials for fictional principals.
TOKENS = {"test-token-demo-customer-101-xxxxxxxx": {"subject_id": "demo-customer-101", "role": "customer"},
          "test-token-demo-agent-301-xxxxxxxxxxx": {"subject_id": "demo-agent-301", "role": "support_agent"},
          "test-token-demo-operator-401-xxxxxxxx": {"subject_id": "demo-operator-401", "role": "operator"}}
CUSTOMER, AGENT, OPERATOR = TOKENS


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


def value(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


class Service:
    def __init__(self, reply_agent=None):
        self.clock = Clock()
        self.store = CaseStore(clock=self.clock)
        self.provider = SimulatedPaymentProvider()
        self.app = create_app(TokenRegistry(TOKENS), case_store=self.store, reply_agent=reply_agent,
                              payment_provider=self.provider)

    def request(self, method, path, token, expect=200, **kwargs):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://testserver") as client:
                return await client.request(method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs)
        response = asyncio.run(go())
        assert response.status_code == expect, response.text
        return response

    def chat(self, message, case_id=None):
        body = {"message": message, **({"case_id": case_id} if case_id else {"order_id": "demo-order-1001"})}
        return self.request("POST", "/api/v1/chat", CUSTOMER, json=body).json()

    def scrape(self):
        return self.request("GET", "/metrics", OPERATOR).text


def gauge(text, name, **labels):
    selector = ",".join(f'{key}="{item}"' for key, item in labels.items())
    match = re.search(rf"^{name}{{{selector}}} (\S+)$" if labels else rf"^{name} (\S+)$", text, re.MULTILINE)
    assert match, f"{name} {labels} not exported"
    return float(match.group(1))


# --- the endpoint -------------------------------------------------------------------------

def test_metrics_are_for_operators_only():
    service = Service()
    service.request("GET", "/metrics", CUSTOMER, expect=403)
    service.request("GET", "/metrics", AGENT, expect=403)
    async def anonymous():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                     base_url="http://testserver") as client:
            return await client.get("/metrics")
    assert asyncio.run(anonymous()).status_code == 401
    response = service.request("GET", "/metrics", OPERATOR)
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE resolvelane_replies_total counter" in response.text
    assert "_created" not in response.text


def test_operators_see_no_cases():
    service = Service()
    case_id = service.chat("One of my wraps is missing.")["case_id"]
    service.request("GET", f"/api/v1/support/cases/{case_id}", OPERATOR, expect=403)
    service.request("GET", f"/api/v1/cases/{case_id}", OPERATOR, expect=403)


def test_metrics_name_routes_not_ids():
    service = Service()
    case_id = service.chat("One of my wraps is missing.")["case_id"]
    before = value("resolvelane_http_requests_total", route="/api/v1/support/cases/{case_id}", method="GET",
                   status="200")
    service.request("GET", f"/api/v1/support/cases/{case_id}", AGENT)
    text = service.scrape()
    assert value("resolvelane_http_requests_total", route="/api/v1/support/cases/{case_id}", method="GET",
                 status="200") == before + 1
    assert "case-" not in text and "demo-customer" not in text and "demo-order" not in text


# --- events -------------------------------------------------------------------------------

def test_reply_source_and_recognition_are_counted():
    service = Service()
    rules = value("resolvelane_replies_total", source="rules", fallback="none")
    keywords = value("resolvelane_issue_recognitions_total", source="keywords")
    service.chat("My order arrived late.")
    assert value("resolvelane_replies_total", source="rules", fallback="none") == rules + 1
    assert value("resolvelane_issue_recognitions_total", source="keywords") == keywords + 1


class FailingAgent:
    def draft(self, request):
        raise AgentFailed("ungrounded_number")


def test_why_the_agent_reply_was_not_used_is_counted():
    service = Service(reply_agent=FailingAgent())
    before = value("resolvelane_replies_total", source="rules", fallback="ungrounded_number")
    service.chat("My order arrived late.")
    assert value("resolvelane_replies_total", source="rules", fallback="ungrounded_number") == before + 1


class SameReplyAgent:
    def draft(self, request):
        return AgentDraft("Your order was delivered 15 minutes late. A support agent will review it.", None, [], 1)


def test_repeated_replies_are_counted():
    service = Service(reply_agent=SameReplyAgent())
    before = value("resolvelane_reply_repeats_total", source="agent")
    case_id = service.chat("My order arrived late.")["case_id"]
    assert value("resolvelane_reply_repeats_total", source="agent") == before
    service.chat("Hello? Any news?", case_id)
    assert value("resolvelane_reply_repeats_total", source="agent") == before + 1


def test_tool_calls_are_counted_without_model_chosen_names():
    from resolvelane.simulator.fixtures import POLICIES
    from resolvelane.tools import ToolContext, ToolGateway
    order = next(order for order in ORDERS if order.order_id == "demo-order-1001")
    context = ToolContext("demo-customer-101", order, NOW, POLICIES)
    ok = value("resolvelane_tool_calls_total", tool="get_support_policy", outcome="ok")
    unknown = value("resolvelane_tool_calls_total", tool="unknown", outcome="unknown_tool")
    ToolGateway().execute("get_support_policy", {"issue": "late_delivery"}, context)
    ToolGateway().execute("drop_all_tables_please", {}, context)
    assert value("resolvelane_tool_calls_total", tool="get_support_policy", outcome="ok") == ok + 1
    assert value("resolvelane_tool_calls_total", tool="unknown", outcome="unknown_tool") == unknown + 1
    assert value("resolvelane_tool_call_seconds_count", tool="get_support_policy") >= 1


# --- state at scrape time ------------------------------------------------------------------

def test_review_queue_and_outbox_are_read_from_the_records():
    service = Service()
    case_id = service.chat("One of my wraps is missing.")["case_id"]
    service.clock.now += timedelta(minutes=30)
    text = service.scrape()
    assert gauge(text, "resolvelane_cases", status="needs_review") == 1
    assert gauge(text, "resolvelane_review_queue_oldest_seconds") == 1800

    orders = InMemoryOrderSource(ORDERS)
    approved = value("resolvelane_refund_events_total", event="approved", author="support_agent")
    amount = value("resolvelane_refunds_approved_minor_total", currency="USD")
    refund = support_desk.propose_refund(service.store, orders, "demo-agent-301", case_id,
                                         RefundProposal((("item-1", 1),), False, "Missing wrap"))
    support_desk.approve_refund(service.store, orders, "demo-agent-301", refund.refund_id, 1)
    assert value("resolvelane_refund_events_total", event="approved", author="support_agent") == approved + 1
    assert value("resolvelane_refunds_approved_minor_total", currency="USD") == amount + 950
    service.clock.now += timedelta(minutes=5)
    text = service.scrape()
    assert gauge(text, "resolvelane_payments", status="pending") == 1
    assert gauge(text, "resolvelane_payment_outbox_oldest_seconds") == 300

    succeeded = value("resolvelane_payment_attempts_total", outcome="succeeded")
    send_due(service.store, service.provider)
    text = service.scrape()
    assert gauge(text, "resolvelane_payments", status="succeeded") == 1
    assert gauge(text, "resolvelane_payment_outbox_oldest_seconds") == 0
    assert value("resolvelane_payment_attempts_total", outcome="succeeded") == succeeded + 1
    assert service.store.get(case_id).messages[-1].role is MessageRole.SYSTEM


def test_open_circuit_breaker_is_exported():
    breaker = CircuitBreaker("demo-provider", failure_threshold=1, cooldown_s=30)
    service = Service()
    assert gauge(service.scrape(), "resolvelane_circuit_breaker_open", breaker="demo-provider") == 0
    breaker.failed(breaker.acquire())
    assert gauge(service.scrape(), "resolvelane_circuit_breaker_open", breaker="demo-provider") == 1


# --- alert rules ----------------------------------------------------------------------------

def test_alert_rules_use_only_exported_metrics():
    yaml = pytest.importorskip("yaml")
    rules = [rule for group in yaml.safe_load(ALERTS.read_text())["groups"] for rule in group["rules"]]
    assert len({rule["alert"] for rule in rules}) == len(rules)
    exported = metrics.metric_names()
    for rule in rules:
        assert rule["labels"]["severity"] in ("page", "ticket"), rule["alert"]
        assert rule["annotations"]["summary"], rule["alert"]
        used = set(re.findall(r"\bresolvelane_[a-z_]+", rule["expr"]))
        assert used and used <= exported, (rule["alert"], used - exported)


def test_state_metric_names_match_the_exposition():
    service = Service()
    exported = set(re.findall(r"^# TYPE (resolvelane_[a-z_]+)", service.scrape(), re.MULTILINE))
    state = {name for name in metrics.metric_names() if not name.endswith(("_total", "_bucket", "_count", "_sum"))}
    assert state <= exported
