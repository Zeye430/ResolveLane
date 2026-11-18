"""Operational metrics in the Prometheus text format, served at /metrics to operators.

Two kinds:

- Events, counted where they happen: requests, how each reply was written and why the
  agent's reply was not used, tool calls, refund decisions, payment attempts.
- State, read from the case store and the circuit breakers when metrics are scraped: how
  many cases wait for a support agent and for how long, payments still in the outbox, open
  breakers. Read at scrape time, these cannot drift from the records.

Labels hold only small fixed sets of values (routes, tool names, reasons), never case,
customer or order IDs, so the number of series stays bounded and metrics carry no
personal data. Alert rules on these metrics are in ops/prometheus/alerts.yml.
"""

from collections.abc import Iterator
from datetime import datetime

from prometheus_client import CollectorRegistry, Counter, Histogram, disable_created_metrics, generate_latest
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from resolvelane import resilience

disable_created_metrics()  # the *_created series only add noise here
REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter("resolvelane_http_requests", "HTTP requests by route and status code",
                        ["route", "method", "status"], registry=REGISTRY)
HTTP_SECONDS = Histogram("resolvelane_http_request_seconds", "HTTP request duration by route", ["route"],
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60), registry=REGISTRY)
RECOGNITIONS = Counter("resolvelane_issue_recognitions", "Customer messages by the recognition tier that decided",
                       ["source"], registry=REGISTRY)
REPLIES = Counter("resolvelane_replies",
                  "Replies sent, by who wrote them (agent or rules); fallback says why the agent's reply was not "
                  "used, or none", ["source", "fallback"], registry=REGISTRY)
REPLY_REPEATS = Counter("resolvelane_reply_repeats",
                        "Assistant replies nearly identical to the previous assistant reply in the case",
                        ["source"], registry=REGISTRY)
IDEMPOTENT_REPLAYS = Counter("resolvelane_idempotent_replays", "Retried requests answered from their receipt",
                             registry=REGISTRY)
TOOL_CALLS = Counter("resolvelane_tool_calls", "Tool calls by the reply agent, by tool and outcome",
                     ["tool", "outcome"], registry=REGISTRY)
TOOL_SECONDS = Histogram("resolvelane_tool_call_seconds", "Tool call duration", ["tool"],
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10), registry=REGISTRY)
REFUND_EVENTS = Counter("resolvelane_refund_events",
                        "Refund proposals and decisions; author is assistant or support_agent",
                        ["event", "author"], registry=REGISTRY)
REFUNDS_APPROVED_MINOR = Counter("resolvelane_refunds_approved_minor", "Approved refund amounts, in minor units",
                                 ["currency"], registry=REGISTRY)
PAYMENT_ATTEMPTS = Counter("resolvelane_payment_attempts",
                           "Payment attempts by outcome: succeeded, declined, no_answer, or checked (an unconfirmed "
                           "payment looked up without an answer)", ["outcome"], registry=REGISTRY)
LLM_CALLS = Counter("resolvelane_llm_calls", "Language model calls answered, by purpose and model",
                    ["purpose", "model"], registry=REGISTRY)
LLM_TOKENS = Counter("resolvelane_llm_tokens",
                     "Tokens reported by the language model provider, by purpose, model and kind (input, output, "
                     "cache_read, cache_write)", ["purpose", "model", "kind"], registry=REGISTRY)
LLM_COST_USD = Counter("resolvelane_llm_cost_usd",
                       "Estimated language model cost in US dollars from configured prices; models without a "
                       "price are not counted", ["purpose", "model"], registry=REGISTRY)
PROVIDER_SECONDS = Histogram("resolvelane_payment_provider_seconds", "Payment provider call duration",
                             buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30), registry=REGISTRY)


class StateCollector(Collector):
    """State read from the case store and the circuit breakers at scrape time."""

    def __init__(self, cases, now: datetime):
        self.cases = cases
        self.now = now

    def collect(self) -> Iterator[Metric]:
        state = self.cases.operational_state(self.now)
        cases = GaugeMetricFamily("resolvelane_cases", "Cases by status", labels=["status"])
        for status, count in state["cases"].items():
            cases.add_metric([status], count)
        yield cases
        yield GaugeMetricFamily("resolvelane_review_queue_oldest_seconds",
                                "How long the case waiting longest for a support agent has waited",
                                value=state["oldest_review_seconds"])
        payments = GaugeMetricFamily("resolvelane_payments", "Refund payments by status", labels=["status"])
        for status, count in state["payments"].items():
            payments.add_metric([status], count)
        yield payments
        yield GaugeMetricFamily("resolvelane_payment_outbox_oldest_seconds",
                                "Age of the oldest approved refund payment not yet sent or settled",
                                value=state["oldest_pending_payment_seconds"])
        breakers = GaugeMetricFamily("resolvelane_circuit_breaker_open",
                                     "1 while a circuit breaker skips calls to its provider", labels=["breaker"])
        open_by_name: dict[str, int] = {}
        for breaker in resilience.breakers():
            is_open = 0 if breaker.state is resilience.BreakerState.CLOSED else 1
            open_by_name[breaker.name] = max(open_by_name.get(breaker.name, 0), is_open)
        for name, is_open in open_by_name.items():
            breakers.add_metric([name], is_open)
        yield breakers


def exposition(cases, now: datetime) -> bytes:
    state = CollectorRegistry()
    state.register(StateCollector(cases, now))
    return generate_latest(REGISTRY) + generate_latest(state)


def metric_names() -> set[str]:
    """Every series name this application can export, for checking alert rules."""
    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_count", "_sum"), "gauge": ("",)}
    names = set()
    for family in REGISTRY.collect():
        names.update(family.name + suffix for suffix in suffixes.get(family.type, ("",)))
    names.update({"resolvelane_cases", "resolvelane_review_queue_oldest_seconds", "resolvelane_payments",
                  "resolvelane_payment_outbox_oldest_seconds", "resolvelane_circuit_breaker_open"})
    return names
