"""The language model provider shared by issue recognition and reply writing.

Both callers share one circuit breaker: once recognition finds the provider down, reply
writing skips it too instead of making the same customer wait for a second timeout.

Every model call goes through ``create_message``, which records the tokens the provider
reports: in metrics, by purpose and model, and in the meter of the reply being written, if any
(``metered``). A meter collects every call made while it is open, including calls whose answer
was not used: a recognition tier that abstained, or an agent draft that a guard replaced.
Costs are estimated from prices in configuration (RESOLVELANE_MODEL_PRICES); a model without a
configured price is counted in tokens only, never priced by guess.
"""

import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import anthropic

from resolvelane import metrics
from resolvelane.resilience import CircuitBreaker

logger = logging.getLogger("resolvelane.llm")

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
# Errors caused by the request rather than the provider; they do not count against the breaker.
REQUEST_ERRORS = (400, 413, 422)


@dataclass(frozen=True)
class LLMProvider:
    client: anthropic.Anthropic
    model: str
    breaker: CircuitBreaker


def load_llm_provider(environ: Mapping[str, str] = os.environ) -> LLMProvider | None:
    api_key = environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.info("ANTHROPIC_API_KEY is not set; the language model is off")
        return None
    client = anthropic.Anthropic(
        api_key=api_key,
        # Passed explicitly: given None, the SDK reads ANTHROPIC_BASE_URL itself and would use an
        # empty value from a .env file as the URL.
        base_url=environ.get("ANTHROPIC_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout=float(environ.get("RESOLVELANE_LLM_TIMEOUT_S", "10")),
        max_retries=1,
    )
    return LLMProvider(client, environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL,
                       CircuitBreaker("llm", failure_threshold=3, cooldown_s=30))


# --- token usage and cost ---------------------------------------------------------------------

PURPOSES = ("recognition", "reply", "summary", "judge")
PRICE_KINDS = ("input", "output", "cache_read", "cache_write")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.cache_read_tokens + other.cache_read_tokens,
                     self.cache_write_tokens + other.cache_write_tokens)


def usage_of(response) -> Usage:
    """The tokens a response reports; compatible providers may leave the cache fields out."""
    usage = getattr(response, "usage", None)

    def count(name: str) -> int:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) and value > 0 else 0

    return Usage(count("input_tokens"), count("output_tokens"), count("cache_read_input_tokens"),
                 count("cache_creation_input_tokens"))


class PriceConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""
    input: float
    output: float
    cache_read: float | None = None   # defaults to the input price
    cache_write: float | None = None  # defaults to the input price

    def cost(self, usage: Usage) -> float:
        return (usage.input_tokens * self.input + usage.output_tokens * self.output
                + usage.cache_read_tokens * (self.input if self.cache_read is None else self.cache_read)
                + usage.cache_write_tokens * (self.input if self.cache_write is None else self.cache_write)) / 1e6


_prices: dict[str, Price] = {}


def load_prices(environ: Mapping[str, str] = os.environ) -> dict[str, Price]:
    """RESOLVELANE_MODEL_PRICES, e.g. {"model-name": {"input": 1.0, "output": 5.0}}, in US dollars
    per million tokens from the provider's price list. Invalid configuration stops start-up."""
    global _prices
    raw = environ.get("RESOLVELANE_MODEL_PRICES", "").strip()
    prices: dict[str, Price] = {}
    if raw:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise TypeError
            for model, entry in data.items():
                if not isinstance(entry, dict) or set(entry) - set(PRICE_KINDS) or not {"input", "output"} <= set(entry):
                    raise TypeError
                values = {kind: float(value) for kind, value in entry.items()}
                if any(value < 0 for value in values.values()):
                    raise ValueError
                prices[model] = Price(**values)
        except (ValueError, TypeError):
            raise PriceConfigError('RESOLVELANE_MODEL_PRICES must be a JSON object like {"model": {"input": 1.0, '
                                   '"output": 5.0}}, in US dollars per million tokens') from None
    _prices = prices
    if prices:
        logger.info("model prices configured for %s", ", ".join(sorted(prices)))
    return prices


def cost_of(model: str, usage: Usage) -> float | None:
    price = _prices.get(model)
    return None if price is None else price.cost(usage)


@dataclass
class Call:
    purpose: str
    model: str
    usage: Usage


@dataclass
class Meter:
    """The model calls made while it was open."""
    calls: list[Call] = field(default_factory=list)

    def usage(self) -> Usage:
        return sum((call.usage for call in self.calls), Usage())

    def cost(self) -> float | None:
        """Estimated cost, or None when any call's model has no configured price."""
        costs = [cost_of(call.model, call.usage) for call in self.calls]
        return None if None in costs else sum(costs)


_meter: ContextVar[Meter | None] = ContextVar("resolvelane_llm_meter", default=None)


@contextmanager
def metered() -> Iterator[Meter]:
    meter = Meter()
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        _meter.reset(token)


def record(purpose: str, model: str, usage: Usage) -> None:
    for kind, tokens in (("input", usage.input_tokens), ("output", usage.output_tokens),
                         ("cache_read", usage.cache_read_tokens), ("cache_write", usage.cache_write_tokens)):
        if tokens:
            metrics.LLM_TOKENS.labels(purpose, model, kind).inc(tokens)
    metrics.LLM_CALLS.labels(purpose, model).inc()
    cost = cost_of(model, usage)
    if cost is not None:
        metrics.LLM_COST_USD.labels(purpose, model).inc(cost)
    meter = _meter.get()
    if meter is not None:
        meter.calls.append(Call(purpose, model, usage))


def create_message(client: anthropic.Anthropic, purpose: str, **request):
    """client.messages.create, recording the tokens of the answer. A call that fails gets no answer,
    so nothing is recorded; the provider's bill is the authority on those."""
    response = client.messages.create(**request)
    record(purpose, request["model"], usage_of(response))
    return response
