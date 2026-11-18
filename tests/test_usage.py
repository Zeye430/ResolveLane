from types import SimpleNamespace

import pytest

from resolvelane import llm, metrics
from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.llm import PriceConfigError, Usage, create_message, load_prices, metered, usage_of


def response(input_tokens=0, output_tokens=0, **extra):
    return SimpleNamespace(usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, **extra),
                           content=[], stop_reason="end_turn")


class FakeClient:
    def __init__(self, answer):
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)
        self.answer = answer

    def _create(self, **request):
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def prices():
    yield lambda text: load_prices({"RESOLVELANE_MODEL_PRICES": text})
    load_prices({})


def sample(name, labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_usage_is_read_from_the_response_even_without_cache_fields():
    assert usage_of(response(120, 30)) == Usage(120, 30)
    assert usage_of(response(120, 30, cache_read_input_tokens=1000, cache_creation_input_tokens=None)) \
        == Usage(120, 30, cache_read_tokens=1000)
    assert usage_of(SimpleNamespace()) == Usage()  # a compatible provider that reports nothing


def test_cost_comes_only_from_configured_prices(prices):
    prices('{"model-a": {"input": 2.0, "output": 10.0, "cache_read": 0.2}}')
    assert llm.cost_of("model-a", Usage(1_000_000, 100_000, cache_read_tokens=500_000)) == pytest.approx(3.1)
    assert llm.cost_of("model-a", Usage(0, 0, cache_write_tokens=1_000_000)) == pytest.approx(2.0)  # input price
    assert llm.cost_of("model-b", Usage(1000, 1000)) is None  # never guessed


@pytest.mark.parametrize("text", ['["model-a"]', '{"model-a": {"input": 1.0}}', '{"model-a": {"input": -1, "output": 1}}',
                                  '{"model-a": {"input": 1, "output": 1, "batch": 1}}', "not json"])
def test_invalid_prices_stop_start_up(prices, text, monkeypatch):
    with pytest.raises(PriceConfigError):
        prices(text)
    monkeypatch.setenv("RESOLVELANE_MODEL_PRICES", text)
    with pytest.raises(PriceConfigError):
        create_app(TokenRegistry({}))


def test_a_meter_collects_every_call_made_while_open_and_metrics_count_them(prices):
    prices('{"model-a": {"input": 1.0, "output": 5.0}}')
    client = FakeClient(response(1000, 200))
    tokens_before = sample("resolvelane_llm_tokens_total", {"purpose": "reply", "model": "model-a", "kind": "input"})
    cost_before = sample("resolvelane_llm_cost_usd_total", {"purpose": "reply", "model": "model-a"})
    create_message(client, "reply", model="model-a", max_tokens=10, messages=[])  # outside any meter
    with metered() as meter:
        create_message(client, "recognition", model="model-a", max_tokens=10, messages=[])
        create_message(client, "reply", model="model-a", max_tokens=10, messages=[])
        create_message(client, "reply", model="model-b", max_tokens=10, messages=[])
    assert client.requests[0] == {"model": "model-a", "max_tokens": 10, "messages": []}
    assert [call.purpose for call in meter.calls] == ["recognition", "reply", "reply"]
    assert meter.usage() == Usage(3000, 600)
    assert meter.cost() is None  # model-b has no price, so the total is unknown
    assert sample("resolvelane_llm_tokens_total", {"purpose": "reply", "model": "model-a", "kind": "input"}) \
        == tokens_before + 2000
    assert sample("resolvelane_llm_cost_usd_total", {"purpose": "reply", "model": "model-a"}) \
        == pytest.approx(cost_before + 0.004)


def test_a_failed_call_records_nothing():
    client = FakeClient(TimeoutError("no answer"))
    with metered() as meter, pytest.raises(TimeoutError):
        create_message(client, "reply", model="model-a", messages=[])
    assert meter.calls == []
