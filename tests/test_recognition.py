import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import InMemoryCaseStore
from resolvelane.domain.case import CaseMessage, Issue, MessageRole
from resolvelane.embeddings import EmbeddingError
from resolvelane.recognition import (DEFAULT_MODEL, SYSTEM_PROMPT, CascadeRecognizer, KeywordRecognizer, KeywordTier,
                                     LLMTier, Recognition, Rung, TierUnavailable, extract_order_id,
                                     load_recognizer)
from resolvelane.services.support_chat import handle_message
from resolvelane.simulator.fixtures import ORDERS
from resolvelane.simulator.order_source import InMemoryOrderSource


class FakeClient:
    """Stands in for anthropic.Anthropic: returns a canned answer or raises an error."""

    def __init__(self, outcome, stop_reason="end_turn"):
        self.outcome = outcome
        self.stop_reason = stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        # Thinking blocks come before the text answer and must be ignored.
        content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.outcome)]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


def recognize(outcome, message="My order was late and a wrap was missing.", history=(), **kwargs):
    client = FakeClient(outcome, **kwargs)
    recognizer = CascadeRecognizer([Rung(LLMTier(client, "test-model")), Rung(KeywordTier())])
    return recognizer.recognize(message, history), client


# --- order ID extraction ---------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("My order demo-order-1001 is late", "demo-order-1001"),
    ("It's DEMO-ORDER-1001", "demo-order-1001"),
    ("demo-order-1001, I mean demo-order-1001", "demo-order-1001"),
    ("Either demo-order-1001 or demo-order-1003", None),
    ("My order is late", None),
])
def test_extract_order_id(message, expected):
    assert extract_order_id(message) == expected


# --- model answers ---------------------------------------------------------

def test_model_labels_are_used_in_a_stable_order():
    result, client = recognize('{"issues": ["missing_item", "late_delivery"], "confidence": 0.9}')
    assert result == Recognition([Issue.LATE_DELIVERY, Issue.MISSING_ITEM], None, "llm")
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["system"] == SYSTEM_PROMPT
    assert "temperature" not in call


def test_json_surrounded_by_text_is_accepted():
    result, _ = recognize('Here you go:\n```json\n{"issues": ["wrong_item"], "confidence": 0.8}\n```')
    assert result.issues == [Issue.WRONG_ITEM]
    assert result.source == "llm"


def test_paraphrase_keywords_miss_is_recognized_by_the_model():
    message = "It never came."
    assert KeywordRecognizer().recognize(message).issues == [Issue.UNKNOWN]
    result, _ = recognize('{"issues": ["missing_item"], "confidence": 0.85}', message=message)
    assert result.issues == [Issue.MISSING_ITEM]


def test_no_applicable_issue_is_unknown():
    result, _ = recognize('{"issues": [], "confidence": 0.9}', message="Thanks!")
    assert result == Recognition([Issue.UNKNOWN], None, "llm")


def test_labels_outside_the_list_are_ignored():
    result, _ = recognize('{"issues": ["refund_now", "late_delivery", 7], "confidence": 0.9}')
    assert result.issues == [Issue.LATE_DELIVERY]


def test_message_and_recent_history_are_sent_as_data():
    sent_at = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)
    history = [CaseMessage(MessageRole.CUSTOMER, f"earlier {n}", sent_at) for n in range(10)]
    _, client = recognize('{"issues": [], "confidence": 0.9}', message="Any update?", history=history)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "<message>\nAny update?\n</message>" in prompt
    assert "customer: earlier 9" in prompt
    assert "earlier 3" not in prompt  # only the last six messages


def test_order_id_is_extracted_by_rule_not_by_the_model():
    result, _ = recognize('{"issues": ["late_delivery"], "confidence": 0.9, "order_id": "demo-order-1003"}',
                          message="demo-order-1001 was late")
    assert result.order_id == "demo-order-1001"


# --- fallback to keywords --------------------------------------------------

def _request():
    return httpx2.Request("POST", "https://api.example.test/v1/messages")


@pytest.mark.parametrize("outcome", [
    "not json at all",
    '{"issues": "late_delivery", "confidence": 0.9}',
    '{"issues": ["late_delivery"]}',
    '{"issues": ["late_delivery"], "confidence": 0.3}',
    '{"issues": ["late_delivery"], "confidence": true}',
    '{"issues": ["late_delivery"], "confidence": 1.5}',
    anthropic.APIConnectionError(request=_request()),
    anthropic.APITimeoutError(request=_request()),
    anthropic.APIStatusError("overloaded", response=httpx2.Response(529, request=_request()), body=None),
    anthropic.APIStatusError("bad request", response=httpx2.Response(400, request=_request()), body=None),
])
def test_unusable_model_answers_fall_back_to_keywords(outcome):
    result, _ = recognize(outcome)
    assert result == Recognition([Issue.LATE_DELIVERY, Issue.MISSING_ITEM], None, "keywords")


def test_refusal_falls_back_to_keywords():
    result, _ = recognize('{"issues": [], "confidence": 0.9}', stop_reason="refusal")
    assert result.source == "keywords"


# --- configuration ---------------------------------------------------------

OFF = {"RESOLVELANE_LOCAL_EMBEDDING_MODEL": "off"}


def test_without_keys_only_keywords_are_used():
    assert load_recognizer(OFF).tier_names == ["keywords"]


def test_key_enables_the_model_with_configured_endpoint():
    recognizer = load_recognizer({**OFF, "ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": "deepseek-flash",
                                  "ANTHROPIC_BASE_URL": "https://llm.example.test"})
    assert recognizer.tier_names == ["llm", "keywords"]
    tier = recognizer.rungs[0].tier
    assert tier.model == "deepseek-flash"
    assert str(tier.client.base_url).startswith("https://llm.example.test")
    assert recognizer.rungs[0].breaker is not None


def test_empty_base_url_setting_uses_the_default_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")  # as left by an empty line in a .env file
    tier = load_recognizer({**OFF, "ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_BASE_URL": ""}).rungs[0].tier
    assert str(tier.client.base_url).startswith("https://api.anthropic.com")


def test_default_model_when_none_is_configured():
    assert load_recognizer({**OFF, "ANTHROPIC_API_KEY": "test-key"}).rungs[0].tier.model == DEFAULT_MODEL


# --- conversation behavior ---------------------------------------------------

class StaticRecognizer:
    def __init__(self, *issues, check=None):
        self.issues = list(issues)
        self.check = check

    def recognize(self, message, history=()):
        if self.check:
            self.check()
        return Recognition(self.issues, extract_order_id(message), "llm")


def chat(recognizer, message, order_id=None, case_id=None):
    app = create_app(token_registry=TokenRegistry({
        "test-token-demo-customer-101-xxxxxxxx": {"subject_id": "demo-customer-101", "role": "customer"}}),
        recognizer=recognizer)
    app.state.case_store = chat.store
    body = {"message": message, **({"order_id": order_id} if order_id else {}),
            **({"case_id": case_id} if case_id else {})}

    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/chat", json=body,
                                     headers={"Authorization": "Bearer test-token-demo-customer-101-xxxxxxxx"})

    response = asyncio.run(send())
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(autouse=True)
def fresh_store():
    chat.store = InMemoryCaseStore()


def test_order_named_in_the_message_is_used_and_bound():
    turn = chat(StaticRecognizer(Issue.MISSING_ITEM), "A wrap is missing from demo-order-1001.")
    assert "2 x Vegetable wrap" in turn["reply"]
    assert turn["order_id"] == "demo-order-1001"


def test_another_order_named_in_a_bound_case_is_redirected():
    first = chat(StaticRecognizer(Issue.DELIVERY_STATUS), "Where is demo-order-1001?")
    second = chat(StaticRecognizer(Issue.LATE_DELIVERY), "And demo-order-1003 was late.", case_id=first["case_id"])
    assert second["reply"] == ("This case is about order demo-order-1001. "
                               "Please start a new case for order demo-order-1003.")
    assert second["order_id"] == "demo-order-1001"


def test_unclassified_follow_up_on_a_case_in_review_only_reports_the_review():
    first = chat(StaticRecognizer(Issue.MISSING_ITEM), "A wrap is missing from demo-order-1001.")
    second = chat(StaticRecognizer(Issue.UNKNOWN), "When will someone get back to me?", case_id=first["case_id"])
    assert second["reply"] == ("Your case about order demo-order-1001 is waiting for a support agent "
                               "to review it. You'll see their reply in this conversation.")
    assert second["case_status"] == "needs_review"


def test_model_call_does_not_hold_the_case_lock():
    acquired = []

    def try_lock_from_another_thread():
        got = chat.store.lock.acquire(timeout=1)
        acquired.append(got)
        if got:
            chat.store.lock.release()

    def other_request_can_proceed():
        worker = threading.Thread(target=try_lock_from_another_thread)
        worker.start()
        worker.join()

    chat(StaticRecognizer(Issue.DELIVERY_STATUS, check=other_request_can_proceed), "Where is demo-order-1001?")
    assert acquired == [True]


def test_case_bound_while_the_model_ran_is_still_checked():
    cases = InMemoryCaseStore()
    orders = InMemoryOrderSource(ORDERS)
    case_id = handle_message(cases, orders, KeywordRecognizer(), "demo-customer-101", "Hello").case.case_id

    def bind_meanwhile():
        cases.get(case_id).order_id = "demo-order-1003"

    from resolvelane.services.support_chat import OrderMismatch
    with pytest.raises(OrderMismatch):
        handle_message(cases, orders, StaticRecognizer(Issue.DELIVERY_STATUS, check=bind_meanwhile),
                       "demo-customer-101", "Where is it?", case_id, "demo-order-1001")


# --- the ladder of tiers ---------------------------------------------------

class ScriptedTier:
    """A tier that answers, abstains (None) or raises, as scripted."""

    def __init__(self, name, outcome):
        self.name = name
        self.outcome = outcome
        self.calls = 0

    def classify(self, message, history=()):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_first_tier_that_answers_wins():
    later = ScriptedTier("local_embedding", [Issue.WRONG_ITEM])
    recognizer = CascadeRecognizer([Rung(ScriptedTier("llm", TierUnavailable("down"))),
                                    Rung(ScriptedTier("remote_embedding", None)),
                                    Rung(ScriptedTier("local_embedding", [Issue.MISSING_ITEM])), Rung(later)])
    result = recognizer.recognize("demo-order-1001 is short a wrap")
    assert result == Recognition([Issue.MISSING_ITEM], "demo-order-1001", "local_embedding")
    assert later.calls == 0


def test_no_answer_from_any_tier_is_unknown():
    recognizer = CascadeRecognizer([Rung(ScriptedTier("llm", TierUnavailable("down"))), Rung(KeywordTier())])
    assert recognizer.recognize("Hmm, about that thing") == Recognition([Issue.UNKNOWN], None, "none")


def test_defect_in_a_tier_does_not_stop_the_ladder(caplog):
    recognizer = CascadeRecognizer([Rung(ScriptedTier("llm", KeyError("bug"))), Rung(KeywordTier())])
    assert recognizer.recognize("It was late").source == "keywords"
    assert any("recognition_tier_error tier=llm" in record.getMessage() for record in caplog.records)


def test_model_outage_opens_the_breaker_and_the_model_is_skipped():
    from resolvelane.resilience import BreakerState, CircuitBreaker
    client = FakeClient(anthropic.APIConnectionError(request=_request()))
    breaker = CircuitBreaker("llm", failure_threshold=3, cooldown_s=30)
    recognizer = CascadeRecognizer([Rung(LLMTier(client, "test-model"), breaker), Rung(KeywordTier())])
    for _ in range(5):
        assert recognizer.recognize("It was late").source == "keywords"
    assert len(client.calls) == 3  # no more waiting for timeouts once the breaker is open
    assert breaker.state is BreakerState.OPEN


def test_unusable_answers_do_not_count_as_an_outage():
    from resolvelane.resilience import BreakerState, CircuitBreaker
    client = FakeClient('{"issues": ["late_delivery"], "confidence": 0.2}')
    breaker = CircuitBreaker("llm", failure_threshold=3, cooldown_s=30)
    recognizer = CascadeRecognizer([Rung(LLMTier(client, "test-model"), breaker), Rung(KeywordTier())])
    for _ in range(5):
        recognizer.recognize("It was late")
    assert len(client.calls) == 5
    assert breaker.state is BreakerState.CLOSED


# --- embedding tier configuration -------------------------------------------

def test_remote_embedding_needs_calibrated_thresholds(caplog):
    recognizer = load_recognizer({**OFF, "RESOLVELANE_EMBEDDING_API_KEY": "test-key"})
    assert recognizer.tier_names == ["keywords"]  # BAAI/bge-m3 has no calibrated thresholds yet
    assert any("no calibrated thresholds for BAAI/bge-m3" in record.getMessage() for record in caplog.records)


def test_remote_embedding_with_configured_thresholds():
    recognizer = load_recognizer({**OFF, "RESOLVELANE_EMBEDDING_API_KEY": "test-key",
                                  "RESOLVELANE_EMBEDDING_BASE_URL": "https://embeddings.example.test/v1",
                                  "RESOLVELANE_EMBEDDING_THRESHOLD": "0.6", "RESOLVELANE_EMBEDDING_MARGIN": "0.04"})
    assert recognizer.tier_names == ["remote_embedding", "keywords"]
    rung = recognizer.rungs[0]
    assert (rung.tier.matcher.threshold, rung.tier.matcher.margin) == (0.6, 0.04)
    assert rung.tier.matcher.embedder.name == "remote/BAAI/bge-m3"
    assert rung.breaker.failure_threshold == 1 and rung.breaker.cooldown_s == 60


def test_local_embedding_tier_is_loaded_at_startup(monkeypatch, tmp_path):
    loaded = []

    class FakeLocalEmbedder:
        def __init__(self, model, cache_dir):
            loaded.append((model, cache_dir))
            self.name = f"local/{model}"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("resolvelane.recognition.LocalEmbedder", FakeLocalEmbedder)
    recognizer = load_recognizer({"RESOLVELANE_LOCAL_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
                                  "RESOLVELANE_MODEL_CACHE_DIR": str(tmp_path)})
    assert recognizer.tier_names == ["local_embedding", "keywords"]
    assert loaded == [("BAAI/bge-small-en-v1.5", str(tmp_path))]
    assert recognizer.rungs[0].tier.matcher._example_vectors is not None  # warmed up


def test_local_embedding_without_fastembed_is_skipped(monkeypatch, caplog):
    def missing(model, cache_dir):
        raise EmbeddingError("fastembed is not installed")

    monkeypatch.setattr("resolvelane.recognition.LocalEmbedder", missing)
    assert load_recognizer({}).tier_names == ["keywords"]
    assert any("fastembed is not installed" in record.getMessage() for record in caplog.records)
