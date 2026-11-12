import json
import math

import httpx
import pytest

from resolvelane.domain.case import Issue
from resolvelane.embeddings import EmbeddingError, EmbeddingMatcher, RemoteEmbedder, normalize

# --- remote embedder -------------------------------------------------------

KEY = "test-embedding-key-not-real"


def remote(handler):
    return RemoteEmbedder("https://embeddings.example.test/v1", KEY, "test-model",
                          transport=httpx.MockTransport(handler))


def rows(*vectors, order=None):
    indexes = order or range(len(vectors))
    return {"data": [{"index": i, "embedding": vectors[i]} for i in indexes]}


def test_remote_request_and_unit_vectors_in_input_order():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=rows([3, 4], [0, 2], order=[1, 0]))

    vectors = remote(handler).embed(["first", "second"])
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]
    request = seen[0]
    assert str(request.url) == "https://embeddings.example.test/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {"model": "test-model", "input": ["first", "second"],
                                           "encoding_format": "float"}


def test_remote_requests_are_batched():
    sizes = []

    def handler(request):
        texts = json.loads(request.content)["input"]
        sizes.append(len(texts))
        return httpx.Response(200, json=rows(*[[1, 0]] * len(texts)))

    assert len(remote(handler).embed([f"text {n}" for n in range(70)])) == 70
    assert sizes == [32, 32, 6]


@pytest.mark.parametrize("response,reason", [
    (httpx.Response(503), "status_503"),
    (httpx.Response(401), "status_401"),
    (httpx.Response(200, text="not json"), "JSONDecodeError"),
    (httpx.Response(200, json=[1, 2]), "AttributeError"),
    (httpx.Response(200, json=rows([1, 0])), "wrong number of vectors"),
    (httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]}),
     "invalid vector index"),
    (httpx.Response(200, json=rows(["a", 0], [1, 0])), "invalid vector"),
    (httpx.Response(200, json=rows([0, 0], [1, 0])), "zero length"),
    (httpx.Response(200, json=rows([1, 0], [1, 0, 0])), "different dimensions"),
])
def test_unusable_remote_responses_raise_without_the_key(response, reason):
    with pytest.raises(EmbeddingError, match=reason) as raised:
        remote(lambda request: response).embed(["first", "second"])
    assert KEY not in str(raised.value)


def test_unreachable_remote_raises_embedding_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError, match="ConnectError"):
        remote(handler).embed(["first"])


def test_normalize_rejects_non_finite_values():
    with pytest.raises(EmbeddingError):
        normalize([math.nan, 1.0])


# --- matching --------------------------------------------------------------

class FakeEmbedder:
    """Maps known texts to fixed vectors along three axes: late, missing, small talk."""
    name = "fake"

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [normalize(self.vectors[text]) for text in texts]


EXAMPLES = {Issue.LATE_DELIVERY: ("late",), Issue.MISSING_ITEM: ("missing",), Issue.UNKNOWN: ("thanks",)}
VECTORS = {
    "late": [1, 0, 0], "missing": [0, 1, 0], "thanks": [0, 0, 1],
    "It came an hour behind.": [0.95, 0.1, 0.1],
    "A wrap was not there.": [0.1, 0.95, 0.1],
    "Cheers!": [0.1, 0.1, 0.95],
    "Something odd happened.": [0.5, 0.45, 0.4],  # close to nothing in particular
    "Late or missing, not sure.": [0.7, 0.68, 0.1],  # close to two issues at once
}


def matcher(threshold=0.8, margin=0.1):
    embedder = FakeEmbedder(VECTORS)
    return EmbeddingMatcher(embedder, threshold, margin, examples=EXAMPLES), embedder


def test_clear_match_answers_the_issue():
    m, _ = matcher()
    assert m.match("It came an hour behind.") == [Issue.LATE_DELIVERY]


def test_each_sentence_is_matched_on_its_own():
    m, embedder = matcher()
    assert m.match("A wrap was not there. It came an hour behind.") == [Issue.LATE_DELIVERY, Issue.MISSING_ITEM]
    assert embedder.calls[-1] == ["A wrap was not there.", "It came an hour behind."]


def test_small_talk_alone_is_confidently_no_issue():
    m, _ = matcher()
    assert m.match("Cheers!") == [Issue.UNKNOWN]


def test_small_talk_next_to_an_issue_keeps_the_issue():
    m, _ = matcher()
    assert m.match("Cheers! A wrap was not there.") == [Issue.MISSING_ITEM]


def test_weak_match_abstains():
    m, _ = matcher()
    assert m.match("Something odd happened.") is None


def test_close_runner_up_abstains():
    m, _ = matcher(threshold=0.5)
    assert m.match("Late or missing, not sure.") is None


def test_unsure_sentence_next_to_small_talk_abstains():
    m, _ = matcher()
    assert m.match("Cheers! Something odd happened.") is None


def test_empty_message_abstains():
    m, _ = matcher()
    assert m.match("   ") is None


def test_example_vectors_are_computed_once():
    m, embedder = matcher()
    m.match("Cheers!")
    m.match("It came an hour behind.")
    assert embedder.calls.count(["late", "missing", "thanks"]) == 1


def test_failed_example_embedding_is_retried_on_next_use():
    m, embedder = matcher()
    real_embed = embedder.embed
    embedder.embed = lambda texts: (_ for _ in ()).throw(EmbeddingError("down"))
    with pytest.raises(EmbeddingError):
        m.match("Cheers!")
    embedder.embed = real_embed
    assert m.match("Cheers!") == [Issue.UNKNOWN]


def test_vectors_from_a_different_model_are_rejected():
    m, embedder = matcher()
    m.warm_up()
    embedder.vectors = {**VECTORS, "Cheers!": [0.1, 0.9]}
    with pytest.raises(EmbeddingError, match="different dimensions"):
        m.match("Cheers!")
