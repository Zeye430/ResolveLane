"""Embedding tiers of issue recognition: they take over when the language model is unavailable.

A message is compared with invented example messages for each issue (issue_examples.py).
Each embedder keeps its own example vectors, so a message vector is only ever compared with
vectors from the same model. A tier answers only when the best match is close enough and
clearly ahead of the runner-up; otherwise it abstains and the next tier is tried, because a
confident wrong label sends the conversation down the wrong path.
"""

import logging
import math
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

from resolvelane.domain.case import Issue, ordered_issues
from resolvelane.issue_examples import ISSUE_EXAMPLES

logger = logging.getLogger("resolvelane.embeddings")

# (minimum similarity, minimum lead over the runner-up issue) per model, measured with
# scripts/calibrate_embeddings.py. A model without an entry is not used unless thresholds
# are configured explicitly, because an uncalibrated tier answers confidently and wrongly.
CALIBRATED_THRESHOLDS: dict[str, tuple[float, float]] = {
    # 44 held-out messages: 39 right, 2 wrong, 3 passed to the next tier. A margin did not help
    # this model: requiring any lead dropped more right answers than wrong ones.
    "BAAI/bge-small-en-v1.5": (0.68, 0.00),
}


class EmbeddingError(Exception):
    """The embedding provider could not produce usable vectors."""


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit-length vector per text, or raise EmbeddingError."""


def normalize(vector: Sequence[float]) -> list[float]:
    if not vector or any(not math.isfinite(x) for x in vector):
        raise EmbeddingError("vector is empty or not finite")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingError("vector has zero length")
    return [x / norm for x in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are unit length


class RemoteEmbedder:
    """An OpenAI-compatible /embeddings endpoint, such as SiliconFlow, Jina or Voyage."""

    BATCH_SIZE = 32

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 5.0,
                 transport: httpx.BaseTransport | None = None):
        self.name = f"remote/{model}"
        self.model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout_s,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    follow_redirects=False, transport=transport)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(self._request(list(texts[start:start + self.BATCH_SIZE])))
        return vectors

    def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.post("embeddings", json={"model": self.model, "input": texts,
                                                             "encoding_format": "float"})
            response.raise_for_status()
            rows = response.json().get("data")
        except httpx.HTTPStatusError as error:
            raise EmbeddingError(f"status_{error.response.status_code}") from None
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            # The reason names the error type only, never request content or credentials.
            raise EmbeddingError(type(error).__name__) from None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingError("wrong number of vectors")
        result: list[list[float] | None] = [None] * len(texts)
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else None
            vector = row.get("embedding") if isinstance(row, dict) else None
            if type(index) is not int or not 0 <= index < len(texts) or result[index] is not None:
                raise EmbeddingError("invalid vector index")
            if not isinstance(vector, list) or any(type(x) not in (int, float) for x in vector):
                raise EmbeddingError("invalid vector")
            result[index] = normalize(vector)
        if len({len(vector) for vector in result}) != 1:
            raise EmbeddingError("vectors have different dimensions")
        return result  # type: ignore[return-value]


class LocalEmbedder:
    """A small embedding model run on this machine with fastembed (ONNX), so it works offline."""

    def __init__(self, model: str, cache_dir: str | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise EmbeddingError("fastembed is not installed") from None
        self.name = f"local/{model}"
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            with self._lock:
                return [normalize([float(x) for x in vector]) for vector in self._model.embed(list(texts))]
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(type(error).__name__) from None


_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|\n+")


class EmbeddingMatcher:
    MAX_SENTENCES = 6

    def __init__(self, embedder: Embedder, threshold: float, margin: float,
                 examples: Mapping[Issue, Sequence[str]] = ISSUE_EXAMPLES):
        self.embedder = embedder
        self.threshold = threshold
        self.margin = margin
        self._examples = {issue: tuple(texts) for issue, texts in examples.items()}
        self._example_vectors: dict[Issue, list[list[float]]] | None = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._vectors()

    def _vectors(self) -> dict[Issue, list[list[float]]]:
        with self._lock:
            if self._example_vectors is None:
                texts = [text for group in self._examples.values() for text in group]
                vectors = self.embedder.embed(texts)
                if len(vectors) != len(texts) or len({len(vector) for vector in vectors}) != 1:
                    raise EmbeddingError("example vectors are incomplete or have different dimensions")
                remaining = iter(vectors)
                self._example_vectors = {issue: [next(remaining) for _ in group]
                                         for issue, group in self._examples.items()}
            return self._example_vectors

    def best_match(self, vector: Sequence[float]) -> tuple[Issue, float, float]:
        """The closest issue, its similarity, and its lead over the runner-up issue."""
        ranked = sorted(((max(cosine(vector, example) for example in group), issue)
                         for issue, group in self._vectors().items()), key=lambda pair: pair[0], reverse=True)
        (score, best), (runner_up, _) = ranked[0], ranked[1]
        return best, score, score - runner_up

    def match(self, message: str) -> list[Issue] | None:
        """Issues raised by the message, [UNKNOWN] if confidently none, or None to abstain."""
        sentences = [part.strip() for part in _SENTENCE_BREAK.split(message) if part.strip()]
        if not sentences:
            return None
        sentences = sentences[:self.MAX_SENTENCES]
        examples = self._vectors()
        dimension = len(next(iter(examples.values()))[0])
        vectors = self.embedder.embed(sentences)
        if len(vectors) != len(sentences):
            raise EmbeddingError("wrong number of message vectors")
        found: list[Issue] = []
        unsure = False
        # Each sentence is matched on its own, so "It was late. A wrap is missing." finds both issues.
        for vector in vectors:
            if len(vector) != dimension:
                raise EmbeddingError("message and examples have different dimensions")
            best, score, lead = self.best_match(vector)
            confident = score >= self.threshold and lead >= self.margin
            logger.debug("embedding_match embedder=%s best=%s score=%.3f lead=%.3f confident=%s",
                         self.embedder.name, best.value, score, lead, confident)
            if confident:
                found.append(best)
            else:
                unsure = True
        issues = [issue for issue in found if issue is not Issue.UNKNOWN]
        if issues:
            return ordered_issues(issues)
        # Nothing matched an issue: answer "no issue" only if every sentence was clearly small talk.
        return None if unsure else [Issue.UNKNOWN]
