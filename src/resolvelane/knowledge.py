"""Search over the help center: keyword (BM25) and embedding retrieval, fused by rank.

Only article versions in effect at the time of the question are searched, and only the newest
of those per article, so outdated guidance cannot be returned. Keyword retrieval catches exact
terms such as "refund" or "address"; embedding retrieval catches the same question asked in
other words. Rankings are combined with reciprocal rank fusion, which needs no tuning between
the two kinds of scores. Without an embedder, or when it fails, search continues on keywords.
"""

import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from resolvelane.domain.knowledge import Citation, HelpArticle
from resolvelane.embeddings import Embedder, EmbeddingError, cosine

logger = logging.getLogger("resolvelane.knowledge")


@dataclass(frozen=True)
class Passage:
    citation: Citation
    heading: str
    text: str

    @property
    def source_id(self) -> str:
        return self.citation.source_id

    @property
    def indexed_text(self) -> str:
        return f"{self.citation.title}. {self.heading}. {self.text}"


@dataclass(frozen=True)
class SearchResult:
    passages: list[Passage]
    mode: str  # "hybrid" or "keyword"


_STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or so that the "
    "this to was what when where which who will with you your".split())


def _stem(word: str) -> str:
    """A deliberately small stemmer: enough to match refund/refunded, cancel/cancelled,
    arrive/arrival and time/timing. Embedding search covers what it misses."""
    for suffix, replacement in (("ations", ""), ("ation", ""), ("ings", ""), ("ing", ""), ("ies", "y"),
                                ("ed", ""), ("es", ""), ("s", ""), ("al", ""), ("e", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: len(word) - len(suffix)] + replacement
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # cancell -> cancel
    return word


def tokens(text: str) -> list[str]:
    return [_stem(word) for word in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")) if word not in _STOP_WORDS]


def bm25_scores(query: Sequence[str], documents: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    count = len(documents)
    average = sum(map(len, documents)) / count if count else 0
    frequency = Counter(term for document in documents for term in set(document))
    scores = []
    for document in documents:
        counts = Counter(document)
        score = 0.0
        for term in set(query):
            if counts[term]:
                idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
                tf = counts[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(document) / average))
        scores.append(score)
    return scores


class HelpCenter:
    RRF_K = 60
    # A passage that shares no keyword with the question must be at least this similar to it.
    # Measured for BAAI/bge-small-en-v1.5 on invented questions: answered questions without a
    # shared keyword scored 0.71 to 0.76, questions the help center does not cover at most 0.64.
    MIN_SIMILARITY = 0.70

    def __init__(self, articles: Sequence[HelpArticle], embedder: Embedder | None = None,
                 min_similarity: float = MIN_SIMILARITY):
        keys = [(article.article_id, article.version) for article in articles]
        if len(set(keys)) != len(keys):
            raise ValueError("Each article version must appear once")
        self.articles = tuple(articles)
        self.embedder = embedder
        self.min_similarity = min_similarity
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def passages_in_effect(self, now: datetime) -> list[Passage]:
        newest: dict[str, HelpArticle] = {}
        for article in self.articles:
            if not article.in_effect(now):
                continue
            current = newest.get(article.article_id)
            if current is None or article.version > current.version:
                newest[article.article_id] = article
        return [Passage(Citation(article.article_id, article.version, article.title, section.section_id),
                        section.heading, section.text)
                for article in newest.values() for section in article.sections]

    def _similarities(self, query: str, passages: list[Passage]) -> list[float]:
        missing = [passage for passage in passages if passage.source_id not in self._vectors]
        if missing:
            vectors = self.embedder.embed([passage.indexed_text for passage in missing])
            with self._lock:
                self._vectors.update(zip((passage.source_id for passage in missing), vectors))
        query_vector = self.embedder.embed([query])[0]
        return [cosine(query_vector, self._vectors[passage.source_id]) for passage in passages]

    def search(self, query: str, now: datetime, limit: int = 4) -> SearchResult:
        passages = self.passages_in_effect(now)
        if not passages:
            return SearchResult([], "keyword")
        lexical = bm25_scores(tokens(query), [tokens(passage.indexed_text) for passage in passages])
        similarities: list[float] | None = None
        if self.embedder is not None:
            try:
                similarities = self._similarities(query, passages)
            except EmbeddingError as error:
                logger.warning("help_center_search embedding unavailable (%s); searching keywords only", error)

        def ranks(scores: list[float]) -> dict[int, int]:
            order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return {index: rank for rank, index in enumerate(order, start=1)}

        lexical_rank = ranks(lexical)
        dense_rank = ranks(similarities) if similarities else {}
        fused = []
        for index in range(len(passages)):
            relevant = lexical[index] > 0 or (similarities is not None and similarities[index] >= self.min_similarity)
            if not relevant:
                continue
            score = 1 / (self.RRF_K + lexical_rank[index]) if lexical[index] > 0 else 0.0
            if similarities is not None:
                score += 1 / (self.RRF_K + dense_rank[index])
            fused.append((score, index))
        fused.sort(reverse=True)
        return SearchResult([passages[index] for _, index in fused[:limit]],
                            "hybrid" if similarities is not None else "keyword")
