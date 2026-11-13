from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from resolvelane.domain.knowledge import HelpArticle
from resolvelane.embeddings import EmbeddingError, normalize
from resolvelane.knowledge import HelpCenter, bm25_scores, tokens
from resolvelane.simulator.help_center import ARTICLES

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
LAST_YEAR = datetime(2025, 6, 1, tzinfo=timezone.utc)


def ids(result):
    return [passage.source_id for passage in result.passages]


def test_word_forms_match():
    assert tokens("refund refunds refunded") == ["refund"] * 3
    assert tokens("cancel cancelled cancelling") == ["cancel"] * 3
    assert len(set(tokens("arrive arrives arrival"))) == 1
    assert len(set(tokens("time timing"))) == 1
    assert tokens("How is the order?") == ["order"]


def test_bm25_prefers_documents_with_rarer_matching_terms():
    scores = bm25_scores(tokens("refund address"), [tokens("order refund"), tokens("order"), tokens("address change")])
    assert scores[1] == 0
    assert scores[0] > 0 and scores[2] > 0


def test_only_the_newest_version_in_effect_is_searched():
    current = HelpCenter(ARTICLES).search("How long does a refund take?", NOW)
    assert ids(current)[0] == "refund-timing@2#arrival"
    assert not any(source.startswith("refund-timing@1") for source in ids(current))
    earlier = HelpCenter(ARTICLES).search("How long does a refund take?", LAST_YEAR)
    assert ids(earlier) == ["refund-timing@1#arrival"]


def test_articles_not_yet_in_effect_are_not_searched():
    assert HelpCenter(ARTICLES).search("change the delivery address", LAST_YEAR).passages == []


def test_question_the_help_center_does_not_cover_finds_nothing():
    result = HelpCenter(ARTICLES).search("Can I pay with cash?", NOW)
    assert result.passages == [] and result.mode == "keyword"


def test_search_returns_at_most_the_limit():
    assert len(HelpCenter(ARTICLES).search("order refund cancel address report", NOW, limit=3).passages) == 3


class FakeEmbedder:
    """Three directions: refund arrival, everything else in the help center, and paying in cash."""
    name = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("down")

        def vector(text):
            if "money back" in text or "When a refund arrives" in text:
                return [1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0] if "cash" in text else [0.0, 1.0, 0.0]
        return [normalize(vector(text)) for text in texts]


def test_paraphrase_without_shared_keywords_is_found_by_embedding():
    keyword_only = HelpCenter(ARTICLES).search("when do I get my money back", NOW)
    assert keyword_only.passages == []
    hybrid = HelpCenter(ARTICLES, FakeEmbedder()).search("when do I get my money back", NOW)
    assert hybrid.mode == "hybrid"
    assert ids(hybrid) == ["refund-timing@2#arrival"]


def test_weak_similarity_alone_does_not_make_a_passage_relevant():
    result = HelpCenter(ARTICLES, FakeEmbedder()).search("Can I pay with cash?", NOW)
    assert result.passages == []


def test_failing_embedder_falls_back_to_keywords():
    result = HelpCenter(ARTICLES, FakeEmbedder(fail=True)).search("How long does a refund take?", NOW)
    assert result.mode == "keyword"
    assert ids(result)[0] == "refund-timing@2#arrival"


def test_passage_vectors_are_computed_once():
    embedder = FakeEmbedder()
    help_center = HelpCenter(ARTICLES, embedder)
    help_center.search("refund", NOW)
    help_center.search("address", NOW)
    assert embedder.calls == 3  # passages once, then one call per query


def article(**overrides):
    values = {"article_id": "demo", "version": 1, "title": "Demo", "effective_from": "2026-01-01T00:00:00Z",
              "sections": [{"section_id": "one", "heading": "One", "text": "Text."}], **overrides}
    return HelpArticle.model_validate(values)


@pytest.mark.parametrize("overrides", [
    {"effective_until": "2025-12-31T00:00:00Z"},
    {"sections": [{"section_id": "one", "heading": "A", "text": "x"}, {"section_id": "one", "heading": "B", "text": "y"}]},
    {"sections": [{"section_id": "one", "heading": "A", "text": "x" * 1201}]},
    {"sections": []},
    {"effective_from": "2026-01-01T00:00:00"},
])
def test_invalid_articles_are_rejected(overrides):
    with pytest.raises(ValidationError):
        article(**overrides)


def test_duplicate_article_versions_are_rejected():
    with pytest.raises(ValueError):
        HelpCenter([article(), article()])
