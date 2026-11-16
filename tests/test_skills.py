import json
import logging
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from resolvelane.domain.case import CaseStatus, Issue
from resolvelane.llm import LLMProvider
from resolvelane.reply_agent import SYSTEM_PROMPT, ReplyAgent, ReplyRequest
from resolvelane.resilience import CircuitBreaker
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.skills import (DEFAULT_SKILLS_DIR, MAX_SELECTED_CHARS, Skill, SkillLibrary, load_skills,
                                parse_skill)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ORDER = next(order for order in ORDERS if order.order_id == "demo-order-1001")


def skill_file(name="missing-item", front=None, body="Name the items the order lists."):
    fields = {"name": name, "version": "1", "description": "Missing items", "issues": "missing_item",
              **(front or {})}
    lines = [f"{key}: {value}" for key, value in fields.items() if value is not None]
    return "---\n" + "\n".join(lines) + "\n---\n" + body + "\n"


def skill(name="demo", issues=(), keywords=(), body="Guidance.", enabled=True):
    return Skill(name=name, version=1, description="Demo", issues=issues, keywords=keywords, body=body,
                 enabled=enabled)


# --- the shipped skills ------------------------------------------------------------

def test_shipped_skills_load():
    library = load_skills({})
    assert {item.name for item in library.skills} >= {"missing-item", "wrong-or-damaged-item", "late-delivery",
                                                       "delivery-status", "cancellation", "allergy-or-illness"}


def test_every_issue_type_has_a_shipped_skill():
    covered = {issue for item in load_skills({}).skills for issue in item.issues}
    assert covered == set(Issue) - {Issue.UNKNOWN}


def test_shipped_skills_contain_no_numbers():
    # Times, amounts and deadlines belong in policies and help center articles.
    for path in DEFAULT_SKILLS_DIR.glob("*/SKILL.md"):
        assert not re.search(r"\d", parse_skill(path.read_text()).body), path


# --- parsing and validation -----------------------------------------------------------

def test_skill_file_is_parsed():
    parsed = parse_skill(skill_file(front={"keywords": "left out, not in the bag", "version": "3"}))
    assert parsed.ref == "missing-item@3"
    assert parsed.issues == (Issue.MISSING_ITEM,)
    assert parsed.keywords == ("left out", "not in the bag")
    assert parsed.body == "Name the items the order lists." and parsed.enabled


@pytest.mark.parametrize("text,problem", [
    ("Name the items.", "front matter"),
    ("---\nname: missing-item\nName the items.", "front matter"),
    (skill_file(front={"owner": "ops"}), "owner"),
    (skill_file(front={"issues": "missing_item, lost_parcel"}), "lost_parcel"),
    (skill_file(front={"keywords": "Left Out"}), "lowercase"),
    (skill_file(front={"issues": None}), "issues or keywords"),
    (skill_file(front={"version": "0"}), "version"),
    (skill_file(name="Missing Item"), "name"),
    (skill_file(body=""), "body"),
    (skill_file(body="x" * 1501), "body"),
    ("---\nname: a\nname: b\n---\nText", "twice"),
    ("---\njust words\n---\nText", "key: value"),
])
def test_invalid_skill_is_rejected(text, problem):
    with pytest.raises(ValueError, match=problem):
        parse_skill(text)


def test_invalid_skill_file_stops_loading_and_names_the_file(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "SKILL.md").write_text(skill_file(name="good"))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text(skill_file(name="broken", front={"issues": "lost_parcel"}))
    with pytest.raises(ValueError, match="broken"):
        SkillLibrary.load(tmp_path)


def test_skill_name_must_match_its_directory(tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "SKILL.md").write_text(skill_file())
    with pytest.raises(ValueError, match="directory"):
        SkillLibrary.load(tmp_path)


def test_duplicate_skill_names_are_rejected():
    with pytest.raises(ValueError, match="once"):
        SkillLibrary((skill(issues=(Issue.LATE_DELIVERY,)), skill(issues=(Issue.MISSING_ITEM,))))


def test_missing_directory_means_no_skills(tmp_path):
    assert load_skills({"RESOLVELANE_SKILLS_DIR": str(tmp_path / "absent")}).skills == ()


# --- selection ---------------------------------------------------------------------

def test_skill_is_selected_by_issue_or_whole_word_keyword():
    library = SkillLibrary((skill("late", issues=(Issue.LATE_DELIVERY,)),
                            skill("illness", keywords=("ill", "food poisoning"))))
    assert [item.name for item in library.select([Issue.LATE_DELIVERY], "It was late.")] == ["late"]
    assert [item.name for item in library.select([], "I feel ill after eating it.")] == ["illness"]
    assert [item.name for item in library.select([], "Is this FOOD POISONING?")] == ["illness"]
    assert library.select([], "I'll check the bill and the pillow.") == []
    assert library.select([Issue.MISSING_ITEM], "A wrap is missing.") == []


def test_disabled_skill_is_not_selected():
    library = SkillLibrary((skill(issues=(Issue.LATE_DELIVERY,), enabled=False),))
    assert library.select([Issue.LATE_DELIVERY], "late") == []


def test_selected_guides_are_bounded(caplog):
    long = "x" * 1500
    library = SkillLibrary(tuple(skill(f"s{n}", issues=(Issue.LATE_DELIVERY,), body=long) for n in range(4)))
    with caplog.at_level(logging.WARNING, logger="resolvelane.skills"):
        selected = library.select([Issue.LATE_DELIVERY], "")
    assert sum(len(item.body) for item in selected) <= MAX_SELECTED_CHARS
    assert [item.name for item in selected] == ["s0", "s1"]
    assert any("s2@1 left out" in record.getMessage() for record in caplog.records)


# --- in the agent ------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.answer)])


def draft(message, issues, case_issues=()):
    client = ScriptedClient(json.dumps({"reply": "Thanks for letting us know.", "sources": []}))
    agent = ReplyAgent(LLMProvider(client, "test-model", CircuitBreaker("llm", 3, 30)), POLICIES,
                       skills=load_skills({}))
    request = ReplyRequest("demo-customer-101", message, [], ORDER, list(issues), list(case_issues or issues),
                           CaseStatus.OPEN, False, "Reference reply.", NOW)
    return agent.draft(request), client.calls[0]["system"]


def test_guides_for_the_case_are_added_to_the_instructions():
    result, system = draft("A wrap is missing.", [Issue.MISSING_ITEM])
    assert system.startswith(SYSTEM_PROMPT)
    assert '<guide name="missing-item" version="1">' in system
    assert "late-delivery" not in system
    assert result.skills == ("missing-item@1",)


def test_guide_for_an_earlier_issue_of_the_case_still_applies():
    result, _ = draft("Any news?", [Issue.UNKNOWN], case_issues=[Issue.WRONG_ITEM])
    assert result.skills == ("wrong-or-damaged-item@1",)


def test_keyword_guide_applies_without_an_issue():
    result, system = draft("I think I'm having an allergic reaction to the wrap.", [Issue.UNKNOWN])
    assert result.skills == ("allergy-or-illness@1",)
    assert "emergency services" in system


def test_without_matching_guides_the_instructions_are_unchanged():
    result, system = draft("Hello there.", [Issue.UNKNOWN])
    assert system == SYSTEM_PROMPT and result.skills == ()
