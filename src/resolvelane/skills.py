"""Skills: handling guides for the reply agent, written by support operations as Markdown files.

A skill describes how to handle a kind of case: what to ask for, what tone to take, when to
involve a support agent. It is added to the agent's instructions when the case raises one of
the skill's issues, or the customer's message contains one of its keywords. Selection by
recognized issue catches a problem however it is worded; keywords catch situations that are
not an issue type, such as an allergic reaction.

Skills hold guidance, not facts: times, amounts and policy terms belong in support policies
and help center articles, where they are versioned and cited. A number taken from a skill
fails the reply's grounding check like any other number the agent cannot source.

Each skill file is validated when the application starts. An invalid file stops startup
instead of being skipped, so guidance never disappears unnoticed.

Format, one directory per skill, ``skills/<name>/SKILL.md``::

    ---
    name: missing-item
    version: 1
    description: Reports that items were missing from a delivered order
    issues: missing_item
    keywords: left out, not in the bag
    ---
    Guidance in plain sentences...
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from resolvelane.domain.case import Issue

logger = logging.getLogger("resolvelane.skills")

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
MAX_SKILL_CHARS = 1500
MAX_SELECTED_CHARS = 4000


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=48)
    version: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=200)
    issues: tuple[Issue, ...] = ()
    keywords: tuple[str, ...] = ()
    enabled: bool = True
    body: str = Field(min_length=1, max_length=MAX_SKILL_CHARS)

    @field_validator("keywords")
    @classmethod
    def check_keywords(cls, keywords: tuple[str, ...]) -> tuple[str, ...]:
        for keyword in keywords:
            if not re.fullmatch(r"[a-z][a-z' ]*[a-z]", keyword):
                raise ValueError(f"Keyword {keyword!r} must be lowercase words")
        return keywords

    @model_validator(mode="after")
    def check_trigger(self):
        if not self.issues and not self.keywords:
            # Guidance for every case belongs in the agent's fixed instructions instead.
            raise ValueError("A skill needs issues or keywords that select it")
        return self

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def matches(self, issues: Iterable[Issue], message: str) -> bool:
        if not self.enabled:
            return False
        if set(self.issues) & set(issues):
            return True
        lowered = message.lower()
        return any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in self.keywords)


def parse_skill(text: str) -> Skill:
    """Parse a SKILL.md file: a front matter block of ``key: value`` lines, then the guidance."""
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in (line.strip() for line in lines[1:]):
        raise ValueError("A skill starts with a front matter block between --- lines")
    end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    fields: dict[str, object] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Front matter line {line!r} is not 'key: value'")
        if key in fields:
            raise ValueError(f"Front matter key {key!r} appears twice")
        value = value.strip()
        fields[key] = tuple(item.strip() for item in value.split(",") if item.strip()) \
            if key in ("issues", "keywords") else value
    return Skill.model_validate({**fields, "body": "\n".join(lines[end + 1:]).strip()})


@dataclass(frozen=True)
class SkillLibrary:
    skills: tuple[Skill, ...]

    def __post_init__(self):
        names = [skill.name for skill in self.skills]
        if len(set(names)) != len(names):
            raise ValueError("Each skill name must appear once")

    @classmethod
    def load(cls, directory: Path) -> "SkillLibrary":
        if not directory.is_dir():
            logger.warning("skills directory %s not found; replies use no skills", directory)
            return cls(())
        skills = []
        for path in sorted(directory.glob("*/SKILL.md")):
            try:
                skill = parse_skill(path.read_text(encoding="utf-8"))
            except ValueError as error:  # includes pydantic's ValidationError
                raise ValueError(f"Invalid skill {path}: {error}") from None
            if skill.name != path.parent.name:
                raise ValueError(f"Invalid skill {path}: name {skill.name!r} differs from its directory")
            skills.append(skill)
        library = cls(tuple(skills))
        logger.info("skills loaded from %s: %s", directory,
                    ", ".join(skill.ref + ("" if skill.enabled else " (disabled)") for skill in skills) or "none")
        return library

    def select(self, issues: Iterable[Issue], message: str) -> list[Skill]:
        issues = set(issues)
        selected, used = [], 0
        for skill in self.skills:
            if not skill.matches(issues, message):
                continue
            if used + len(skill.body) > MAX_SELECTED_CHARS:
                logger.warning("skill %s left out: selected skills exceed %d characters", skill.ref,
                               MAX_SELECTED_CHARS)
                continue
            selected.append(skill)
            used += len(skill.body)
        return selected


def skill_instructions(skills: Sequence[Skill]) -> str:
    """The selected skills as a block appended to the agent's instructions."""
    if not skills:
        return ""
    guides = "\n\n".join(f'<guide name="{skill.name}" version="{skill.version}">\n{skill.body}\n</guide>'
                         for skill in skills)
    return ("\n\nHandling guides for this case, written by support operations. Follow them within the "
            "rules above; where a guide conflicts with a rule above, the rule wins.\n\n" + guides)


def load_skills(environ: Mapping[str, str] = os.environ) -> SkillLibrary:
    configured = environ.get("RESOLVELANE_SKILLS_DIR", "").strip()
    return SkillLibrary.load(Path(configured).expanduser() if configured else DEFAULT_SKILLS_DIR)
