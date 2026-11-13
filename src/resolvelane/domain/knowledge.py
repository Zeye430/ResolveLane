"""Help center articles: versioned customer-facing guidance, searched by the reply agent.

Support policies (delivery.SupportPolicy) tell support how to handle an issue and are looked
up by issue type. Help articles answer customers' general questions and are found by search.
Each article version has an effective period, so an answer can never rest on outdated text.
"""

from dataclasses import dataclass
from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class ArticleSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    heading: str = Field(min_length=1)
    # A section is the unit of search and citation, so it is kept short enough to quote whole.
    text: str = Field(min_length=1, max_length=1200)


class HelpArticle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    article_id: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    version: int = Field(ge=1, strict=True)
    title: str = Field(min_length=1)
    effective_from: AwareDatetime
    effective_until: AwareDatetime | None = None
    sections: tuple[ArticleSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_article(self):
        if self.effective_until is not None and self.effective_until <= self.effective_from:
            raise ValueError("Article end must be after its start")
        if len({section.section_id for section in self.sections}) != len(self.sections):
            raise ValueError("Section IDs must be unique within an article")
        return self

    def in_effect(self, now: datetime) -> bool:
        return self.effective_from <= now and (self.effective_until is None or now < self.effective_until)


@dataclass(frozen=True)
class Citation:
    """The exact article version and section a reply relied on."""
    article_id: str
    version: int
    title: str
    section: str

    @property
    def source_id(self) -> str:
        return f"{self.article_id}@{self.version}#{self.section}"
