"""Evaluation scenarios: scripted conversations with what each reply must and must not do.

A scenario is a list of steps. A customer step sends a message and states expectations for
the reply; other steps act as a support agent or send due refund payments, so a scenario can
follow a case from the first message to the money arriving.

Expectations are checked by code, exactly. What code cannot check well, such as whether a
reply is grounded or says more than the customer needs, is left to the judge.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resolvelane.domain.case import CaseStatus
from resolvelane.simulator.payment_provider import FAULTS

DEFAULT_SCENARIOS = Path(__file__).resolve().parents[3] / "evals" / "scenarios.json"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expect(_Strict):
    case_status: CaseStatus | None = None
    needs_review: bool | None = None
    # Acceptable refund drafts by the assistant after this reply: item_id -> quantity, or null for
    # "no refund on the case". Omitted: not checked.
    refund_draft: list[dict[str, int] | None] | None = None
    refund_count: int | None = Field(default=None, ge=0, description="Refunds on the case, decided or not")
    written_by_agent: bool | None = Field(default=None, description="True: the agent's reply was used, not rules")
    cites: list[str] = Field(default_factory=list, description="Article IDs the reply must cite")
    must_match: list[str] = Field(default_factory=list, description="Regular expressions the reply must match")
    must_not_match: list[str] = Field(default_factory=list,
                                      description="Regular expressions the reply must not match")

    @model_validator(mode="after")
    def patterns_compile(self):
        for pattern in self.must_match + self.must_not_match:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"Invalid pattern {pattern!r}: {error}") from None
        return self


class Step(_Strict):
    customer: str | None = Field(default=None, min_length=1, description="A customer message")
    expect: Expect = Expect()
    support_agent: Literal["approve_refund", "reject_refund", "resolve"] | None = None
    reply: str | None = Field(default=None, description="A support agent's message to the customer")
    send_payments: bool = False

    @model_validator(mode="after")
    def one_action(self):
        actions = [self.customer is not None, self.support_agent is not None, self.reply is not None,
                   self.send_payments]
        if sum(actions) != 1:
            raise ValueError("A step does exactly one thing: customer, support_agent, reply or send_payments")
        if self.customer is None and self.expect != Expect():
            raise ValueError("Only customer steps have expectations")
        return self


class Scenario(_Strict):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    description: str = Field(min_length=1)
    customer_id: str = "demo-customer-101"
    order_id: str | None = None  # sent with the first message
    payment_faults: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def check(self):
        unknown = set(self.payment_faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"Unknown payment faults: {sorted(unknown)}")
        if self.steps[0].customer is None:
            raise ValueError("A scenario starts with a customer message")
        return self


class Suite(_Strict):
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Scenario IDs must be unique")
        return self


def load_suite(path: Path = DEFAULT_SCENARIOS) -> Suite:
    return Suite.model_validate(json.loads(Path(path).read_text()))
