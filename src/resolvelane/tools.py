"""Tools the reply agent may call, and the gateway that runs them.

The gateway, not the model, decides whose data a tool sees: every handler receives a
ToolContext built by the application for the authenticated customer and the case's order.
Tool inputs therefore never name a customer or an order, so the model cannot ask for
someone else's records.

Tools that would change a case do not change it. They record a proposed effect, and the
conversation service applies it later under the case lock, after re-checking the case.
"""

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from resolvelane.domain.case import Issue
from resolvelane.domain.delivery import Order, SupportPolicy
from resolvelane.knowledge import HelpCenter, Passage

logger = logging.getLogger("resolvelane.tools")


@dataclass(frozen=True)
class ReviewRequest:
    reason: str


@dataclass
class ToolContext:
    customer_id: str
    order: Order | None
    now: datetime
    policies: Sequence[SupportPolicy]
    help_center: HelpCenter | None = None
    effects: list[ReviewRequest] = field(default_factory=list)
    # Help center passages returned during this reply; a reply may cite only these.
    retrieved: dict[str, Passage] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolTrace:
    name: str
    ok: bool
    latency_ms: float
    error: str = ""


@dataclass(frozen=True)
class ToolResult:
    content: str  # JSON text for the model
    is_error: bool
    trace: ToolTrace


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[_Input]
    handler: Callable[[ToolContext, Any], dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


# --- support tools ---------------------------------------------------------

IssueName = Literal[tuple(issue.value for issue in Issue if issue is not Issue.UNKNOWN)]


class PolicyInput(_Input):
    issue: IssueName = Field(description="The issue to look up")


def policy_in_effect(policies: Sequence[SupportPolicy], issue: str, now: datetime) -> dict[str, Any]:
    """The newest policy for ``issue`` in effect at ``now``; expired and future policies do not apply."""
    for_issue = [policy for policy in policies if policy.issue_type == issue]
    if not for_issue:
        return {"policy": None, "note": "No support policy applies to this kind of question."}
    in_effect = [policy for policy in for_issue
                 if policy.effective_from <= now and (policy.effective_until is None or now < policy.effective_until)]
    if not in_effect:
        return {"policy": None,
                "note": "The policy for this issue is not in effect. A support agent must review the case."}
    policy = max(in_effect, key=lambda item: item.effective_from)
    return {"policy": {"title": policy.title, "version": policy.version, "guidance": policy.guidance}}


def get_support_policy(context: ToolContext, args: PolicyInput) -> dict[str, Any]:
    return {"issue": args.issue, **policy_in_effect(context.policies, args.issue, context.now)}


class SearchInput(_Input):
    query: str = Field(min_length=3, max_length=200,
                       description="What to look up, in a few words, e.g. 'refund arrival time'")


def search_help_center(context: ToolContext, args: SearchInput) -> dict[str, Any]:
    if context.help_center is None:
        return {"passages": [], "note": "The help center is not available."}
    result = context.help_center.search(args.query, context.now)
    context.retrieved.update((passage.source_id, passage) for passage in result.passages)
    if not result.passages:
        return {"passages": [], "note": "Nothing in the help center matches. Do not answer from memory."}
    return {"passages": [{"source_id": passage.source_id, "article": passage.citation.title,
                          "section": passage.heading, "text": passage.text} for passage in result.passages]}


class ReviewInput(_Input):
    reason: str = Field(min_length=1, max_length=300, description="Why a support agent should look at this case")


def request_human_review(context: ToolContext, args: ReviewInput) -> dict[str, Any]:
    # Asking twice in one reply changes nothing, so a repeated call is harmless.
    if not context.effects:
        context.effects.append(ReviewRequest(args.reason))
    return {"status": "review_requested",
            "note": "A support agent will review the case. Do not promise an outcome or a time."}


SUPPORT_TOOLS = (
    ToolSpec("search_help_center",
             "Search the help center articles currently in effect, for general questions such as refund "
             "timing, cancelling, changing the address, or reporting a problem. Returns passages with a "
             "source_id to cite; returns none when the help center does not cover the question.",
             SearchInput, search_help_center),
    ToolSpec("get_support_policy",
             "Look up the support policy currently in effect for an issue. Use it before describing how "
             "an issue is handled. Returns no policy when none is in effect.",
             PolicyInput, get_support_policy),
    ToolSpec("request_human_review",
             "Ask a support agent to review this case. Use it when the customer asks for a person, or when the "
             "customer needs something the available facts and policies cannot answer.",
             ReviewInput, request_human_review),
)


# --- gateway ---------------------------------------------------------------

def _validation_summary(error: ValidationError) -> str:
    # Field names and problems only; the customer-influenced input values are not echoed.
    return "; ".join(f"{'.'.join(map(str, item['loc'])) or 'input'}: {item['msg']}" for item in error.errors())


class ToolGateway:
    def __init__(self, specs: Sequence[ToolSpec] = SUPPORT_TOOLS):
        self.specs = {spec.name: spec for spec in specs}

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self.specs.values()]

    def execute(self, name: str, args: Any, context: ToolContext) -> ToolResult:
        started = time.perf_counter()

        def done(payload: dict[str, Any], error: str = "") -> ToolResult:
            trace = ToolTrace(name, not error, round((time.perf_counter() - started) * 1000, 2), error)
            logger.info("tool_call name=%s ok=%s error=%s", name, trace.ok, error or "-")
            return ToolResult(json.dumps(payload), bool(error), trace)

        spec = self.specs.get(name)
        if spec is None:
            return done({"error": f"Unknown tool {name!r}"}, "unknown_tool")
        try:
            parsed = spec.input_model.model_validate(args)
        except ValidationError as error:
            return done({"error": f"Invalid input: {_validation_summary(error)}"}, "invalid_input")
        try:
            return done(spec.handler(context, parsed))
        except Exception:
            logger.exception("tool_failed name=%s", name)
            return done({"error": "The tool failed"}, "tool_failed")
