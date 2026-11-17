"""Run evaluation scenarios through the whole application and score the replies.

Each scenario runs against a fresh application with an empty in-memory case store, through
the HTTP API like a real client, with the configured recognizer and reply agent. Support
agent steps and payments go through the same services the API uses. After each customer
message, code checks the expectations exactly and the judge (optional) scores the reply.

The model is not deterministic, so a scenario can be run several times (``repeat``) and
results are pass rates. A report is compared with the saved baseline only when both ran the
same suite: the same scenarios, judge instructions and models (``suite_id``).
"""

import asyncio
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.evaluation import judge as judge_module
from resolvelane.evaluation.judge import Judgement, Material
from resolvelane.evaluation.scenarios import Scenario, Step, Suite
from resolvelane.payments import send_due
from resolvelane.refunds import open_refund
from resolvelane.reply_agent import AgentFailed, case_facts
from resolvelane.services import support_desk
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.simulator.help_center import ARTICLES
from resolvelane.simulator.order_source import InMemoryOrderSource
from resolvelane.simulator.payment_provider import SimulatedPaymentProvider

AGENT_ID = "eval-agent-901"
# Invented credentials, valid only inside one evaluation run.
TOKENS = {f"eval-token-{customer}-xxxxxxxxxxxx": {"subject_id": customer, "role": "customer"}
          for customer in ("demo-customer-101", "demo-customer-102", "demo-customer-103", "demo-customer-104")}
REGRESSION_TOLERANCE = 0.1  # pass rates may vary by this much between runs before it counts


class RecordingAgent:
    """Wraps the reply agent to keep what it was given and what it wrote, for checks and the judge."""

    def __init__(self, agent):
        self.agent = agent
        self.llm = getattr(agent, "llm", None)
        self.request = None
        self.draft_result = None
        self.failure = ""

    def reset(self) -> None:
        self.request, self.draft_result, self.failure = None, None, ""

    def draft(self, request):
        self.request = request
        try:
            self.draft_result = self.agent.draft(request)
        except AgentFailed as error:
            self.failure = str(error)
            raise
        return self.draft_result


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TurnResult:
    scenario: str
    run: int
    step: int
    message: str
    reply: str
    written_by_agent: bool
    fallback: str
    seconds: float
    checks: list[Check] = field(default_factory=list)
    judgement: Judgement | None = None


def _refund_draft(case) -> dict[str, int] | None:
    if not case.refunds:
        return None
    version = case.refunds[-1].current
    lines = {line.item_id: line.quantity for line in version.lines}
    if version.delivery_fee_minor:
        lines["delivery_fee"] = 1
    return lines


def _checks(step: Step, turn: dict, case, written_by_agent: bool) -> list[Check]:
    expect, reply = step.expect, turn["reply"]
    checks = []
    if expect.case_status is not None:
        checks.append(Check("case_status", turn["case_status"] == expect.case_status.value,
                            f"expected {expect.case_status.value}, got {turn['case_status']}"))
    if expect.needs_review is not None:
        checks.append(Check("needs_review", turn["needs_human_review"] is expect.needs_review,
                            f"expected {expect.needs_review}, got {turn['needs_human_review']}"))
    if expect.refund_draft is not None:
        actual = _refund_draft(case)
        checks.append(Check("refund_draft", actual in expect.refund_draft,
                            f"expected one of {expect.refund_draft}, got {actual}"))
    if expect.refund_count is not None:
        checks.append(Check("refund_count", len(case.refunds) == expect.refund_count,
                            f"expected {expect.refund_count}, got {len(case.refunds)}"))
    if expect.written_by_agent is not None:
        checks.append(Check("written_by_agent", written_by_agent is expect.written_by_agent,
                            f"expected {expect.written_by_agent}, got {written_by_agent}"))
    cited = {citation["article_id"] for citation in turn["citations"]}
    for article in expect.cites:
        checks.append(Check("cites", article in cited, f"expected a citation of {article}, got {sorted(cited)}"))
    for pattern in expect.must_match:
        checks.append(Check("must_match", re.search(pattern, reply) is not None, f"no match for {pattern!r}"))
    for pattern in expect.must_not_match:
        found = re.search(pattern, reply)
        checks.append(Check("must_not_match", found is None, f"{pattern!r} matched {found.group(0)!r}"
                            if found else ""))
    return checks


def _passage(citation: dict) -> str:
    for article in ARTICLES:
        if article.article_id == citation["article_id"] and article.version == citation["version"]:
            for section in article.sections:
                if section.section_id == citation["section"]:
                    return f"[{article.article_id}@{article.version}#{section.section_id}] {section.text}"
    return f"[{citation['article_id']}] (not found)"


def _material(agent: RecordingAgent, turn: dict) -> Material | None:
    request = agent.request
    if request is None:
        return None  # the agent was not asked, e.g. a redirect to another order
    history = [f"{message.role.value}: {message.text}" for message in request.history]
    if request.summary:
        history.insert(0, f"(summary of earlier messages) {request.summary}")
    traces = agent.draft_result.tool_traces if agent.draft_result else []
    actions = ", ".join(f"{trace.name}: {'ok' if trace.ok else trace.error}" for trace in traces) or "none"
    return Material(json.dumps(case_facts(request, POLICIES), indent=1), "\n".join(history) or "(none)",
                    "\n".join(_passage(citation) for citation in turn["citations"]) or "(none)", actions,
                    request.message, turn["reply"], request.reference_reply)


class Evaluation:
    def __init__(self, recognizer, reply_agent, judge=None):
        self.recognizer = recognizer
        self.agent = RecordingAgent(reply_agent) if reply_agent is not None else None
        self.judge = judge

    def run_scenario(self, scenario: Scenario, run: int = 1) -> list[TurnResult]:
        store = CaseStore()
        orders = InMemoryOrderSource(ORDERS)
        provider = SimulatedPaymentProvider(scenario.payment_faults)
        app = create_app(TokenRegistry(TOKENS), orders, store, self.recognizer, self.agent, provider)
        token = next(token for token, account in TOKENS.items() if account["subject_id"] == scenario.customer_id)
        case_id, results, skipped = None, [], []
        try:
            for number, step in enumerate(scenario.steps, start=1):
                if step.customer is None:
                    problem = self._act(step, store, orders, provider, case_id)
                    if problem:
                        skipped.append(Check("step", False, f"step {number} skipped: {problem}"))
                    continue
                if self.agent is not None:
                    self.agent.reset()
                body = {"message": step.customer,
                        **({"case_id": case_id} if case_id else
                           {"order_id": scenario.order_id} if scenario.order_id else {})}
                started = time.perf_counter()
                turn = _post(app, token, body)
                seconds = time.perf_counter() - started
                case_id = turn["case_id"]
                drafted = self.agent.draft_result if self.agent else None
                written = drafted is not None and drafted.reply == turn["reply"]
                fallback = "none" if written else (self.agent.failure if self.agent and self.agent.failure
                                                   else "rules")
                result = TurnResult(scenario.id, run, number, step.customer, turn["reply"], written, fallback,
                                    round(seconds, 2), skipped + _checks(step, turn, store.get(case_id), written))
                skipped = []
                material = _material(self.agent, turn) if self.agent else None
                if self.judge is not None and material is not None:
                    result.judgement = self.judge.judge(material)
                results.append(result)
        finally:
            if app.state.summaries is not None:
                app.state.summaries.close()
            store.close()
        return results

    @staticmethod
    def _act(step: Step, store: CaseStore, orders, provider, case_id: str) -> str:
        """Do a support agent or payment step; returns why it could not be done, if so."""
        if step.send_payments:
            send_due(store, provider)
        elif step.reply is not None:
            support_desk.reply(store, AGENT_ID, case_id, step.reply)
        elif step.support_agent == "resolve":
            support_desk.resolve(store, AGENT_ID, case_id, "Resolved during evaluation")
        else:
            refund = open_refund(store.get(case_id).refunds)
            if refund is None:
                return f"{step.support_agent}: the case has no refund proposal"
            version = len(refund.versions)
            if step.support_agent == "approve_refund":
                support_desk.approve_refund(store, orders, AGENT_ID, refund.refund_id, version)
            else:
                support_desk.reject_refund(store, AGENT_ID, refund.refund_id, version, "Rejected during evaluation")
        return ""

    def run(self, suite: Suite, repeat: int = 1, progress=None) -> list[TurnResult]:
        results = []
        for scenario in suite.scenarios:
            for run in range(1, repeat + 1):
                turns = self.run_scenario(scenario, run)
                results.extend(turns)
                if progress:
                    progress(scenario, run, turns)
        return results


def _post(app, token: str, body: dict) -> dict:
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval") as client:
            response = await client.post("/api/v1/chat", json=body, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.json()
    return asyncio.run(go())


# --- reports --------------------------------------------------------------------------------

def suite_id(scenarios_text: str, agent_model: str, judge_model: str | None) -> str:
    parts = [scenarios_text, agent_model, judge_model or "no judge"]
    if judge_model:
        parts += [judge_module.PROMPT, judge_module.EXAMPLES, json.dumps(judge_module.CRITERIA, sort_keys=True)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def summarize(results: Sequence[TurnResult]) -> dict:
    checks = [check for result in results for check in result.checks]
    judged = [result.judgement for result in results if result.judgement and not result.judgement.failed]
    metrics = {"checks_passed": _rate(sum(check.passed for check in checks), len(checks)),
               "written_by_agent": _rate(sum(result.written_by_agent for result in results), len(results))}
    for criterion in judge_module.CRITERIA:
        if judged:
            metrics[f"judge_{criterion}"] = _rate(sum(item.verdicts[criterion].passed for item in judged),
                                                  len(judged))
    seconds = sorted(result.seconds for result in results)
    return {
        "metrics": metrics,
        "counts": {"turns": len(results), "checks": len(checks), "judged": len(judged),
                   "judge_failures": sum(1 for result in results if result.judgement and result.judgement.failed)},
        "fallbacks": dict(Counter(result.fallback for result in results if not result.written_by_agent)),
        "seconds": {"mean": round(statistics.mean(seconds), 2) if seconds else 0,
                    "p95": seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))] if seconds else 0},
        "failed_checks": [{"scenario": result.scenario, "run": result.run, "step": result.step,
                           "check": check.name, "detail": check.detail, "reply": result.reply}
                          for result in results for check in result.checks if not check.passed],
        "failed_judgements": [{"scenario": result.scenario, "run": result.run, "step": result.step,
                               "criterion": name, "reason": verdict.reason, "reply": result.reply}
                              for result in results if result.judgement
                              for name, verdict in result.judgement.verdicts.items() if not verdict.passed],
    }


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 3) if total else 1.0


def compare(current: dict, baseline: dict | None) -> list[str]:
    """Metrics that fell by more than the tolerance since a baseline of the same suite."""
    if baseline is None or baseline.get("suite_id") != current["suite_id"]:
        return []
    regressions = []
    for name, value in current["summary"]["metrics"].items():
        before = baseline["summary"]["metrics"].get(name)
        if before is not None and value < before - REGRESSION_TOLERANCE:
            regressions.append(f"{name}: {before:.3f} -> {value:.3f}")
    return regressions


def report(results: Sequence[TurnResult], suite: str, models: dict, repeat: int) -> dict:
    return {"suite_id": suite, "models": models, "repeat": repeat,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": summarize(results),
            "turns": [{**asdict(result), "judgement": asdict(result.judgement) if result.judgement else None}
                      for result in results]}


def load_report(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
