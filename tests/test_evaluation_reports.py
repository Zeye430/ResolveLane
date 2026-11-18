import asyncio
import json

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.evaluation.reports import EVALS, REGRESSION_TOLERANCE, ReportStore

# Invented test credentials for fictional accounts.
OPERATOR = "test-token-operator-501-aaaaaaaa"
AGENT = "test-token-agent-301-bbbbbbbbbbbb"
CUSTOMER = "test-token-customer-101-cccccccc"
TOKENS = {OPERATOR: {"subject_id": "demo-operator-501", "role": "operator"},
          AGENT: {"subject_id": "demo-agent-301", "role": "support_agent"},
          CUSTOMER: {"subject_id": "demo-customer-101", "role": "customer"}}


def saved_report(created_at, suite="suite-a", focused=0.5, **extra):
    return {
        "suite_id": suite, "models": {"agent": "agent-model", "judge": "judge-model"}, "repeat": 1,
        "created_at": created_at,
        "summary": {"metrics": {"checks_passed": 1.0, "judge_focused": focused},
                    "counts": {"turns": 1, "checks": 2, "judged": 1, "judge_failures": 0},
                    "fallbacks": {}, "seconds": {"mean": 3.2, "p95": 3.2},
                    "failed_checks": [], "failed_judgements": []},
        "turns": [{"scenario": "late-delivery", "run": 1, "step": 1, "message": "Late?", "reply": "Late by 15 minutes.",
                   "written_by_agent": True, "fallback": "none", "seconds": 3.2,
                   "checks": [{"name": "must_match", "passed": True, "detail": "15"}],
                   "judgement": {"verdicts": {"focused": {"passed": False, "reason": "Adds a policy."}}, "error": ""}}],
        **extra,
    }


@pytest.fixture
def evals(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-01-02T100000Z.json").write_text(json.dumps(saved_report("2026-01-02T10:00:00Z", focused=0.25)))
    (reports / "2026-01-03T100000Z.json").write_text(json.dumps(saved_report(
        "2026-01-03T10:00:00Z", commit="abc1234", scenarios=[{"id": "late-delivery", "description": "Late."}])))
    baseline = saved_report("2026-01-02T10:00:00Z", focused=0.25)
    del baseline["turns"]
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    return tmp_path


def get(app, path, token):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(path, headers={"Authorization": f"Bearer {token}"} if token else {})
    return asyncio.run(send())


def app_for(directory):
    return create_app(token_registry=TokenRegistry(TOKENS), evaluation_reports=ReportStore(directory))


def test_operator_lists_runs_newest_first_with_the_baseline(evals):
    response = get(app_for(evals), "/api/v1/evaluation/reports", OPERATOR)
    assert response.status_code == 200
    body = response.json()
    assert [run["report_id"] for run in body["reports"]] == ["2026-01-03T100000Z", "2026-01-02T100000Z"]
    newest = body["reports"][0]
    assert newest["commit"] == "abc1234" and newest["metrics"]["judge_focused"] == 0.5
    assert newest["counts"]["judged"] == 1 and "turns" not in newest  # the list carries no replies
    assert body["baseline"]["report_id"] == "baseline" and body["baseline"]["metrics"]["judge_focused"] == 0.25
    assert body["tolerance"] == REGRESSION_TOLERANCE and body["unreadable"] == 0


def test_operator_reads_one_run_with_replies_and_reasons(evals):
    body = get(app_for(evals), "/api/v1/evaluation/reports/2026-01-03T100000Z", OPERATOR).json()
    turn = body["turns"][0]
    assert turn["reply"] == "Late by 15 minutes." and turn["checks"][0]["passed"]
    assert turn["judgement"]["verdicts"]["focused"] == {"passed": False, "reason": "Adds a policy."}
    assert body["scenarios"] == [{"id": "late-delivery", "description": "Late."}]


def test_a_run_saved_before_commits_and_descriptions_were_recorded_still_reads(evals):
    body = get(app_for(evals), "/api/v1/evaluation/reports/2026-01-02T100000Z", OPERATOR).json()
    assert body["commit"] is None and body["scenarios"] == []


@pytest.mark.parametrize("token,status", [(AGENT, 403), (CUSTOMER, 403), (None, 401)])
def test_reports_are_for_operators_only(evals, token, status):
    app = app_for(evals)
    assert get(app, "/api/v1/evaluation/reports", token).status_code == status
    assert get(app, "/api/v1/evaluation/reports/2026-01-03T100000Z", token).status_code == status


@pytest.mark.parametrize("report_id,status", [("2026-01-01T100000Z", 404), ("..%2Fbaseline", 404),
                                              ("baseline", 422), ("2026-01-03T100000Z.json", 422)])
def test_only_saved_reports_can_be_read(evals, report_id, status):
    assert get(app_for(evals), f"/api/v1/evaluation/reports/{report_id}", OPERATOR).status_code == status


def test_unreadable_reports_are_counted_and_other_files_ignored(evals):
    (evals / "reports" / "2026-01-04T100000Z.json").write_text("{not json")
    (evals / "reports" / "2026-01-05T100000Z.json").write_text(json.dumps({"suite_id": "x"}))
    (evals / "reports" / "notes.json").write_text("{}")
    app = app_for(evals)
    body = get(app, "/api/v1/evaluation/reports", OPERATOR).json()
    assert len(body["reports"]) == 2 and body["unreadable"] == 2
    assert get(app, "/api/v1/evaluation/reports/2026-01-04T100000Z", OPERATOR).status_code == 404


def test_no_reports_yet(tmp_path):
    body = get(app_for(tmp_path), "/api/v1/evaluation/reports", OPERATOR).json()
    assert body["reports"] == [] and body["baseline"] is None and body["unreadable"] == 0


def test_the_committed_baseline_is_readable():
    baseline = ReportStore(EVALS).baseline()
    assert baseline is not None and baseline.counts.turns > 0
