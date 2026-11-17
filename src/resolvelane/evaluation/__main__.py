"""Run the evaluation suite against the configured language model.

    python -m resolvelane.evaluation                    # every scenario once, with the judge
    python -m resolvelane.evaluation --repeat 3         # each scenario three times
    python -m resolvelane.evaluation --only refund-flow --no-judge
    python -m resolvelane.evaluation --save-baseline    # make this run the baseline
    python -m resolvelane.evaluation --calibrate        # measure the judge against human labels

Needs ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL / ANTHROPIC_MODEL as for the server). The
judge uses RESOLVELANE_JUDGE_MODEL, by default the same model; a judge of the same model may
favour its own writing, so a different, stronger judge model is better when one is available.
Reports are written to evals/reports/ (ignored by git). The exit status is 1 when a metric
fell below the baseline by more than the tolerance, so the suite can gate a change.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set before resolvelane.api is imported: importing it builds the server's application, which
# must not open the development database or send its payments during an evaluation.
os.environ["RESOLVELANE_DATABASE"] = ":memory:"
os.environ["RESOLVELANE_PAYMENTS"] = "paused"  # scenarios send payments themselves
os.environ.pop("RESOLVELANE_MCP_CONFIG", None)  # scenarios do not depend on external services

import anthropic  # noqa: E402

from resolvelane.evaluation.judge import CRITERIA, Judge, Material  # noqa: E402
from resolvelane.evaluation.runner import Evaluation, compare, load_report, report, suite_id  # noqa: E402
from resolvelane.evaluation.scenarios import DEFAULT_SCENARIOS, Suite, load_suite  # noqa: E402
from resolvelane.knowledge import HelpCenter  # noqa: E402
from resolvelane.llm import DEFAULT_BASE_URL, load_llm_provider  # noqa: E402
from resolvelane.recognition import load_local_embedder, load_recognizer  # noqa: E402
from resolvelane.reply_agent import load_reply_agent  # noqa: E402
from resolvelane.simulator.fixtures import POLICIES  # noqa: E402
from resolvelane.simulator.help_center import ARTICLES  # noqa: E402
from resolvelane.skills import load_skills  # noqa: E402
from resolvelane.tools import ToolGateway  # noqa: E402

EVALS = DEFAULT_SCENARIOS.parent
BASELINE = EVALS / "baseline.json"
CALIBRATION = EVALS / "judge_calibration.json"


def _judge(environ) -> Judge:
    client = anthropic.Anthropic(api_key=environ["ANTHROPIC_API_KEY"],
                                 base_url=environ.get("ANTHROPIC_BASE_URL", "").strip() or DEFAULT_BASE_URL,
                                 timeout=60, max_retries=2)
    model = (environ.get("RESOLVELANE_JUDGE_MODEL", "").strip() or environ.get("ANTHROPIC_MODEL", "").strip()
             or "claude-opus-5-5")
    return Judge(client, model)


def calibrate(judge: Judge, path: Path = CALIBRATION) -> int:
    items = json.loads(path.read_text())["items"]
    agree = {name: 0 for name in CRITERIA}
    false_passes = {name: 0 for name in CRITERIA}
    judged = 0
    for item in items:
        judgement = judge.judge(Material(**item["material"]))
        if judgement.failed:
            print(f"  {item['id']}: judge failed ({judgement.error})")
            continue
        judged += 1
        for name, label in item["labels"].items():
            verdict = judgement.verdicts[name]
            if verdict.passed == label:
                agree[name] += 1
            else:
                false_passes[name] += verdict.passed and not label
                print(f"  {item['id']} {name}: labeled {'pass' if label else 'fail'}, judge "
                      f"{'pass' if verdict.passed else 'fail'}: {verdict.reason}")
    print(f"\nJudge {judge.model} on {judged} of {len(items)} labeled replies:")
    for name in CRITERIA:
        print(f"  {name:<18} agreement {agree[name] / judged if judged else 0:.0%}"
              f"   false passes {false_passes[name]}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m resolvelane.evaluation")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these scenarios")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args(argv)
    logging.getLogger("resolvelane").setLevel(logging.WARNING)  # the report is the output, not request logs

    environ = os.environ
    if not environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set; the evaluation needs the language model.", file=sys.stderr)
        return 2
    if args.calibrate:
        return calibrate(_judge(environ))

    text = args.scenarios.read_text()
    suite = load_suite(args.scenarios)
    if args.only:
        unknown = set(args.only) - {scenario.id for scenario in suite.scenarios}
        if unknown:
            print(f"Unknown scenarios: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        suite = Suite(scenarios=[scenario for scenario in suite.scenarios if scenario.id in args.only])

    llm = load_llm_provider(environ)
    embedder = load_local_embedder(environ)
    recognizer = load_recognizer(environ, llm=llm, local_embedder=embedder)
    agent = load_reply_agent(llm, POLICIES, HelpCenter(ARTICLES, embedder), load_skills(environ), ToolGateway())
    judge = None if args.no_judge else _judge(environ)
    evaluation = Evaluation(recognizer, agent, judge)

    def progress(scenario, run, turns):
        failed = sum(not check.passed for turn in turns for check in turn.checks)
        verdicts = [verdict for turn in turns if turn.judgement for verdict in turn.judgement.verdicts.values()]
        print(f"{scenario.id} run {run}: {len(turns)} replies, {failed} failed checks"
              + (f", {sum(not v.passed for v in verdicts)} failed judgements" if judge else ""))

    results = evaluation.run(suite, args.repeat, progress)
    models = {"agent": llm.model, "judge": judge.model if judge else None}
    full_suite = not args.only and args.scenarios == DEFAULT_SCENARIOS
    current = report(results, suite_id(text if full_suite else json.dumps(suite.model_dump(mode="json")),
                                       llm.model, models["judge"]), models, args.repeat)
    summary = current["summary"]

    print("\nMetrics:")
    for name, value in summary["metrics"].items():
        print(f"  {name:<26} {value:.0%}")
    print(f"  replies {summary['counts']['turns']}, judge failures {summary['counts']['judge_failures']}, "
          f"seconds per reply mean {summary['seconds']['mean']} p95 {summary['seconds']['p95']}")
    if summary["fallbacks"]:
        print(f"  rule-based replies: {summary['fallbacks']}")
    for item in summary["failed_checks"]:
        print(f"\nFAILED CHECK {item['scenario']} step {item['step']} {item['check']}: {item['detail']}\n"
              f"  reply: {item['reply']}")
    for item in summary["failed_judgements"]:
        print(f"\nFAILED JUDGEMENT {item['scenario']} step {item['step']} {item['criterion']}: {item['reason']}\n"
              f"  reply: {item['reply']}")

    reports = EVALS / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / f"{current['created_at'].replace(':', '')}.json"
    path.write_text(json.dumps(current, indent=1))
    print(f"\nReport: {path}")

    baseline = load_report(BASELINE)
    regressions = compare(current, baseline)
    if baseline is None or baseline.get("suite_id") != current["suite_id"]:
        print("No baseline for this suite, models and judge; nothing to compare.")
    elif regressions:
        print("REGRESSIONS against the baseline:\n  " + "\n  ".join(regressions))
    else:
        print("No regressions against the baseline.")
    if args.save_baseline:
        if not full_suite:
            print("A baseline is saved only from the full suite.", file=sys.stderr)
            return 2
        BASELINE.write_text(json.dumps({key: current[key] for key in ("suite_id", "models", "repeat",
                                                                       "created_at", "summary")}, indent=1))
        print(f"Saved as the baseline: {BASELINE}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
