// How the evaluation page reads runs: which runs can be compared, each metric with its sample
// size and interval, and the replies grouped by scenario. Pure functions, tested on their own.
import type { EvalReport, EvalRun, EvalTurn } from "../api/types";
import { humanize } from "../shared/format";

type Denominator = "checks" | "turns" | "judged";

export const METRICS: { name: string; label: string; of: Denominator; help: string }[] = [
  { name: "checks_passed", label: "Code checks passed", of: "checks",
    help: "Expectations checked exactly by code: case status, refund draft, required and forbidden phrases." },
  { name: "written_by_agent", label: "Written by the AI agent", of: "turns",
    help: "The rest were rule-based replies, sent when the agent's reply failed a guard or the model was unavailable." },
  { name: "judge_grounded", label: "Grounded", of: "judged",
    help: "Judge: every factual claim is supported by the case facts, the conversation, cited passages or actions taken." },
  { name: "judge_answers", label: "Answers the message", of: "judged",
    help: "Judge: responds to what the customer's latest message asks or reports." },
  { name: "judge_focused", label: "Focused", of: "judged",
    help: "Judge: nothing the customer does not need, such as unasked policy terms or repeated details." },
  { name: "judge_no_false_promises", label: "No false promises", of: "judged",
    help: "Judge: no outcome promised or implied that the facts do not show has happened or been decided." },
];

/** Same run: the baseline file is a copy of one report's summary. */
export const sameRun = (a: EvalRun, b: EvalRun) => a.suite_id === b.suite_id && a.created_at === b.created_at;

export type Target = { id: string; label: string; run: EvalRun };

/** Runs this one can be compared with: the same suite (scenarios, judge instructions and models),
 * run earlier. The baseline comes first and is the default. */
export function comparisons(run: EvalRun, reports: EvalRun[], baseline: EvalRun | null | undefined): Target[] {
  const targets: Target[] = [];
  if (baseline && baseline.suite_id === run.suite_id && !sameRun(baseline, run)) {
    targets.push({ id: "baseline", label: "Baseline", run: baseline });
  }
  for (const other of reports) {
    if (targets[0]?.id === "baseline" && sameRun(targets[0].run, other)) continue; // already offered as the baseline
    if (other.suite_id === run.suite_id && other.created_at < run.created_at) {
      targets.push({ id: other.report_id, label: `Run of ${other.created_at.slice(0, 16).replace("T", " ")} UTC`,
                     run: other });
    }
  }
  return targets;
}

/** 95% Wilson score interval for a pass rate: honest at small samples and near 0% or 100%. */
export function wilson(passed: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = passed / n;
  const centre = p + (z * z) / (2 * n);
  const spread = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  const scale = 1 + (z * z) / n;
  return [Math.max(0, (centre - spread) / scale), Math.min(1, (centre + spread) / scale)];
}

export type Change = "regression" | "improved" | "steady";

export type MetricRow = {
  name: string; label: string; help: string;
  rate: number; n: number; passed: number; interval: [number, number];
  before?: number; change?: Change;
};

export function metricRows(run: EvalRun, against: EvalRun | undefined, tolerance: number): MetricRow[] {
  const known = METRICS.filter((metric) => metric.name in run.metrics);
  const others = Object.keys(run.metrics).filter((name) => !METRICS.some((metric) => metric.name === name))
    .map((name) => ({ name, label: humanize(name), of: "turns" as Denominator, help: "" }));
  return [...known, ...others].map((metric) => {
    const rate = run.metrics[metric.name];
    const n = run.counts[metric.of];
    const passed = Math.round(rate * n);
    const before = against?.metrics[metric.name];
    // The same rule as the command line's regression check, in the same arithmetic.
    const change: Change | undefined = before === undefined ? undefined
      : rate < before - tolerance ? "regression" : rate > before + tolerance ? "improved" : "steady";
    return { name: metric.name, label: metric.label, help: metric.help, rate, n, passed,
             interval: wilson(passed, n), before, change };
  });
}

export type ScenarioResult = {
  id: string; description?: string; turns: EvalTurn[];
  failedChecks: number; failedVerdicts: number; judgeErrors: number; ruleBased: number;
};

export const turnFailed = (turn: EvalTurn) =>
  turn.checks.some((check) => !check.passed) || Boolean(turn.judgement?.error)
  || Object.values(turn.judgement?.verdicts ?? {}).some((verdict) => !verdict.passed);

/** Replies grouped by scenario, in the order they ran, scenarios with failures first. */
export function byScenario(report: EvalReport): ScenarioResult[] {
  const descriptions = new Map(report.scenarios.map((item) => [item.id, item.description]));
  const groups = new Map<string, ScenarioResult>();
  for (const turn of report.turns) {
    let group = groups.get(turn.scenario);
    if (!group) {
      group = { id: turn.scenario, description: descriptions.get(turn.scenario), turns: [],
                failedChecks: 0, failedVerdicts: 0, judgeErrors: 0, ruleBased: 0 };
      groups.set(turn.scenario, group);
    }
    group.turns.push(turn);
    group.failedChecks += turn.checks.filter((check) => !check.passed).length;
    group.failedVerdicts += Object.values(turn.judgement?.verdicts ?? {}).filter((verdict) => !verdict.passed).length;
    group.judgeErrors += turn.judgement?.error ? 1 : 0;
    group.ruleBased += turn.written_by_agent ? 0 : 1;
  }
  const failures = (group: ScenarioResult) => group.failedChecks + group.failedVerdicts + group.judgeErrors;
  return [...groups.values()].sort((a, b) => Number(failures(b) > 0) - Number(failures(a) > 0));
}

export const percent = (rate: number) => `${Math.round(rate * 100)}%`;

/** A change in percentage points, signed. */
export function points(rate: number, before: number): string {
  const change = Math.round((rate - before) * 100);
  return change === 0 ? "±0 pts" : `${change > 0 ? "+" : "−"}${Math.abs(change)} pts`;
}
