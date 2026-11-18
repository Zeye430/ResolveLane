// Invented evaluation runs for tests.
import type { EvalReport, EvalRun } from "../api/types";

export function run(created_at: string, metrics: Record<string, number>, suite_id = "suite-a"): EvalRun {
  return {
    report_id: created_at.replace(/:/g, ""), suite_id, created_at, repeat: 1, commit: null,
    models: { agent: "agent-model", judge: "judge-model" }, metrics,
    counts: { turns: 22, checks: 79, judged: 22, judge_failures: 0 }, fallbacks: {}, seconds: { mean: 4.5, p95: 6.6 },
  };
}

export function report(base: EvalRun): EvalReport {
  const turn = (scenario: string, step: number, focused: boolean) => ({
    scenario, run: 1, step, message: "Where is my order?", reply: "It is on its way.", written_by_agent: true,
    fallback: "none", seconds: 3.1, checks: [{ name: "case_status", passed: true, detail: "expected open, got open" }],
    judgement: { verdicts: { focused: { passed: focused, reason: focused ? "Only what was asked." : "Adds a fee rule." } },
                 error: "" },
  });
  return { ...base, scenarios: [{ id: "late-delivery", description: "A late delivery is explained." }],
           turns: [turn("delivery-status", 1, true), turn("late-delivery", 1, true), turn("late-delivery", 2, false)] };
}
