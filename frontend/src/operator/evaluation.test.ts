import { describe, expect, it } from "vitest";
import { byScenario, comparisons, metricRows, points, wilson } from "./evaluation";
import { report, run } from "./fixtures";

describe("wilson", () => {
  it("gives a wide interval for a small sample and stays inside 0 to 1", () => {
    const [low, high] = wilson(14, 22);
    expect(low).toBeCloseTo(0.430, 2);
    expect(high).toBeCloseTo(0.803, 2);
    expect(wilson(22, 22)[1]).toBe(1);
    expect(wilson(22, 22)[0]).toBeCloseTo(0.851, 2);
    expect(wilson(0, 0)).toEqual([0, 1]);
  });
});

describe("comparisons", () => {
  const baseline = run("2026-01-02T10:00:00Z", { judge_focused: 0.5 });
  const earlier = run("2026-01-02T10:00:00Z", { judge_focused: 0.5 });
  const other = run("2026-01-01T10:00:00Z", { judge_focused: 0.2 }, "suite-b");
  const latest = run("2026-01-03T10:00:00Z", { judge_focused: 0.6 });

  it("offers the baseline first, then other earlier runs of the same suite", () => {
    const first = run("2026-01-01T09:00:00Z", { judge_focused: 0.3 });
    const targets = comparisons(latest, [latest, earlier, other, first], baseline);
    expect(targets.map((target) => target.id)).toEqual(["baseline", "2026-01-01T090000Z"]);
  });

  it("does not compare a run with itself, later runs or another suite", () => {
    expect(comparisons(earlier, [latest, earlier, other], baseline)).toEqual([]);
    expect(comparisons(other, [latest, earlier, other], baseline)).toEqual([]);
  });
});

describe("metricRows", () => {
  it("uses the command line's tolerance and each metric's own sample size", () => {
    const before = run("2026-01-01T10:00:00Z", { checks_passed: 1, judge_focused: 0.227, judge_grounded: 1,
                                                 judge_answers: 1 });
    const now = run("2026-01-02T10:00:00Z", { checks_passed: 1, judge_focused: 0.636, judge_grounded: 0.864,
                                              judge_answers: 0.955 });
    const rows = Object.fromEntries(metricRows(now, before, 0.1).map((row) => [row.name, row]));
    expect(rows.judge_focused).toMatchObject({ change: "improved", passed: 14, n: 22 });
    expect(rows.judge_grounded.change).toBe("regression");
    expect(rows.judge_answers.change).toBe("steady");
    expect(rows.checks_passed).toMatchObject({ n: 79, passed: 79, change: "steady" });
    expect(metricRows(now, undefined, 0.1)[0].change).toBeUndefined();
  });

  it("formats changes in percentage points", () => {
    expect(points(0.636, 0.227)).toBe("+41 pts");
    expect(points(0.955, 1)).toBe("−5 pts");
    expect(points(1, 1)).toBe("±0 pts");
  });
});

describe("byScenario", () => {
  it("groups replies, counts failures and puts failing scenarios first", () => {
    const groups = byScenario(report(run("2026-01-02T10:00:00Z", {})));
    expect(groups.map((group) => group.id)).toEqual(["late-delivery", "delivery-status"]);
    expect(groups[0]).toMatchObject({ failedVerdicts: 1, failedChecks: 0, description: "A late delivery is explained." });
    expect(groups[0].turns).toHaveLength(2);
    expect(groups[1].description).toBeUndefined();
  });
});
