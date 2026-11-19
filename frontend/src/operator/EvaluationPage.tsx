import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { Api } from "../api/client";
import type { EvalReport, EvalRun, EvalTurn } from "../api/types";
import { count, humanize, time, usd } from "../shared/format";
import { totalTokens } from "../support/usage";
import { ErrorNote, Loading, Panel } from "../shared/ui";
import {
  byScenario, comparisons, metricRows, percent, points, sameRun, turnFailed, type ScenarioResult, type Target,
} from "./evaluation";

const RUN_COMMAND = "python -m resolvelane.evaluation";

export function EvaluationPage({ api }: { api: Api }) {
  const list = useQuery({ queryKey: ["evaluation-reports"], queryFn: api.evaluationReports });
  const [chosen, setChosen] = useState<string | null>(null);

  if (list.isPending) return <main className="main"><Loading what="evaluation runs" /></main>;
  if (list.error) return <main className="main"><ErrorNote error={list.error} onRetry={() => list.refetch()} /></main>;
  const { reports, baseline, tolerance, unreadable } = list.data;
  const selected = reports.find((run) => run.report_id === chosen) ?? reports[0];

  return (
    <div className="layout layout-operator">
      <aside className="sidebar">
        <Panel title="Evaluation runs" aside={<button onClick={() => list.refetch()}>Refresh</button>}>
          {reports.length === 0 && (
            <p className="muted">
              No saved runs yet. Run <code>{RUN_COMMAND}</code> on the server; each run is saved to evals/reports/.
            </p>
          )}
          <ul className="list">
            {reports.map((run) => (
              <li key={run.report_id}>
                <button className={`list-item ${run === selected ? "selected" : ""}`} aria-current={run === selected}
                        onClick={() => setChosen(run.report_id)}>
                  <span className="row"><span>{time(run.created_at)}</span>
                    {baseline && sameRun(baseline, run) && <span className="badge badge-agent">Baseline</span>}</span>
                  <span className="muted">{run.models.agent} · {run.counts.turns} replies</span>
                  <span className="mono muted" title="Suite: scenarios, judge instructions and models">
                    suite {run.suite_id.slice(0, 8)}</span>
                </button>
              </li>
            ))}
          </ul>
          {unreadable > 0 && <p className="error">{unreadable} report file{unreadable > 1 ? "s" : ""} could not be read.</p>}
        </Panel>
        <p className="hint">Live operational metrics are at <code>/metrics</code> in the Prometheus format.</p>
      </aside>
      {selected ? <RunDetail key={selected.report_id} api={api} run={selected}
                             targets={comparisons(selected, reports, baseline)} tolerance={tolerance} />
        : <main className="main"><p className="muted empty">
            {baseline ? "A baseline is saved, but no runs are. " : ""}Runs appear here after the evaluation is run.
          </p></main>}
    </div>
  );
}

function RunDetail({ api, run, targets, tolerance }: {
  api: Api; run: EvalRun; targets: Target[]; tolerance: number;
}) {
  const [targetId, setTargetId] = useState(targets[0]?.id ?? "");
  const target = targets.find((item) => item.id === targetId);
  const report = useQuery({ queryKey: ["evaluation-report", run.report_id],
                            queryFn: () => api.evaluationReport(run.report_id) });
  const rows = metricRows(run, target?.run, tolerance);
  const regressions = rows.filter((row) => row.change === "regression").length;

  return (
    <main className="main">
      <Panel title="Run" aside={target && (
        <span className={`badge ${regressions ? "badge-rejected" : "badge-approved"}`}>
          {regressions ? `${regressions} regression${regressions > 1 ? "s" : ""}` : "No regressions"}
        </span>
      )}>
        <dl className="facts">
          <dt>Started</dt><dd>{time(run.created_at)}</dd>
          <dt>Suite</dt><dd><span className="mono">{run.suite_id}</span> <span className="hint">scenarios, judge instructions and models</span></dd>
          <dt>Models</dt><dd><Models models={run.models} /></dd>
          <dt>Code</dt><dd>{run.commit ? <span className="mono">{run.commit}</span> : <span className="muted">commit not recorded</span>}</dd>
          <dt>Sample</dt>
          <dd>{run.counts.turns} replies{run.repeat > 1 ? ` (each scenario ${run.repeat} times)` : ""} · {run.counts.checks} code checks · {run.counts.judged} judged
            {run.counts.judge_failures > 0 && <span className="late"> · {run.counts.judge_failures} not judged (judge error)</span>}</dd>
          <dt>Time per reply</dt><dd>mean {run.seconds.mean} s · p95 {run.seconds.p95} s <span className="hint">of {run.counts.turns} replies</span></dd>
          <dt>Model usage</dt>
          <dd>{run.usage ? <RunUsage usage={run.usage} /> : <span className="muted">not recorded</span>}</dd>
          <dt>Rule-based</dt>
          <dd>{Object.keys(run.fallbacks).length === 0 ? "none"
            : Object.entries(run.fallbacks).map(([reason, count]) => `${count} × ${humanize(reason)}`).join(", ")}</dd>
        </dl>
      </Panel>

      <Panel title="Metrics" aside={targets.length > 0 && (
        <label className="hint">Compare with{" "}
          <select value={targetId} onChange={(event) => setTargetId(event.target.value)}>
            {targets.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            <option value="">Nothing</option>
          </select>
        </label>
      )}>
        {targets.length === 0 && (
          <p className="notice">No earlier run of this suite to compare with. Runs with other scenarios, judge
            instructions or models are not compared: their differences would not be the code's.</p>
        )}
        <div className="table-scroll">
          <table className="table metrics">
            <thead>
              <tr><th>Metric</th><th>This run</th><th>95% interval</th>
                {target && <><th>{target.label}</th><th>Change</th></>}</tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name}>
                  <td><span title={row.help}>{row.label}</span></td>
                  <td><strong>{percent(row.rate)}</strong> <span className="hint">{row.passed}/{row.n}</span></td>
                  <td className="muted">{row.n ? `${percent(row.interval[0])}–${percent(row.interval[1])}` : "no sample"}</td>
                  {target && (
                    <>
                      <td>{row.before === undefined ? <span className="muted">—</span> : percent(row.before)}</td>
                      <td>{row.before !== undefined && (
                        <span className={`change change-${row.change}`}>
                          {points(row.rate, row.before)}{row.change === "regression" ? " regression"
                            : row.change === "improved" ? " improved" : ""}
                        </span>
                      )}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hint">
          A pass rate that falls by more than {Math.round(tolerance * 100)} points is a regression, the rule the
          command line uses to fail a run. The interval shows how far a rate from this many replies can move by
          chance: changes inside it need more runs (<code>--repeat</code>) before they mean much.
        </p>
      </Panel>

      {report.isPending ? <Loading what="replies" />
        : report.error ? <ErrorNote error={report.error} onRetry={() => report.refetch()} />
          : <Scenarios report={report.data} />}
    </main>
  );
}

function Scenarios({ report }: { report: EvalReport }) {
  const [onlyFailures, setOnlyFailures] = useState(false);
  const scenarios = byScenario(report);
  const failing = scenarios.filter((item) => item.turns.some(turnFailed));
  const shown = onlyFailures ? failing : scenarios;
  return (
    <Panel title={`Scenarios (${failing.length} of ${scenarios.length} with failures)`} aside={
      <label className="hint"><input type="checkbox" className="checkbox" checked={onlyFailures}
                                     onChange={(event) => setOnlyFailures(event.target.checked)} /> Only with failures</label>
    }>
      {shown.length === 0 && <p className="muted">No failures in this run.</p>}
      <ul className="list">
        {shown.map((scenario) => <li key={scenario.id}><ScenarioView scenario={scenario} /></li>)}
      </ul>
    </Panel>
  );
}

function ScenarioView({ scenario }: { scenario: ScenarioResult }) {
  const failures = scenario.failedChecks + scenario.failedVerdicts + scenario.judgeErrors;
  return (
    <details className="scenario">
      <summary>
        <span className="mono">{scenario.id}</span>
        {failures === 0 ? <span className="badge badge-approved">Passed</span> : (
          <>
            {scenario.failedChecks > 0 && <span className="badge badge-rejected">{scenario.failedChecks} failed check{scenario.failedChecks > 1 ? "s" : ""}</span>}
            {scenario.failedVerdicts > 0 && <span className="badge badge-proposed">{scenario.failedVerdicts} judge fail{scenario.failedVerdicts > 1 ? "s" : ""}</span>}
            {scenario.judgeErrors > 0 && <span className="badge badge-proposed">{scenario.judgeErrors} not judged</span>}
          </>
        )}
        {scenario.ruleBased > 0 && <span className="badge">{scenario.ruleBased} rule-based</span>}
        <span className="hint">{scenario.turns.length} repl{scenario.turns.length > 1 ? "ies" : "y"}</span>
      </summary>
      {scenario.description && <p className="muted">{scenario.description}</p>}
      <ol className="turns">
        {scenario.turns.map((turn) => <TurnView key={`${turn.run}-${turn.step}`} turn={turn} />)}
      </ol>
    </details>
  );
}

function TurnView({ turn }: { turn: EvalTurn }) {
  const verdicts = Object.entries(turn.judgement?.verdicts ?? {});
  return (
    <li className={`turn ${turnFailed(turn) ? "turn-failed" : ""}`}>
      <p className="hint">Step {turn.step}{turn.run > 1 ? `, run ${turn.run}` : ""} · {turn.seconds} s ·{" "}
        {turn.written_by_agent ? "written by the AI agent" : `rule-based (${humanize(turn.fallback)})`}
        {turn.usage && ` · ${count(totalTokens(turn.usage))} tokens`}</p>
      <p><strong>Customer:</strong> {turn.message}</p>
      <p className="reply"><strong>Reply:</strong> {turn.reply}</p>
      <ul className="verdicts">
        {turn.checks.map((check, index) => (
          <li key={`check-${index}`} className={check.passed ? "" : "failed"}>
            <span className={`badge ${check.passed ? "badge-approved" : "badge-rejected"}`}>{check.passed ? "Pass" : "Fail"}</span>
            <span>check <span className="mono">{check.name}</span>{check.detail && <span className="muted"> · {check.detail}</span>}</span>
          </li>
        ))}
        {verdicts.map(([criterion, verdict]) => (
          <li key={criterion} className={verdict.passed ? "" : "failed"}>
            <span className={`badge ${verdict.passed ? "badge-approved" : "badge-proposed"}`}>{verdict.passed ? "Pass" : "Fail"}</span>
            <span>judge <strong>{humanize(criterion)}</strong><span className="muted"> · {verdict.reason}</span></span>
          </li>
        ))}
        {turn.judgement?.error && <li className="failed"><span className="badge badge-proposed">Not judged</span> {turn.judgement.error}</li>}
      </ul>
    </li>
  );
}

/** The application's model usage per reply and per case, and the judge's apart: a production
 * system makes the first, never the second. */
function RunUsage({ usage }: { usage: NonNullable<EvalRun["usage"]> }) {
  const { app, judge } = usage;
  const judgeTokens = judge.input_tokens + judge.output_tokens + judge.cache_read_tokens + judge.cache_write_tokens;
  return (
    <>
      {count(app.tokens_per_reply)} tokens per reply ·{" "}
      {app.cost_per_reply_usd === null || app.cost_per_case_usd === null
        ? <span className="muted">no price configured</span>
        : <>{usd(app.cost_per_reply_usd)} per reply · {usd(app.cost_per_case_usd)} per case</>}
      <span className="hint"> · {app.calls} model calls in all</span>
      {judge.calls > 0 && (
        <span className="hint"> · judge {count(judgeTokens)} tokens
          {judge.cost_usd === null ? "" : `, ${usd(judge.cost_usd)}`}</span>
      )}
    </>
  );
}

/** The agent and judge models with their providers. A judge from the agent's own provider is
 * flagged: a model family may favour its own writing. */
function Models({ models }: { models: EvalRun["models"] }) {
  const at = (provider?: string | null) => (provider ? <span className="hint"> at {provider}</span> : null);
  const sameProvider = models.judge !== null && Boolean(models.agent_provider)
    && models.agent_provider === models.judge_provider;
  return (
    <>
      agent <span className="mono">{models.agent}</span>{at(models.agent_provider)} · judge{" "}
      {models.judge === null ? "none (no judge metrics)" : <><span className="mono">{models.judge}</span>{at(models.judge_provider)}</>}
      {sameProvider && <> <span className="badge badge-proposed" title="A model family may favour its own writing">
        Judge from the agent&apos;s provider</span></>}
    </>
  );
}

