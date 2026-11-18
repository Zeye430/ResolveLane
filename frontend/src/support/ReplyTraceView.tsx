import type { ModelUsage, ReplyTrace } from "../api/types";
import { count, humanize, issueLabel, usd } from "../shared/format";

// Why the agent's reply was replaced by the rule-based one. Reasons not listed (provider errors
// such as status_529 or APITimeoutError) are shown as they are.
const FALLBACKS: Record<string, string> = {
  circuit_open: "the model provider was paused after repeated failures",
  ungrounded_number: "it stated a number found neither in the case facts nor in a cited passage",
  cites_unretrieved_source: "it cited a help center passage it had not retrieved",
  reply_commits_to_an_outcome: "it seemed to promise a refund, credit or other outcome",
  reply_too_long: "it was longer than allowed",
  too_many_rounds: "it needed more model or tool calls than allowed",
  empty_reply: "it was empty",
  invalid_answer: "the model's answer was not in the expected format",
  invalid_sources: "its list of sources was not valid",
  case_changed: "another message changed the case while it was written",
  error: "an unexpected error",
};

const TIERS: Record<string, string> = {
  llm: "language model", remote_embedding: "remote embedding", local_embedding: "local embedding",
  keywords: "keywords", none: "no tier",
};

export function fallbackText(reason: string): string {
  return FALLBACKS[reason] ?? reason;
}

/** The model calls behind one reply, counting answers that were not used. */
export function UsageText({ usage }: { usage: ModelUsage | null }) {
  if (!usage) return <span className="muted">not recorded</span>;
  if (usage.calls === 0) return <>no model calls</>;
  const parts = [`${usage.calls} model call${usage.calls > 1 ? "s" : ""}`, `${count(usage.input_tokens)} in`,
                 `${count(usage.output_tokens)} out tokens`];
  if (usage.cache_read_tokens > 0) parts.push(`${count(usage.cache_read_tokens)} cached`);
  return (
    <>{parts.join(" · ")} · {usage.cost_usd === null
      ? <span className="muted">no price configured</span> : usd(usage.cost_usd)}</>
  );
}

/** How an assistant reply came about: who wrote it, why rules took over, what the agent used. */
export function ReplyTraceView({ trace }: { trace: ReplyTrace }) {
  const agent = trace.written_by === "agent";
  const fellBack = !agent && trace.fallback !== "none";
  return (
    <details className={`trace ${fellBack ? "trace-fallback" : ""}`}>
      <summary>
        <span className={`badge ${agent ? "badge-agent" : fellBack ? "badge-delayed" : ""}`}>
          {agent ? "Written by the AI agent" : fellBack ? "Rule-based fallback" : "Rule-based reply"}
        </span>
        {fellBack && <span> agent reply not used: {fallbackText(trace.fallback)}</span>}
        <span className="muted"> · {trace.seconds.toFixed(1)} s</span>
        {trace.repeat && <span className="badge badge-delayed">Repeats previous reply</span>}
      </summary>
      <dl className="facts">
        <dt>Recognized</dt>
        <dd>{trace.issues.map(issueLabel).join(", ") || "nothing"} by {TIERS[trace.recognition] ?? trace.recognition}</dd>
        {trace.model && <><dt>Model</dt><dd className="mono">{trace.model}{agent && ` · ${trace.rounds} calls`}</dd></>}
        {trace.tools.length > 0 && <><dt>Tools</dt><dd>
          <ul className="tools">
            {trace.tools.map((tool, index) => (
              <li key={index} className={tool.ok ? "" : "late"}>
                <span className="mono">{tool.name}</span> {tool.ok ? "ok" : `failed: ${humanize(tool.error)}`}
                <span className="muted"> · {Math.round(tool.latency_ms)} ms</span>
              </li>
            ))}
          </ul>
        </dd></>}
        {trace.skills.length > 0 && <><dt>Guides</dt><dd className="mono">{trace.skills.join(", ")}</dd></>}
        <dt>Usage</dt>
        <dd><UsageText usage={trace.usage ?? null} /></dd>
        {!agent && !fellBack && <><dt>Why rules</dt><dd>No language model is configured, or rules decide this reply.</dd></>}
      </dl>
    </details>
  );
}
