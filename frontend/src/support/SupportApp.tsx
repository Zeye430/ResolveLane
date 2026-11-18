import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { Api } from "../api/client";
import type { CaseStatus, SupportCase } from "../api/types";
import { humanize, issueLabel, shortId, time, waited } from "../shared/format";
import { Composer, Conversation, ErrorNote, Loading, Panel, StatusBadge } from "../shared/ui";
import { OrderFacts } from "./OrderFacts";
import { RefundCard } from "./RefundCard";
import { ReplyTraceView } from "./ReplyTraceView";
import { useRetryKey } from "../shared/useRetryKey";

const STATUSES: CaseStatus[] = ["needs_review", "open", "resolved"];
const POLL_MS = 4000;

export function SupportApp({ api }: { api: Api }) {
  const [status, setStatus] = useState<CaseStatus>("needs_review");
  const [caseId, setCaseId] = useState<string | null>(null);
  const queue = useQuery({ queryKey: ["queue", status], queryFn: () => api.queue(status), refetchInterval: POLL_MS });

  return (
    <div className="layout layout-support">
      <aside className="sidebar">
        <Panel title="Queue">
          <div className="tabs" role="tablist" aria-label="Case status">
            {STATUSES.map((item) => (
              <button key={item} role="tab" aria-selected={item === status} className={item === status ? "active" : ""}
                      onClick={() => setStatus(item)}>{humanize(item)}</button>
            ))}
          </div>
          {queue.isPending ? <Loading what="queue" /> : <ErrorNote error={queue.error} onRetry={() => queue.refetch()} />}
          {queue.data?.cases.length === 0 && <p className="muted">No cases.</p>}
          <ul className="list">
            {queue.data?.cases.map((entry) => (
              <li key={entry.case_id}>
                <button className={`list-item ${entry.case_id === caseId ? "selected" : ""}`}
                        aria-current={entry.case_id === caseId} onClick={() => setCaseId(entry.case_id)}>
                  <span className="row"><span className="mono" title={entry.case_id}>{shortId(entry.case_id)}</span>
                    <span className="muted">{waited(entry.waiting_since)}</span></span>
                  <span>{entry.issues.map(issueLabel).join(", ") || "No issue recognized"}</span>
                  <span className="muted">{entry.customer_id} · {entry.message_count} messages</span>
                </button>
              </li>
            ))}
          </ul>
          {queue.data?.next_cursor && <p className="muted">Showing the 50 longest waiting.</p>}
        </Panel>
      </aside>
      {caseId ? <CaseWorkspace api={api} caseId={caseId} /> : (
        <main className="main"><p className="muted empty">Select a case from the queue.</p></main>
      )}
    </div>
  );
}

function CaseWorkspace({ api, caseId }: { api: Api; caseId: string }) {
  const client = useQueryClient();
  const detail = useQuery({ queryKey: ["support-case", caseId], queryFn: () => api.supportCase(caseId),
                            refetchInterval: POLL_MS });
  const refresh = () => {
    client.invalidateQueries({ queryKey: ["support-case", caseId] });
    client.invalidateQueries({ queryKey: ["queue"] });
  };
  const keys = useRetryKey();
  const reply = useMutation({
    mutationFn: (text: string) => api.reply(caseId, text, keys.keyFor({ caseId, text })),
    onSettled: (_data, error) => { keys.settled(error); refresh(); },
  });

  if (detail.isPending) return <main className="main"><Loading what="case" /></main>;
  if (detail.error) return <main className="main"><ErrorNote error={detail.error} onRetry={() => detail.refetch()} /></main>;
  const item = detail.data;

  return (
    <>
      <main className="main">
        <Panel title="Case" aside={<StatusBadge status={item.status} />}>
          <p className="muted">
            <span className="mono">{item.case_id}</span> · {item.customer_id} · opened {time(item.created_at)} · {item.issues.map(issueLabel).join(", ") || "no issue"}
          </p>
          {item.summary && (
            <details className="summary"><summary>Summary of earlier messages (model-written)</summary>
              <p>{item.summary}</p></details>
          )}
          <Conversation messages={item.messages} mine="support_agent"
                        pending={reply.isPending ? reply.variables : undefined}
                        details={(message) => message.trace && <ReplyTraceView trace={message.trace} />} />
          <ErrorNote error={reply.error} />
          {item.status !== "resolved" && (
            <Composer label="Reply to the customer" placeholder="Reply to the customer" busy={reply.isPending}
                      onSend={(text) => reply.mutateAsync(text)} />
          )}
        </Panel>
      </main>
      <aside className="sidebar evidence">
        {item.review_notes.length > 0 && (
          <Panel title="Why this needs review">
            <ul className="notes">{item.review_notes.map((note, index) => <li key={index}>{note}</li>)}</ul>
          </Panel>
        )}
        {item.refunds.map((refund) => (
          <RefundCard key={refund.refund_id} api={api} refund={refund} order={item.order} onChanged={refresh} />
        ))}
        {item.order ? <OrderFacts order={item.order} /> : (
          <Panel title="Order"><p className="muted">No order is known for this case yet.</p></Panel>
        )}
        <Resolve api={api} item={item} onChanged={refresh} />
        <Panel title="History">
          <ol className="history">
            {item.transitions.map((change, index) => (
              <li key={index}>
                <time dateTime={change.at}>{time(change.at)}</time>{" "}
                {humanize(change.from_status)} → {humanize(change.to_status)} by {humanize(change.actor_role)}
                {change.actor_id && <> <span className="mono">{change.actor_id}</span></>}
                {change.note && <span className="muted">: {change.note}</span>}
              </li>
            ))}
          </ol>
        </Panel>
      </aside>
    </>
  );
}

function Resolve({ api, item, onChanged }: { api: Api; item: SupportCase; onChanged: () => void }) {
  const [note, setNote] = useState("");
  const keys = useRetryKey();
  const resolve = useMutation({
    mutationFn: () => api.resolve(item.case_id, note.trim(), keys.keyFor({ case: item.case_id, note: note.trim() })),
    onSuccess: () => setNote(""),
    onSettled: (_data, error) => { keys.settled(error); onChanged(); },
  });
  if (item.status === "resolved") return null;
  const undecided = item.refunds.some((refund) => refund.status === "proposed");
  return (
    <Panel title="Resolve">
      {undecided && <p className="muted">Decide the refund proposal before resolving.</p>}
      <form className="stack" onSubmit={(event) => { event.preventDefault(); resolve.mutate(); }}>
        <label htmlFor="resolve-note">Internal note</label>
        <input id="resolve-note" value={note} maxLength={500} onChange={(event) => setNote(event.target.value)}
               placeholder="How the case was resolved" />
        <ErrorNote error={resolve.error} />
        <button type="submit" disabled={!note.trim() || undecided || resolve.isPending}>Resolve case</button>
      </form>
    </Panel>
  );
}
