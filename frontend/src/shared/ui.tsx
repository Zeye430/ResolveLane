import type { ReactNode } from "react";
import { ApiError, ApiTimeout } from "../api/client";
import type { CaseStatus, Citation } from "../api/types";
import { humanize, time } from "./format";

export function StatusBadge({ status }: { status: CaseStatus | string }) {
  return <span className={`badge badge-${status}`}>{humanize(status)}</span>;
}

export function Loading({ what }: { what: string }) {
  return <p className="muted" role="status">Loading {what}…</p>;
}

export function errorText(error: unknown): string {
  if (error instanceof ApiError) return `${error.detail} (${error.status})`;
  if (error instanceof ApiTimeout) return "The server did not answer in time. It may still be working on the request.";
  return "The server could not be reached. Check that the API is running.";
}

export function ErrorNote({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null;
  return (
    <p className="error" role="alert">
      {errorText(error)}
      {onRetry && <> <button type="button" onClick={onRetry}>Try again</button></>}
    </p>
  );
}

export function Panel({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head"><h2>{title}</h2>{aside}</header>
      {children}
    </section>
  );
}

const ROLE_LABELS: Record<string, string> = {
  customer: "Customer", assistant: "Assistant", support_agent: "Support agent", system: "Notice",
};

type Message = { role: string; text: string; sent_at: string; citations: Citation[]; author_id?: string | null };

/** A case conversation. ``mine`` is the role shown on the right; author IDs appear only when given,
 * and ``details`` adds internal information under a message (support agents only). */
export function Conversation<M extends Message>({ messages, mine, pending, details }: {
  messages: M[]; mine: string; pending?: string; details?: (message: M) => ReactNode;
}) {
  if (!messages.length && !pending) return <p className="muted">No messages yet.</p>;
  return (
    <ol className="conversation" aria-live="polite">
      {messages.map((message, index) => (
        <li key={index} className={`message message-${message.role} ${message.role === mine ? "mine" : ""}`}>
          <div className="message-meta">
            <strong>{ROLE_LABELS[message.role] ?? message.role}</strong>
            {message.author_id && <span className="mono">{message.author_id}</span>}
            <time dateTime={message.sent_at}>{time(message.sent_at)}</time>
          </div>
          <p>{message.text}</p>
          {message.citations.length > 0 && (
            <ul className="citations" aria-label="Help center sources">
              {message.citations.map((citation) => (
                <li key={`${citation.article_id}@${citation.version}#${citation.section}`}
                    title={`${citation.article_id} version ${citation.version}, section ${citation.section}`}>
                  {citation.title} · v{citation.version}
                </li>
              ))}
            </ul>
          )}
          {details?.(message)}
        </li>
      ))}
      {pending && (
        <li className={`message message-${mine} mine pending`}>
          <div className="message-meta"><strong>{ROLE_LABELS[mine]}</strong><span>Sending…</span></div>
          <p>{pending}</p>
        </li>
      )}
    </ol>
  );
}

export function Composer({ label, placeholder, busy, onSend, maxLength = 2000 }: {
  label: string; placeholder: string; busy: boolean; onSend: (text: string) => Promise<unknown>; maxLength?: number;
}) {
  return (
    <form className="composer" onSubmit={async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const field = form.elements.namedItem("text") as HTMLTextAreaElement;
      const text = field.value.trim();
      if (!text || busy) return;
      try {
        await onSend(text);
        field.value = "";
      } catch {
        // The error is shown by the caller; the text stays so it can be sent again.
      }
    }}>
      <label className="sr-only" htmlFor={`composer-${label}`}>{label}</label>
      <textarea id={`composer-${label}`} name="text" rows={2} maxLength={maxLength} placeholder={placeholder}
                disabled={busy} onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }} />
      <button type="submit" disabled={busy}>{busy ? "Sending…" : "Send"}</button>
    </form>
  );
}
