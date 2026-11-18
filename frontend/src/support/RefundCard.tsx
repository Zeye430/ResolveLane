import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError, type Api } from "../api/client";
import type { Order, Payment, Refund, RefundVersion } from "../api/types";
import { humanize, money, time } from "../shared/format";
import { ErrorNote, Panel, StatusBadge } from "../shared/ui";
import { useRetryKey } from "../shared/useRetryKey";

/**
 * A refund proposal and its decision. The amount shown is the server's, priced from the order
 * record; a decision names the version on screen, so the server refuses it (409) if the proposal
 * changed in the meantime, and the card then shows the current version.
 */
export function RefundCard({ api, refund, order, onChanged }: {
  api: Api; refund: Refund; order: Order | null | undefined; onChanged: () => void;
}) {
  const current = refund.versions[refund.versions.length - 1];
  const [note, setNote] = useState("");
  const keys = useRetryKey();
  const decide = useMutation({
    mutationFn: (decision: "approve" | "reject") => {
      const request = { decision, refund: refund.refund_id, version: current.version, note: note.trim() };
      return decision === "approve"
        ? api.approveRefund(refund.refund_id, current.version, request.note, keys.keyFor(request))
        : api.rejectRefund(refund.refund_id, current.version, request.note, keys.keyFor(request));
    },
    onSuccess: () => setNote(""),
    onSettled: (_data, error) => { keys.settled(error); onChanged(); },
  });
  const stale = decide.error instanceof ApiError && decide.error.status === 409 ? decide.error : null;
  const shown = refund.decided_version
    ? refund.versions.find((version) => version.version === refund.decided_version) ?? current : current;

  return (
    <Panel title="Refund" aside={<StatusBadge status={refund.status} />}>
      <Version version={shown} currency={refund.currency} order={order} />
      {refund.versions.length > 1 && (
        <details>
          <summary>{refund.versions.length} versions</summary>
          <ol className="history">
            {refund.versions.map((version) => (
              <li key={version.version}>
                v{version.version} by {humanize(version.author_role)}
                {version.author_id && <> <span className="mono">{version.author_id}</span></>}:{" "}
                {money(version.amount_minor, refund.currency)} · {version.reason}
              </li>
            ))}
          </ol>
        </details>
      )}

      {refund.status === "proposed" ? (
        <form className="stack" onSubmit={(event) => event.preventDefault()}>
          <label htmlFor={`note-${refund.refund_id}`}>Internal note (required to reject)</label>
          <input id={`note-${refund.refund_id}`} value={note} maxLength={500}
                 onChange={(event) => setNote(event.target.value)} />
          {stale ? (
            <p className="error" role="alert">
              The proposal changed while you were reviewing it ({stale.detail}). Review version{" "}
              {current.version} above before deciding.
            </p>
          ) : <ErrorNote error={decide.error} />}
          <div className="row">
            <button type="button" className="primary" disabled={decide.isPending}
                    onClick={() => decide.mutate("approve")}>
              Approve v{current.version} · {money(current.amount_minor, refund.currency)}
            </button>
            <button type="button" disabled={decide.isPending || !note.trim()} onClick={() => decide.mutate("reject")}>
              Reject
            </button>
          </div>
        </form>
      ) : (
        <p className="muted">
          {humanize(refund.status)} v{refund.decided_version} by <span className="mono">{refund.decided_by}</span>
          {refund.decided_at && <> on {time(refund.decided_at)}</>}
          {refund.decision_note && <>: {refund.decision_note}</>}
        </p>
      )}
      {refund.payment && <PaymentFacts payment={refund.payment} />}
    </Panel>
  );
}

function Version({ version, currency, order }: { version: RefundVersion; currency: string; order: Order | null | undefined }) {
  const ordered = new Map(order?.items.map((item) => [item.item_id, item.quantity]));
  return (
    <div className="stack">
      <ul className="lines">
        {version.lines.map((line) => (
          <li key={line.item_id}>
            <span>{line.quantity} × {line.name}
              {ordered.has(line.item_id) && <span className="muted"> of {ordered.get(line.item_id)} ordered</span>}</span>
            <span>{money(line.amount_minor, currency)}</span>
          </li>
        ))}
        {version.delivery_fee_minor > 0 && (
          <li><span>Delivery fee</span><span>{money(version.delivery_fee_minor, currency)}</span></li>
        )}
        <li className="total"><span>Refund</span><span>{money(version.amount_minor, currency)}</span></li>
      </ul>
      <p className="muted">
        v{version.version} by {humanize(version.author_role)}
        {version.author_id && <> <span className="mono">{version.author_id}</span></>} · {version.reason}
      </p>
      <p className="hint">Amount priced by the server from the order record.</p>
    </div>
  );
}

function PaymentFacts({ payment }: { payment: Payment }) {
  return (
    <div className="payment">
      <p className="row"><strong>Payment</strong><StatusBadge status={payment.status} /></p>
      <dl className="facts">
        <dt>Payment ID</dt><dd className="mono" title="Also the idempotency key sent to the provider">{payment.payment_id}</dd>
        <dt>Attempts</dt><dd>{payment.attempts}</dd>
        {payment.status === "pending" && <><dt>Next attempt</dt><dd>{time(payment.next_attempt_at)}</dd></>}
        {payment.provider_refund_id && <><dt>Provider ref</dt><dd className="mono">{payment.provider_refund_id}</dd></>}
        {payment.last_error && <><dt>Last error</dt><dd>{payment.last_error}</dd></>}
        {payment.completed_at && <><dt>Completed</dt><dd>{time(payment.completed_at)}</dd></>}
      </dl>
    </div>
  );
}
