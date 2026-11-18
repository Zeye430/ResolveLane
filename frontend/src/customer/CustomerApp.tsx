import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { Api } from "../api/client";
import type { CustomerRefund, Order } from "../api/types";
import { issueLabel, money, time } from "../shared/format";
import { Composer, Conversation, ErrorNote, Loading, Panel, StatusBadge } from "../shared/ui";
import { useRetryKey } from "../shared/useRetryKey";

// What the main area shows: an existing case, or a new conversation about an order (or none,
// for a general question).
type Selection = { caseId: string } | { newFor: string | null };

const POLL_MS = 3000; // a support agent's reply and a refund being sent arrive without a reload

export function CustomerApp({ api }: { api: Api }) {
  const client = useQueryClient();
  const orders = useQuery({ queryKey: ["orders"], queryFn: api.orders });
  const cases = useQuery({ queryKey: ["cases"], queryFn: api.cases, refetchInterval: POLL_MS * 2 });
  const [selection, setSelection] = useState<Selection | null>(null);

  // Open the newest case, or a new conversation about the newest order, once both are known.
  useEffect(() => {
    if (selection || !cases.data || !orders.data) return;
    setSelection(cases.data[0] ? { caseId: cases.data[0].case_id } : { newFor: orders.data[0]?.order_id ?? null });
  }, [selection, cases.data, orders.data]);

  const caseId = selection && "caseId" in selection ? selection.caseId : null;
  const current = useQuery({
    queryKey: ["case", caseId], queryFn: () => api.case(caseId!), enabled: caseId !== null, refetchInterval: POLL_MS,
  });

  const keys = useRetryKey();
  const send = useMutation({
    mutationFn: (message: string) => {
      const body = caseId ? { message, case_id: caseId }
        : { message, order_id: selection && "newFor" in selection ? selection.newFor : null };
      return api.chat(body, keys.keyFor(body));
    },
    onSettled: (_data, error) => keys.settled(error),
    onSuccess: (response) => {
      setSelection({ caseId: response.case_id });
      client.invalidateQueries({ queryKey: ["case", response.case_id] });
      client.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  const orderId = current.data?.order_id ?? (selection && "newFor" in selection ? selection.newFor : null);
  const order = orders.data?.find((item) => item.order_id === orderId);

  return (
    <div className="layout layout-customer">
      <aside className="sidebar">
        <Panel title="Your orders">
          {orders.isPending ? <Loading what="orders" /> : <ErrorNote error={orders.error} onRetry={() => orders.refetch()} />}
          <ul className="list">
            {orders.data?.map((item) => (
              <li key={item.order_id}>
                <button className="list-item" onClick={() => setSelection({ newFor: item.order_id })}>
                  <span className="mono">{item.order_id}</span>
                  <span className="muted">{item.items.length} items · {money(item.total_minor, item.currency)}</span>
                  <span className="link">Get help with this order</span>
                </button>
              </li>
            ))}
            <li>
              <button className="list-item" onClick={() => setSelection({ newFor: null })}>
                <span>General question</span>
                <span className="muted">Refund times, delivery areas, your account</span>
              </button>
            </li>
          </ul>
        </Panel>
        <Panel title="Your conversations">
          {cases.isPending ? <Loading what="conversations" /> : <ErrorNote error={cases.error} onRetry={() => cases.refetch()} />}
          {cases.data?.length === 0 && <p className="muted">None yet.</p>}
          <ul className="list">
            {cases.data?.map((item) => (
              <li key={item.case_id}>
                <button className={`list-item ${item.case_id === caseId ? "selected" : ""}`}
                        aria-current={item.case_id === caseId} onClick={() => setSelection({ caseId: item.case_id })}>
                  <span className="row"><StatusBadge status={item.status} />
                    <span className="muted">{time(item.opened_at)}</span></span>
                  <span>{item.issues.map(issueLabel).join(", ") || "Conversation"}</span>
                  {item.order_id && <span className="mono muted">{item.order_id}</span>}
                </button>
              </li>
            ))}
          </ul>
        </Panel>
      </aside>

      <main className="main">
        <Panel title={caseId ? "Conversation" : "New conversation"}
               aside={current.data && <StatusBadge status={current.data.status} />}>
          {caseId && current.isPending && <Loading what="conversation" />}
          <ErrorNote error={current.error} onRetry={() => current.refetch()} />
          {!caseId && (
            <p className="muted">
              {orderId ? <>About order <span className="mono">{orderId}</span>. </> : "A question not about an order. "}
              Tell us what happened.
            </p>
          )}
          <Conversation messages={current.data?.messages ?? []} mine="customer"
                        pending={send.isPending ? send.variables : undefined} />
          {current.data?.status === "needs_review" && (
            <p className="notice">A support agent will review this conversation. Their reply appears here.</p>
          )}
          <ErrorNote error={send.error} />
          <Composer label="Message" placeholder="Write your message" busy={send.isPending}
                    onSend={(text) => send.mutateAsync(text)} />
        </Panel>
      </main>

      <aside className="sidebar">
        {order && <OrderSummary order={order} />}
        {(current.data?.refunds ?? []).length > 0 && (
          <Panel title="Refunds">
            {(current.data?.refunds ?? []).map((refund) => <CustomerRefundCard key={refund.refund_id} refund={refund} />)}
          </Panel>
        )}
      </aside>
    </div>
  );
}

function OrderSummary({ order }: { order: Order }) {
  return (
    <Panel title="Order" aside={<StatusBadge status={order.status} />}>
      <p className="mono">{order.order_id}</p>
      <ul className="lines">
        {order.items.map((item) => (
          <li key={item.item_id}>
            <span>{item.quantity} × {item.name}</span>
            <span>{money(item.quantity * item.unit_price_minor, order.currency)}</span>
          </li>
        ))}
        <li className="total"><span>Total</span><span>{money(order.total_minor, order.currency)}</span></li>
      </ul>
      <p className="muted">Estimated delivery {time(order.estimated_delivery_at)}</p>
    </Panel>
  );
}

const PAYMENT_TEXT: Record<CustomerRefund["payment_status"], string> = {
  processing: "Being sent to your original payment method",
  sent: "Sent to your original payment method",
  delayed: "Delayed: a support agent is looking into it",
};

function CustomerRefundCard({ refund }: { refund: CustomerRefund }) {
  return (
    <div className={`refund refund-${refund.payment_status}`}>
      <p className="refund-amount">{money(refund.amount_minor, refund.currency)} approved</p>
      <ul className="lines">
        {refund.lines.map((line) => (
          <li key={line.item_id}><span>{line.quantity} × {line.name}</span>
            <span>{money(line.amount_minor, refund.currency)}</span></li>
        ))}
        {refund.delivery_fee_minor > 0 && (
          <li><span>Delivery fee</span><span>{money(refund.delivery_fee_minor, refund.currency)}</span></li>
        )}
      </ul>
      <p role="status"><StatusBadge status={refund.payment_status} /> {PAYMENT_TEXT[refund.payment_status]}
        {refund.sent_at && <> on {time(refund.sent_at)}</>}</p>
    </div>
  );
}
