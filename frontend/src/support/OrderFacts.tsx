import type { Order } from "../api/types";
import { humanize, minutesBetween, money, time } from "../shared/format";
import { Panel, StatusBadge } from "../shared/ui";

/** The order record: what the case is checked against, not what the customer reported. */
export function OrderFacts({ order }: { order: Order }) {
  const delivered = order.delivery_events.find((event) => event.event_type === "delivered");
  const late = delivered ? minutesBetween(order.estimated_delivery_at, delivered.occurred_at) : null;
  const subtotal = order.items.reduce((sum, item) => sum + item.quantity * item.unit_price_minor, 0);
  return (
    <Panel title="Order record" aside={<StatusBadge status={order.status} />}>
      <p className="mono">{order.order_id}</p>
      <table className="table">
        <thead><tr><th>Item</th><th>Qty</th><th>Unit</th><th>Amount</th></tr></thead>
        <tbody>
          {order.items.map((item) => (
            <tr key={item.item_id}>
              <td>{item.name} <span className="mono muted">{item.item_id}</span></td>
              <td>{item.quantity}</td>
              <td>{money(item.unit_price_minor, order.currency)}</td>
              <td>{money(item.quantity * item.unit_price_minor, order.currency)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr><td colSpan={3}>Subtotal</td><td>{money(subtotal, order.currency)}</td></tr>
          <tr><td colSpan={3}>Delivery fee</td><td>{money(order.delivery_fee_minor, order.currency)}</td></tr>
          <tr><td colSpan={3}>Tax</td><td>{money(order.tax_minor, order.currency)}</td></tr>
          <tr className="total"><td colSpan={3}>Total</td><td>{money(order.total_minor, order.currency)}</td></tr>
        </tfoot>
      </table>
      <dl className="facts">
        <dt>Estimated</dt><dd>{time(order.estimated_delivery_at)}</dd>
        {delivered && <><dt>Delivered</dt><dd>{time(delivered.occurred_at)}
          {late !== null && <span className={late > 0 ? "late" : "muted"}>
            {" "}({late > 0 ? `${late} min late` : "on time"})</span>}</dd></>}
      </dl>
      <ol className="timeline" aria-label="Delivery events">
        {order.delivery_events.map((event) => (
          <li key={`${event.event_type}-${event.occurred_at}`}>
            <time dateTime={event.occurred_at}>{time(event.occurred_at)}</time> {humanize(event.event_type)}
          </li>
        ))}
      </ol>
    </Panel>
  );
}
