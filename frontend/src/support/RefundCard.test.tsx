import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Api, ApiError } from "../api/client";
import type { Refund } from "../api/types";
import { RefundCard } from "./RefundCard";

afterEach(cleanup);

const version = (number: number, quantity: number) => ({
  version: number, lines: [{ item_id: "item-1", name: "Vegetable wrap", quantity, amount_minor: 950 * quantity }],
  delivery_fee_minor: 0, amount_minor: 950 * quantity, reason: "Wrap missing", author_role: "assistant",
  author_id: null, created_at: "2026-03-01T12:00:00Z",
});

const proposed: Refund = {
  refund_id: "refund-1", order_id: "demo-order-1001", currency: "USD", status: "proposed",
  created_at: "2026-03-01T12:00:00Z", versions: [version(1, 1)], decided_version: null, decided_at: null,
  decided_by: null, decision_note: "", payment: null,
};

function show(refund: Refund, api: Partial<Api>, onChanged = () => {}) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RefundCard api={api as Api} refund={refund} order={null} onChanged={onChanged} />
    </QueryClientProvider>,
  );
}

describe("RefundCard", () => {
  it("approves the version on screen", async () => {
    const approveRefund = vi.fn(async () => ({}) as never);
    const onChanged = vi.fn();
    show(proposed, { approveRefund }, onChanged);
    fireEvent.click(screen.getByRole("button", { name: /Approve v1 · \$9\.50/ }));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(approveRefund).toHaveBeenCalledWith("refund-1", 1, "", expect.any(String));
  });

  it("explains a proposal that changed during review", async () => {
    const approveRefund = vi.fn(async () => { throw new ApiError(409, "Version 2 is the current version"); });
    show(proposed, { approveRefund });
    fireEvent.click(screen.getByRole("button", { name: /Approve v1/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/changed while you were reviewing/);
  });

  it("requires a note to reject", () => {
    show(proposed, {});
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/Internal note/), { target: { value: "Restaurant confirmed it was sent" } });
    expect(screen.getByRole("button", { name: "Reject" })).toBeEnabled();
  });

  it("shows the decided version and the payment, without decision buttons", () => {
    show({
      ...proposed, status: "approved", versions: [version(1, 1), version(2, 2)], decided_version: 2,
      decided_by: "demo-agent-301", decided_at: "2026-03-01T12:05:00Z",
      payment: { payment_id: "payment-1", amount_minor: 1900, currency: "USD", status: "succeeded", attempts: 1,
                 next_attempt_at: "2026-03-01T12:05:00Z", provider_refund_id: "sim-ref-1", last_error: "",
                 created_at: "2026-03-01T12:05:00Z", completed_at: "2026-03-01T12:05:02Z" },
    }, {});
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();
    expect(screen.getByText("2 × Vegetable wrap")).toBeInTheDocument();
    expect(screen.getByText("sim-ref-1")).toBeInTheDocument();
  });
});
