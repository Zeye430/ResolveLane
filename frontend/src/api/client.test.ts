import { describe, expect, it, vi } from "vitest";
import { Api, ApiError, ApiTimeout } from "./client";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("Api", () => {
  it("sends the bearer token and reads the body", async () => {
    const fetcher = vi.fn(async () => json(200, { subject_id: "demo-customer-101", role: "customer" }));
    const me = await new Api("token-1", fetcher).me();
    expect(me.role).toBe("customer");
    const [path, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toBe("/api/v1/me");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer token-1");
  });

  it("resends a write with the same Idempotency-Key when the connection drops", async () => {
    const fetcher = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(json(200, { case_id: "case-1" }));
    let keys = 0;
    const api = new Api("token-1", fetcher, () => `key-${++keys}`);
    await api.chat({ message: "Hi", order_id: "demo-order-1001" });
    const sent = fetcher.mock.calls.map(([, init]) => (init.headers as Record<string, string>)["Idempotency-Key"]);
    expect(sent).toEqual(["key-1", "key-1"]);
  });

  it("does not resend a write the server refused", async () => {
    const fetcher = vi.fn(async () => json(409, { detail: "Version 2 is the current version" }));
    const api = new Api("token-1", fetcher);
    const error = await api.approveRefund("refund-1", 1, "").catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect([error.status, error.detail]).toEqual([409, "Version 2 is the current version"]);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("joins validation messages", async () => {
    const fetcher = vi.fn(async () => json(422, { detail: [{ msg: "Field required" }, { msg: "Too long" }] }));
    const error = await new Api("t", fetcher).reply("case-1", "x").catch((caught) => caught);
    expect(error.detail).toBe("Field required; Too long");
  });

  it("gives up after the timeout without resending, and keeps a key the caller passes", async () => {
    const fetcher = vi.fn((_path: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    }));
    vi.useFakeTimers();
    try {
      const sent = new Api("t", fetcher as unknown as typeof fetch).chat({ message: "Hi" }, "key-from-caller");
      const outcome = sent.catch((caught) => caught);
      await vi.advanceTimersByTimeAsync(90_000);
      const error = await outcome;
      expect(error).toBeInstanceOf(ApiTimeout);
      expect(fetcher).toHaveBeenCalledTimes(1);
      const init = fetcher.mock.calls[0][1];
      expect((init.headers as Record<string, string>)["Idempotency-Key"]).toBe("key-from-caller");
    } finally {
      vi.useRealTimers();
    }
  });
});
