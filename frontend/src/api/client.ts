import type {
  CaseListEntry, CaseStatus, ChatRequest, ChatResponse, CustomerCase, Order, Principal, QueuePage,
  RefundAction, SupportAction, SupportCase,
} from "./types";

/** A response the server refused, with its status and the server's explanation. */
export class ApiError extends Error {
  constructor(readonly status: number, readonly detail: string) {
    super(detail);
    this.name = "ApiError";
  }
}

/** The server did not answer in time. A write may still be applied: send it again with the same key. */
export class ApiTimeout extends Error {
  constructor(readonly path: string, readonly seconds: number) {
    super(`No answer from ${path} within ${seconds} s`);
    this.name = "ApiTimeout";
  }
}

type Fetch = typeof fetch;

// How long to wait for an answer. A chat message waits for issue recognition and the reply agent,
// each bounded by the server's model timeout; everything else is a database read or write.
export const TIMEOUT_S = { read: 15, write: 20, chat: 90 };

// A write whose response was lost (the connection dropped) may still have been applied. It is
// sent again with the same Idempotency-Key, so the server returns the original response instead
// of applying it twice. Responses the server did send are never retried, and neither is a
// timeout, which leaves the request running on the server: the caller decides, and a retry by
// the person reuses the key (see useRetryKey).
const WRITE_ATTEMPTS = 3;

function detailOf(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => item?.msg ?? String(item)).join("; ");
  }
  return `Request failed with status ${status}`;
}

export class Api {
  constructor(private readonly token: string, private readonly fetcher: Fetch = (...args) => fetch(...args),
              private readonly newKey: () => string = () => crypto.randomUUID()) {}

  private async send<T>(method: string, path: string, seconds: number, body?: unknown, key?: string): Promise<T> {
    const headers: Record<string, string> = { Authorization: `Bearer ${this.token}` };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (key) headers["Idempotency-Key"] = key;
    // The timer covers the whole exchange, reading the body included.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), seconds * 1000);
    try {
      const response = await this.fetcher(path, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
      });
      const data = response.status === 204 ? undefined : await response.json().catch(() => undefined);
      if (controller.signal.aborted) throw new ApiTimeout(path, seconds); // the body was cut off
      if (!response.ok) throw new ApiError(response.status, detailOf(data, response.status));
      return data as T;
    } catch (error) {
      if (controller.signal.aborted) throw new ApiTimeout(path, seconds);
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  private get<T>(path: string): Promise<T> {
    return this.send<T>("GET", path, TIMEOUT_S.read);
  }

  private async write<T>(path: string, body: unknown, key?: string, seconds = TIMEOUT_S.write): Promise<T> {
    key ??= this.newKey();
    for (let attempt = 1; ; attempt++) {
      try {
        return await this.send<T>("POST", path, seconds, body, key);
      } catch (error) {
        if (error instanceof ApiError || error instanceof ApiTimeout || attempt >= WRITE_ATTEMPTS) throw error;
        await new Promise((resolve) => setTimeout(resolve, 300 * attempt));
      }
    }
  }

  me = () => this.get<Principal>("/api/v1/me");

  // customer
  orders = () => this.get<{ orders: Order[] }>("/api/v1/orders").then((body) => body.orders);
  cases = () => this.get<{ cases: CaseListEntry[] }>("/api/v1/cases").then((body) => body.cases);
  case = (caseId: string) => this.get<CustomerCase>(`/api/v1/cases/${encodeURIComponent(caseId)}`);
  chat = (body: ChatRequest, key?: string) => this.write<ChatResponse>("/api/v1/chat", body, key, TIMEOUT_S.chat);

  // support desk
  queue = (status: CaseStatus) => this.get<QueuePage>(`/api/v1/support/cases?status=${status}&limit=50`);
  supportCase = (caseId: string) => this.get<SupportCase>(`/api/v1/support/cases/${encodeURIComponent(caseId)}`);
  reply = (caseId: string, text: string, key?: string) =>
    this.write<SupportAction>(`/api/v1/support/cases/${encodeURIComponent(caseId)}/messages`, { text }, key);
  resolve = (caseId: string, note: string, key?: string) =>
    this.write<SupportAction>(`/api/v1/support/cases/${encodeURIComponent(caseId)}/resolution`, { note }, key);
  approveRefund = (refundId: string, version: number, note: string, key?: string) =>
    this.write<RefundAction>(`/api/v1/support/refunds/${encodeURIComponent(refundId)}/approval`, { version, note },
                             key);
  rejectRefund = (refundId: string, version: number, note: string, key?: string) =>
    this.write<RefundAction>(`/api/v1/support/refunds/${encodeURIComponent(refundId)}/rejection`, { version, note },
                             key);
}
