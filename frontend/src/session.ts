// The session is kept for this browser tab only (sessionStorage), so two tabs can be signed in
// as different accounts. A side-by-side session holds a customer's and a support agent's token
// and shows both screens in one tab, for a demo.
const KEY = "resolvelane.session";

export type Session =
  | { kind: "single"; token: string }
  | { kind: "side-by-side"; customer: string; agent: string };

export function savedSession(): Session | null {
  try {
    const text = sessionStorage.getItem(KEY);
    return text ? (JSON.parse(text) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session | null): void {
  try {
    if (session) sessionStorage.setItem(KEY, JSON.stringify(session));
    else sessionStorage.removeItem(KEY);
  } catch {
    // Storage can be unavailable (private windows); the session then lasts until reload.
  }
}

export type DemoAccount = { token: string; subject_id: string; role: string };

const ROLES = ["customer", "support_agent", "operator"];

/** Accounts from a file shaped like RESOLVELANE_API_TOKENS ({token: {subject_id, role}}); entries
 * of any other shape are left out. */
export function parseDemoAccounts(data: unknown): DemoAccount[] {
  if (!data || typeof data !== "object" || Array.isArray(data)) return [];
  return Object.entries(data as Record<string, unknown>).flatMap(([token, entry]) => {
    const account = entry as { subject_id?: unknown; role?: unknown } | null;
    return account && typeof account.subject_id === "string" && typeof account.role === "string"
      && ROLES.includes(account.role) ? [{ token, subject_id: account.subject_id, role: account.role }] : [];
  });
}

let accounts: DemoAccount[] = [];

/** Demo accounts are offered in the sign-in menu only when the server publishes
 * /demo-accounts.json: the development server does, from RESOLVELANE_API_TOKENS, and the web
 * container only when RESOLVELANE_DEMO_SIGN_IN is on. The built application contains no tokens. */
export async function loadDemoAccounts(fetcher: typeof fetch = (...args) => fetch(...args)): Promise<void> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000); // never hold up the sign-in page
  try {
    const response = await fetcher("/demo-accounts.json", { cache: "no-store", signal: controller.signal });
    accounts = response.ok ? parseDemoAccounts(await response.json()) : [];
  } catch {
    accounts = [];
  } finally {
    clearTimeout(timer);
  }
}

export const demoAccounts = (): DemoAccount[] => accounts;
