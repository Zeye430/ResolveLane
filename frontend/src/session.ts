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

export const demoAccounts = __DEMO_ACCOUNTS__;
