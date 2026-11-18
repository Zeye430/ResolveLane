import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Api, ApiError } from "./api/client";
import { CustomerApp } from "./customer/CustomerApp";
import { EvaluationPage } from "./operator/EvaluationPage";
import { newQueryClient } from "./queryClient";
import { demoAccounts, savedSession, saveSession, type Session } from "./session";
import { humanize } from "./shared/format";
import { ErrorNote, Loading } from "./shared/ui";
import { SupportApp } from "./support/SupportApp";

export function App() {
  const [session, setSession] = useState<Session | null>(savedSession);
  const change = (next: Session | null) => {
    saveSession(next);
    setSession(next);
  };

  return (
    <>
      <header className="topbar">
        <span className="brand">ResolveLane</span>
        <span className="simulated" title="All orders, customers and payments are invented">Simulated data</span>
        {session && (
          <span className="account">
            {session.kind === "side-by-side" && <span className="muted">Customer and support agent side by side</span>}
            <button onClick={() => change(null)}>{session.kind === "single" ? "Switch account" : "Leave demo"}</button>
          </span>
        )}
      </header>
      {!session ? <SignIn onSignIn={change} />
        : session.kind === "single" ? (
          // Keyed by token: a new account gets a new cache.
          <Account key={session.token} token={session.token} onSignIn={change} showIdentity />
        ) : (
          <div className="panes">
            <Account key={session.customer} token={session.customer} onSignIn={change} />
            <Account key={session.agent} token={session.agent} onSignIn={change} />
          </div>
        )}
    </>
  );
}

/** One signed-in account with its own cache: finds out who the token belongs to, then shows their screen. */
function Account({ token, onSignIn, showIdentity = false }: {
  token: string; onSignIn: (session: Session) => void; showIdentity?: boolean;
}) {
  const [client] = useState(newQueryClient);
  return (
    <section className="app-body">
      <QueryClientProvider client={client}>
        <Screen token={token} onSignIn={onSignIn} showIdentity={showIdentity} />
      </QueryClientProvider>
    </section>
  );
}

function Screen({ token, onSignIn, showIdentity }: {
  token: string; onSignIn: (session: Session) => void; showIdentity: boolean;
}) {
  const api = useMemo(() => new Api(token), [token]);
  const me = useQuery({ queryKey: ["me"], queryFn: api.me });

  if (me.isPending) return <Loading what="account" />;
  if (me.error) {
    return (
      <div className="signin">
        <ErrorNote error={me.error} onRetry={() => me.refetch()} />
        {me.error instanceof ApiError && me.error.status === 401 && <SignIn onSignIn={onSignIn} />}
      </div>
    );
  }
  const identity = (
    <p className={showIdentity ? "identity" : "pane-title"}>
      <span className="mono">{me.data.subject_id}</span> <span className="muted">{humanize(me.data.role)}</span>
    </p>
  );
  return (
    <>
      {identity}
      {me.data.role === "customer" ? <CustomerApp api={api} />
        : me.data.role === "support_agent" ? <SupportApp api={api} />
          : <EvaluationPage api={api} />}
    </>
  );
}

function SignIn({ onSignIn }: { onSignIn: (session: Session) => void }) {
  const [pasted, setPasted] = useState("");
  const accounts = demoAccounts();
  const customers = accounts.filter((account) => account.role === "customer");
  const agents = accounts.filter((account) => account.role === "support_agent");
  const [customer, setCustomer] = useState(customers[0]?.token ?? "");
  const [agent, setAgent] = useState(agents[0]?.token ?? "");
  return (
    <div className="signin">
      <h1>Sign in to the simulator</h1>
      {customers.length > 0 && agents.length > 0 && (
        <form className="stack demo-form" onSubmit={(event) => {
          event.preventDefault();
          onSignIn({ kind: "side-by-side", customer, agent });
        }}>
          <h2>Side by side</h2>
          <p className="muted">A customer and a support agent in one window: report a problem on the left, decide it on the right.</p>
          <div className="row">
            <label>Customer{" "}
              <select value={customer} onChange={(event) => setCustomer(event.target.value)}>
                {customers.map((account) => <option key={account.token} value={account.token}>{account.subject_id}</option>)}
              </select>
            </label>
            <label>Support agent{" "}
              <select value={agent} onChange={(event) => setAgent(event.target.value)}>
                {agents.map((account) => <option key={account.token} value={account.token}>{account.subject_id}</option>)}
              </select>
            </label>
          </div>
          <button type="submit" className="primary">Start side by side</button>
        </form>
      )}
      {accounts.length > 0 ? (
        <>
          <h2>One account</h2>
          <p className="muted">Demo accounts from RESOLVELANE_API_TOKENS. Each browser tab keeps its own sign-in.</p>
          <ul className="accounts">
            {accounts.map((account) => (
              <li key={account.token}>
                <button onClick={() => onSignIn({ kind: "single", token: account.token })}>
                  <span className="mono">{account.subject_id}</span>
                  <span className="muted">{humanize(account.role)}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : <p className="muted">No development accounts are configured for this server. Paste an API token.</p>}
      <form className="stack" onSubmit={(event) => {
        event.preventDefault();
        if (pasted.trim()) onSignIn({ kind: "single", token: pasted.trim() });
      }}>
        <label htmlFor="token">API token</label>
        <input id="token" type="password" autoComplete="off" value={pasted} onChange={(event) => setPasted(event.target.value)} />
        <button type="submit" disabled={!pasted.trim()}>Sign in</button>
      </form>
    </div>
  );
}
