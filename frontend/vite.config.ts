/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

type DemoAccount = { token: string; subject_id: string; role: string };

// The development server offers the backend's development accounts (RESOLVELANE_API_TOKENS from
// ../.env or the shell) in an account menu, so a demo needs no copying of tokens. They are fictional
// principals for the simulator; a production build contains none and asks for a token instead.
function demoAccounts(mode: string): DemoAccount[] {
  const raw = loadEnv(mode, "..", "RESOLVELANE_API_TOKENS").RESOLVELANE_API_TOKENS;
  if (!raw) return [];
  try {
    const entries = JSON.parse(raw) as Record<string, { subject_id: string; role: string }>;
    return Object.entries(entries).map(([token, account]) => ({ token, ...account }));
  } catch {
    console.warn("RESOLVELANE_API_TOKENS is not valid JSON; no demo accounts are offered");
    return [];
  }
}

export default defineConfig(({ command, mode }) => {
  const api = loadEnv(mode, ".", "RESOLVELANE_API_URL").RESOLVELANE_API_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    define: {
      __DEMO_ACCOUNTS__: JSON.stringify(command === "serve" ? demoAccounts(mode) : []),
    },
    server: {
      port: 5173,
      // Same origin for the browser, so the API needs no CORS settings. RESOLVELANE_API_URL points
      // the proxy at an API on another port.
      proxy: { "/api": api, "/health": api },
    },
    test: { environment: "jsdom", setupFiles: ["src/test-setup.ts"] },
  };
});
