/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

// The development server publishes the backend's development accounts (RESOLVELANE_API_TOKENS from
// ../.env or the shell) at /demo-accounts.json, which the sign-in page offers, so a demo needs no
// copying of tokens. They are fictional principals for the simulator. The built application contains
// none: in a container, the web server publishes the file only when RESOLVELANE_DEMO_SIGN_IN is on.
function demoAccountsFile(mode: string): Plugin {
  return {
    name: "resolvelane-demo-accounts",
    configureServer(server) {
      server.middlewares.use("/demo-accounts.json", (_request, response) => {
        const raw = loadEnv(mode, "..", "RESOLVELANE_API_TOKENS").RESOLVELANE_API_TOKENS ?? "";
        let body = "{}";
        try {
          if (raw) body = JSON.stringify(JSON.parse(raw));
        } catch {
          console.warn("RESOLVELANE_API_TOKENS is not valid JSON; no demo accounts are offered");
        }
        response.setHeader("Content-Type", "application/json");
        response.setHeader("Cache-Control", "no-store");
        response.end(body);
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const api = loadEnv(mode, ".", "RESOLVELANE_API_URL").RESOLVELANE_API_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react(), demoAccountsFile(mode)],
    server: {
      port: 5173,
      // Same origin for the browser, so the API needs no CORS settings. RESOLVELANE_API_URL points
      // the proxy at an API on another port.
      proxy: { "/api": api, "/health": api },
    },
    test: { environment: "jsdom", setupFiles: ["src/test-setup.ts"] },
  };
});
