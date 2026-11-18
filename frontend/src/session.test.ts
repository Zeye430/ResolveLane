import { describe, expect, it, vi } from "vitest";
import { demoAccounts, loadDemoAccounts, parseDemoAccounts } from "./session";

// Invented tokens for fictional accounts.
const FILE = {
  "test-token-customer-101-xxxxxxxx": { subject_id: "demo-customer-101", role: "customer" },
  "test-token-agent-301-xxxxxxxxxxx": { subject_id: "demo-agent-301", role: "support_agent" },
  "test-token-unknown-role-xxxxxxxx": { subject_id: "someone", role: "administrator" },
  "test-token-no-subject-xxxxxxxxxx": { role: "customer" },
};

describe("demo accounts", () => {
  it("reads the RESOLVELANE_API_TOKENS shape and leaves out anything else", () => {
    expect(parseDemoAccounts(FILE).map((account) => account.subject_id)).toEqual(["demo-customer-101", "demo-agent-301"]);
    expect(parseDemoAccounts([FILE])).toEqual([]);
    expect(parseDemoAccounts(null)).toEqual([]);
  });

  it("offers the published accounts, and none when the file is missing or unreachable", async () => {
    await loadDemoAccounts(vi.fn(async () => new Response(JSON.stringify(FILE))));
    expect(demoAccounts()).toHaveLength(2);
    await loadDemoAccounts(vi.fn(async () => new Response("Not found", { status: 404 })));
    expect(demoAccounts()).toEqual([]);
    await loadDemoAccounts(vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    expect(demoAccounts()).toEqual([]);
  });
});
