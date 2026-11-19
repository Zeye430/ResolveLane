import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Api } from "../api/client";
import type { EvalReportList } from "../api/types";
import { EvaluationPage } from "./EvaluationPage";
import { report, run } from "./fixtures";

afterEach(cleanup);

function show(list: EvalReportList) {
  const api = {
    evaluationReports: vi.fn(async () => list),
    evaluationReport: vi.fn(async (id: string) => report(list.reports.find((item) => item.report_id === id)!)),
  };
  render(<QueryClientProvider client={new QueryClient()}><EvaluationPage api={api as unknown as Api} /></QueryClientProvider>);
  return api;
}

const before = run("2026-01-01T10:00:00Z", { judge_focused: 0.227, judge_grounded: 1 });
const latest = run("2026-01-02T10:00:00Z", { judge_focused: 0.636, judge_grounded: 0.864 });

describe("EvaluationPage", () => {
  it("compares the newest run with the baseline and shows the regression", async () => {
    show({ reports: [latest, before], baseline: before, tolerance: 0.1, unreadable: 0 });
    expect(await screen.findByText("1 regression")).toBeInTheDocument();
    const focused = screen.getByText("Focused").closest("tr")!;
    expect(within(focused).getByText("+41 pts improved")).toBeInTheDocument();
    expect(within(focused).getByText("14/22")).toBeInTheDocument();
    expect(within(screen.getByText("Grounded").closest("tr")!).getByText(/regression/)).toBeInTheDocument();
  });

  it("shows the judge's reason for a failed reply", async () => {
    show({ reports: [latest], baseline: null, tolerance: 0.1, unreadable: 0 });
    expect(await screen.findByText(/No earlier run of this suite/)).toBeInTheDocument();
    fireEvent.click(await screen.findByText("late-delivery"));
    expect(screen.getByText(/Adds a fee rule\./)).toBeInTheDocument();
    expect(screen.getByText("Scenarios (1 of 2 with failures)")).toBeInTheDocument();
  });

  it("shows the cost per reply and per case, and the judge apart", async () => {
    const calls = { input_tokens: 40000, output_tokens: 6000, cache_read_tokens: 0, cache_write_tokens: 0 };
    const priced = { ...latest, usage: {
      app: { calls: 30, ...calls, cost_usd: 0.07, tokens_per_reply: 2091, cost_per_reply_usd: 0.0032,
             cost_per_case_usd: 0.0047 },
      judge: { calls: 22, ...calls, cost_usd: null } } };
    show({ reports: [priced], baseline: null, tolerance: 0.1, unreadable: 0 });
    expect(await screen.findByText(/2,091 tokens per reply/)).toBeInTheDocument();
    expect(screen.getByText(/\$0\.0032 per reply · \$0\.0047 per case/)).toBeInTheDocument();
    expect(screen.getByText(/judge 46,000 tokens/)).toBeInTheDocument();
  });

  it("flags a judge from the agent's own provider", async () => {
    const same = { ...latest, models: { ...latest.models, agent_provider: "api.example.com", judge_provider: "api.example.com" } };
    show({ reports: [same], baseline: null, tolerance: 0.1, unreadable: 0 });
    expect(await screen.findByText("Judge from the agent's provider")).toBeInTheDocument();
    cleanup();
    const other = { ...latest, models: { ...latest.models, agent_provider: "api.example.com", judge_provider: "api.other.example" } };
    show({ reports: [other], baseline: null, tolerance: 0.1, unreadable: 0 });
    expect(await screen.findByText(/at api\.other\.example/)).toBeInTheDocument();
    expect(screen.queryByText("Judge from the agent's provider")).toBeNull();
  });

  it("explains how to make the first run", async () => {
    show({ reports: [], baseline: null, tolerance: 0.1, unreadable: 1 });
    expect(await screen.findByText("python -m resolvelane.evaluation")).toBeInTheDocument();
    expect(screen.getByText("1 report file could not be read.")).toBeInTheDocument();
  });
});
