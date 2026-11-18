import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ReplyTrace } from "../api/types";
import { ReplyTraceView } from "./ReplyTraceView";

afterEach(cleanup);

const base: ReplyTrace = {
  written_by: "agent", fallback: "none", recognition: "llm", issues: ["missing_item"], seconds: 3.42,
  model: "test-model", rounds: 2, skills: ["missing-item@3"], repeat: false,
  tools: [{ name: "propose_refund", ok: true, latency_ms: 4.4, error: "" },
          { name: "courier_tracking", ok: false, latency_ms: 30000, error: "server_unavailable" }],
};

describe("ReplyTraceView", () => {
  it("shows what the agent used", () => {
    render(<ReplyTraceView trace={base} />);
    expect(screen.getByText("Written by the AI agent")).toBeInTheDocument();
    expect(screen.getByText("Missing item by language model")).toBeInTheDocument();
    expect(screen.getByText("test-model · 2 calls")).toBeInTheDocument();
    expect(screen.getByText(/failed: Server unavailable/)).toBeInTheDocument();
    expect(screen.getByText("missing-item@3")).toBeInTheDocument();
  });

  it("explains why rules replaced the agent's reply", () => {
    render(<ReplyTraceView trace={{ ...base, written_by: "rules", fallback: "ungrounded_number", tools: [],
                                    skills: [], rounds: 0, repeat: true }} />);
    expect(screen.getByText("Rule-based fallback")).toBeInTheDocument();
    expect(screen.getByText(/stated a number found neither in the case facts/)).toBeInTheDocument();
    expect(screen.getByText("Repeats previous reply")).toBeInTheDocument();
  });

  it("shows an unknown reason as it is", () => {
    render(<ReplyTraceView trace={{ ...base, written_by: "rules", fallback: "status_529", tools: [] }} />);
    expect(screen.getByText(/status_529/)).toBeInTheDocument();
  });
});
