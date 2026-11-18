import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ModelUsage, SupportMessage } from "../api/types";
import { usd } from "../shared/format";
import { UsageText } from "./ReplyTraceView";
import { caseUsage } from "./usage";

afterEach(cleanup);

const usage = (cost_usd: number | null, calls = 2): ModelUsage => ({
  calls, input_tokens: 1500, output_tokens: 120, cache_read_tokens: 0, cache_write_tokens: 0, cost_usd,
});

const reply = (traceUsage: ModelUsage | null | undefined): SupportMessage => ({
  role: "assistant", text: "Noted.", sent_at: "2026-03-01T12:00:00Z", citations: [], author_id: null,
  trace: {
    written_by: "agent", fallback: "none", recognition: "llm", issues: [], seconds: 3, model: "m", rounds: 1,
    tools: [], skills: [], repeat: false, usage: traceUsage,
  },
});

const customer: SupportMessage = {
  role: "customer", text: "Hi", sent_at: "2026-03-01T12:00:00Z", citations: [], author_id: null, trace: null,
};

describe("model usage", () => {
  it("adds up a case and keeps unpriced and unrecorded replies honest", () => {
    expect(caseUsage([customer, reply(usage(0.0042)), reply(usage(0.0018))]))
      .toMatchObject({ calls: 4, tokens: 3240, replies: 2, unrecorded: 0 });
    expect(caseUsage([reply(usage(0.0042)), reply(usage(0.0042))]).cost).toBeCloseTo(0.0084);
    expect(caseUsage([reply(usage(0.0042)), reply(usage(null))]).cost).toBeNull();
    expect(caseUsage([reply(null), reply(usage(0.001))])).toMatchObject({ replies: 1, unrecorded: 1 });
  });

  it("shows a reply's calls, tokens and cost", () => {
    render(<UsageText usage={usage(0.0042)} />);
    expect(screen.getByText(/2 model calls · 1,500 in · 120 out tokens · \$0\.0042/)).toBeInTheDocument();
    cleanup();
    render(<UsageText usage={usage(null)} />);
    expect(screen.getByText("no price configured")).toBeInTheDocument();
    cleanup();
    render(<UsageText usage={null} />);
    expect(screen.getByText("not recorded")).toBeInTheDocument();
  });

  it("formats fractions of a cent", () => {
    expect(usd(0)).toBe("$0");
    expect(usd(0.00042)).toBe("$0.0004");
    expect(usd(0.125)).toBe("$0.125");
    expect(usd(12.5)).toBe("$12.50");
  });
});
