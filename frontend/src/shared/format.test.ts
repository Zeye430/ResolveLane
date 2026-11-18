import { describe, expect, it } from "vitest";
import { minutesBetween, money, shortId, waited } from "./format";

describe("format", () => {
  it("shows minor units as currency", () => {
    expect(money(950, "USD")).toBe("$9.50");
    expect(money(0, "USD")).toBe("$0.00");
  });

  it("measures lateness in whole minutes", () => {
    expect(minutesBetween("2026-03-01T12:00:00Z", "2026-03-01T12:15:30Z")).toBe(15);
    expect(minutesBetween("2026-03-01T12:00:00Z", "2026-03-01T11:50:00Z")).toBe(-10);
  });

  it("describes waiting time", () => {
    const now = Date.parse("2026-03-01T15:30:00Z");
    expect(waited("2026-03-01T15:10:00Z", now)).toBe("20 min");
    expect(waited("2026-03-01T12:00:00Z", now)).toBe("3 h 30 min");
  });
});

describe("shortId", () => {
  it("keeps the prefix and the start of a long ID", () => {
    expect(shortId("case-f6865414753a444b8b2a240c87dc7997")).toBe("case-f6865414…");
    expect(shortId("demo-order-1001")).toBe("demo-order-1001");
  });
});
