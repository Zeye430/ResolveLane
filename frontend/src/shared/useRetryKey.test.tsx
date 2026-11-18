import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ApiError, ApiTimeout } from "../api/client";
import { useRetryKey } from "./useRetryKey";

describe("useRetryKey", () => {
  it("reuses the key when the same request is sent again after no answer", () => {
    const { result } = renderHook(() => useRetryKey());
    const first = result.current.keyFor({ message: "Hi" });
    result.current.settled(new ApiTimeout("/api/v1/chat", 90));
    expect(result.current.keyFor({ message: "Hi" })).toBe(first);
    result.current.settled(new TypeError("Failed to fetch"));
    expect(result.current.keyFor({ message: "Hi" })).toBe(first);
  });

  it("uses a new key once the server has answered", () => {
    const { result } = renderHook(() => useRetryKey());
    const first = result.current.keyFor({ message: "Hi" });
    result.current.settled(undefined);
    const second = result.current.keyFor({ message: "Hi" });
    expect(second).not.toBe(first);
    result.current.settled(new ApiError(422, "Invalid"));
    expect(result.current.keyFor({ message: "Hi" })).not.toBe(second);
  });

  it("uses a new key for a different request", () => {
    const { result } = renderHook(() => useRetryKey());
    const first = result.current.keyFor({ message: "Hi" });
    result.current.settled(new ApiTimeout("/api/v1/chat", 90));
    expect(result.current.keyFor({ message: "Hello" })).not.toBe(first);
  });
});
