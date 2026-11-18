import { useRef } from "react";
import { ApiError } from "../api/client";

/**
 * Idempotency keys for a person's writes. Sending the same request again after it failed without
 * an answer (a timeout or a dropped connection) reuses its key, so the server applies it at most
 * once even if the first attempt went through. Once the server has answered, success or refusal,
 * the next request gets a new key: writing the same text again later is a new message.
 */
export function useRetryKey() {
  const pending = useRef<{ request: string; key: string } | null>(null);
  return {
    keyFor(request: unknown): string {
      const text = JSON.stringify(request);
      if (pending.current?.request !== text) pending.current = { request: text, key: crypto.randomUUID() };
      return pending.current.key;
    },
    settled(error: unknown): void {
      if (!error || error instanceof ApiError) pending.current = null;
    },
  };
}
