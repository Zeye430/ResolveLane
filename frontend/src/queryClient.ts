import { QueryClient } from "@tanstack/react-query";
import { ApiError, ApiTimeout } from "./api/client";

/** One cache per signed-in account, so nothing read as one account is shown to another. */
export function newQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // A refusal (4xx) will not change on retry, and a timeout already waited long enough;
        // a dropped connection may recover.
        retry: (failures, error) =>
          !(error instanceof ApiError && error.status < 500) && !(error instanceof ApiTimeout) && failures < 2,
        refetchOnWindowFocus: true,
      },
      mutations: { retry: false }, // a person retries a write, with the same Idempotency-Key
    },
  });
}
