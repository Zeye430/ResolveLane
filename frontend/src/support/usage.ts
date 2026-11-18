import type { ModelUsage, SupportMessage } from "../api/types";

export type CaseUsage = { calls: number; tokens: number; cost: number | null; replies: number; unrecorded: number };

export const totalTokens = (usage: ModelUsage) =>
  usage.input_tokens + usage.output_tokens + usage.cache_read_tokens + usage.cache_write_tokens;

/** Model usage added up over a case's assistant replies. The cost is null when any recorded
 * reply has no price; replies stored before usage was recorded are counted apart. */
export function caseUsage(messages: SupportMessage[]): CaseUsage {
  const total: CaseUsage = { calls: 0, tokens: 0, cost: 0, replies: 0, unrecorded: 0 };
  for (const message of messages) {
    if (!message.trace) continue;
    const usage = message.trace.usage;
    if (!usage) {
      total.unrecorded += 1;
      continue;
    }
    total.replies += 1;
    total.calls += usage.calls;
    total.tokens += totalTokens(usage);
    total.cost = total.cost === null || usage.cost_usd === null ? null : total.cost + usage.cost_usd;
  }
  return total;
}
