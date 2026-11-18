// Names for the generated API types (src/api/schema.d.ts, from openapi.json; npm run api-types).
import type { components } from "./schema";

type Schemas = components["schemas"];

export type Principal = Schemas["PrincipalView"];
export type Order = Schemas["CustomerOrderView"];
export type CaseListEntry = Schemas["CaseListEntryView"];
export type CustomerCase = Schemas["CaseView"];
export type CustomerRefund = Schemas["CustomerRefundView"];
export type ChatRequest = Schemas["ChatRequest"];
export type ChatResponse = Schemas["ChatResponse"];
export type Citation = Schemas["CitationView"];
export type CaseStatus = Schemas["CaseStatus"];
export type Issue = Schemas["Issue"];
export type QueueEntry = Schemas["QueueEntryView"];
export type QueuePage = Schemas["QueuePageView"];
export type SupportCase = Schemas["SupportCaseView"];
export type SupportMessage = Schemas["SupportMessageView"];
export type Refund = Schemas["RefundView"];
export type RefundVersion = Schemas["RefundVersionView"];
export type Payment = Schemas["PaymentView"];
export type RefundAction = Schemas["RefundActionResponse"];
export type SupportAction = Schemas["SupportActionResponse"];
export type ReplyTrace = Schemas["ReplyTraceView"];
