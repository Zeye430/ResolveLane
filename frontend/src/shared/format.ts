import type { Issue } from "../api/types";

/** Amounts travel as integer minor units (cents); they are only turned into text for display. */
export function money(minor: number, currency: string): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(minor / 100);
}

export function time(value: string): string {
  return new Date(value).toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  });
}

/** Minutes from one timestamp to another, rounded down; negative when early. */
export function minutesBetween(from: string, to: string): number {
  return Math.floor((Date.parse(to) - Date.parse(from)) / 60000);
}

export function waited(since: string, now: number = Date.now()): string {
  const minutes = Math.max(0, Math.floor((now - Date.parse(since)) / 60000));
  if (minutes < 60) return `${minutes} min`;
  if (minutes < 48 * 60) return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
  return `${Math.floor(minutes / 1440)} days`;
}

const ISSUE_LABELS: Record<Issue, string> = {
  late_delivery: "Late delivery", missing_item: "Missing item", wrong_item: "Wrong item",
  damaged_item: "Damaged item", cancellation: "Cancellation", delivery_status: "Delivery status",
  unknown: "Other",
};

export const issueLabel = (issue: Issue) => ISSUE_LABELS[issue];

export function humanize(value: string): string {
  const text = value.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** A long generated ID shortened for lists; the full ID goes in a tooltip. */
export function shortId(id: string, keep = 8): string {
  const at = id.lastIndexOf("-");
  const body = id.slice(at + 1);
  return body.length > keep + 1 ? `${id.slice(0, at + 1)}${body.slice(0, keep)}…` : id;
}
