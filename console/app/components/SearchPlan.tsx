"use client";

import type { AgentSearch, FilterNode, FilterGroup, QuerySort } from "@/lib/results";

/* ── Search-plan rendering ────────────────────────────────────────────────── */
// Shared between the trial detail view and the per-query comparison view.

// Map raw Query Agent operators to readable symbols/words.
const OPERATOR_LABELS: Record<string, string> = {
  "=": "=",
  "!=": "≠",
  "<": "<",
  ">": ">",
  "<=": "≤",
  ">=": "≥",
  LIKE: "like",
  contains_any: "contains any",
  contains_all: "contains all",
};

function isFilterGroup(node: FilterNode): node is FilterGroup {
  return (node as FilterGroup).combine !== undefined;
}

function formatFilterValue(value: unknown): string {
  if (value === null || value === undefined) return "∅";
  if (Array.isArray(value)) return value.map((v) => formatFilterValue(v)).join(", ");
  if (typeof value === "object") {
    const v = value as Record<string, unknown>;
    // Date filter value shapes (exact / from / to / between)
    if ("exact_timestamp" in v) return String(v.exact_timestamp);
    if ("date_from" in v && "date_to" in v) return `${v.date_from} → ${v.date_to}`;
    if ("date_from" in v) return `from ${v.date_from}`;
    if ("date_to" in v) return `to ${v.date_to}`;
    return JSON.stringify(v);
  }
  return String(value);
}

/** A single leaf filter rendered as a monospace chip. */
function FilterChip({ node }: { node: Exclude<FilterNode, FilterGroup> }) {
  let body: string;
  if (node.filter_type === "is_null") {
    body = `${node.property_name} is ${node.is_null ? "null" : "not null"}`;
  } else if (node.filter_type === "geo") {
    body = `${node.property_name} within ${node.max_distance_meters}m of (${node.latitude}, ${node.longitude})`;
  } else {
    const op = node.operator ? OPERATOR_LABELS[node.operator] ?? node.operator : "";
    body = `${node.property_name ?? "?"} ${op} ${formatFilterValue(node.value)}`.trim();
  }
  return (
    <span
      className="inline-flex items-center rounded-md px-2 py-0.5"
      style={{
        background: "rgba(122,199,192,0.12)",
        border: "1px solid rgba(122,199,192,0.3)",
        color: "var(--color-teal)",
        fontFamily: "var(--font-mono)",
        fontSize: "0.7rem",
      }}
    >
      {body}
    </span>
  );
}

/** Recursively render a filter tree: leaves as chips, groups joined by AND/OR. */
export function FilterTree({ node }: { node: FilterNode }) {
  if (!isFilterGroup(node)) {
    return <FilterChip node={node} />;
  }
  return (
    <span className="inline-flex items-center gap-1.5 flex-wrap">
      <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>(</span>
      {node.filters.map((child, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {i > 0 && (
            <span
              className="font-semibold"
              style={{ color: "var(--color-lavender)", fontSize: "0.65rem", letterSpacing: "0.05em" }}
            >
              {node.combine}
            </span>
          )}
          <FilterTree node={child} />
        </span>
      ))}
      <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>)</span>
    </span>
  );
}

function formatSort(sort: QuerySort): string {
  const arrow = sort.order === "descending" ? "↓" : "↑";
  let label = `${sort.property_name} ${arrow}`;
  if (sort.tie_break) label += `, then ${formatSort(sort.tie_break)}`;
  return label;
}

/** The agent's search plan for one benchmark query: a list of sub-searches. */
export function SearchPlan({ searches }: { searches: AgentSearch[] }) {
  if (searches.length === 0) {
    return (
      <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        No search plan recorded for this agent.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      {searches.map((s, i) => (
        <div
          key={i}
          className="rounded-md p-3"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)" }}
        >
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="eyebrow">Search {i + 1}</span>
            <span
              className="brand-badge"
              style={{ background: "rgba(165,144,221,0.15)", color: "var(--color-lavender)", fontSize: "0.65rem" }}
            >
              {s.collection}
            </span>
            {s.uuid_value && (
              <span
                className="brand-badge"
                style={{ background: "rgba(122,199,192,0.12)", color: "var(--color-teal)", fontSize: "0.65rem" }}
                title="Direct UUID lookup"
              >
                uuid: {s.uuid_value.slice(0, 8)}…
              </span>
            )}
          </div>

          <div className="grid gap-2" style={{ gridTemplateColumns: "auto 1fr" }}>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>query</span>
            <span className="text-xs" style={{ color: "var(--text-primary)" }}>
              {s.query ? (
                <span style={{ fontFamily: "var(--font-mono)" }}>&ldquo;{s.query}&rdquo;</span>
              ) : (
                <em style={{ color: "var(--text-muted)" }}>none (filter / lookup only)</em>
              )}
            </span>

            <span className="text-xs" style={{ color: "var(--text-muted)" }}>filters</span>
            <span className="text-xs">
              {s.filters ? (
                <FilterTree node={s.filters} />
              ) : (
                <em style={{ color: "var(--text-muted)" }}>none</em>
              )}
            </span>

            {s.sort_property && (
              <>
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>sort</span>
                <span
                  className="text-xs"
                  style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
                >
                  {formatSort(s.sort_property)}
                </span>
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
