"use client";

import { Fragment, useEffect, useState, useCallback, useMemo } from "react";
import type { TrialResultFile, SearchQuery, AskQuery } from "@/lib/results";
import { SearchPlan } from "@/app/components/SearchPlan";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="brand-btn-secondary"
      style={{ padding: "2px 10px", fontSize: "0.75rem" }}
      title="Copy formatted query to clipboard"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function formatAskQueryForCopy(q: AskQuery): string {
  let text = `${q.query_id}
${q.question_type ?? ""}
${q.time_taken.toFixed(2)}s
${q.tenant_id ? `tenant: ${q.tenant_id}` : ""}
${q.score !== undefined ? (q.score === 1 ? "Correct" : "Incorrect") : ""}
Question
${q.question}

Ground Truth
${q.ground_truth_answer}

System Answer
${q.system_answer}`;

  if (q.retrieved_context) {
    const ctx = q.retrieved_context as Record<string, unknown>;
    const memories = Array.isArray(ctx.memories) ? ctx.memories as { memory: string; time_added?: string }[] : null;
    if (memories) {
      text += `\n\nRetrieved Context\n${memories.length}`;
      memories.forEach((m, i) => {
        text += `\n\nMemory ${i + 1}\n${m.time_added ?? ""}\n${m.memory}`;
      });
    } else {
      text += `\n\nRetrieved Context\n${JSON.stringify(ctx, null, 2)}`;
    }
  }

  return text;
}

export default function TrialQueriesPage({
  params: paramsPromise,
}: {
  params: Promise<{ id: string; trialNum: string }>;
}) {
  const [params, setParams] = useState<{ id: string; trialNum: string } | null>(null);
  const [data, setData] = useState<TrialResultFile | null>(null);
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect" | "errors">("all");
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [tenantFilter, setTenantFilter] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"id" | "time">("id");
  const [loading, setLoading] = useState(true);

  useEffect(() => { paramsPromise.then(setParams); }, [paramsPromise]);

  useEffect(() => {
    if (!params) return;
    fetch(`/api/trial?id=${params.id}&trial=${params.trialNum}`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [params]);

  // Collect unique question types for the type filter (must be before early returns)
  const questionTypes = useMemo(() => {
    if (!data || data.metadata.mode !== "ask") return [];
    const types = new Set<string>();
    for (const q of data.queries as AskQuery[]) {
      if (q.question_type) types.add(q.question_type);
    }
    return [...types].sort();
  }, [data]);

  // Collect unique tenant IDs for the tenant filter
  const tenantIds = useMemo(() => {
    if (!data || data.metadata.mode !== "ask") return [];
    const ids = new Set<string>();
    for (const q of data.queries as AskQuery[]) {
      if (q.tenant_id) ids.add(q.tenant_id);
    }
    return [...ids].sort();
  }, [data]);

  if (loading || !params) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        <div className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3" style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }} />
        <p>Loading...</p>
      </div>
    );
  }

  if (!data) {
    return <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>Trial data not found.</div>;
  }

  const mode = data.metadata.mode;
  const isAsk = mode === "ask";

  let filteredQueries: SearchQuery[] | AskQuery[];
  if (isAsk) {
    let askQueries = data.queries as AskQuery[];
    if (filter !== "all") {
      askQueries = askQueries.filter((q) => {
        if (filter === "correct") return q.score === 1 && !q.is_error;
        if (filter === "incorrect") return q.score === 0 && !q.is_error;
        if (filter === "errors") return q.is_error;
        return true;
      });
    }
    if (typeFilter) {
      askQueries = askQueries.filter((q) => q.question_type === typeFilter);
    }
    if (tenantFilter) {
      const tf = tenantFilter.toLowerCase();
      askQueries = askQueries.filter((q) => q.tenant_id?.toLowerCase().includes(tf));
    }
    if (sortBy === "time") {
      askQueries = [...askQueries].sort((a, b) => b.time_taken - a.time_taken);
    }
    filteredQueries = askQueries;
  } else {
    let searchQueries = data.queries as SearchQuery[];
    if (sortBy === "time") {
      searchQueries = [...searchQueries].sort((a, b) => b.time_taken - a.time_taken);
    }
    filteredQueries = searchQueries;
  }

  return (
    <div>
      {/* ── Breadcrumb ───────────────────────────────────────────────────── */}
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/results" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <a
          href={`/results/experiments/${params.id}`}
          className="transition-colors hover:underline"
          style={{ color: "var(--color-green)" }}
        >
          {data.metadata.dataset}
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Trial {data.metadata.trial_number}</span>
      </nav>

      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
          Trial {data.metadata.trial_number}
        </h1>
        <div className="flex items-center gap-4 flex-wrap text-sm" style={{ color: "var(--text-secondary)" }}>
          <span>{data.metadata.total_queries} queries</span>
          <span className="brand-badge" style={{
            background: mode === "search" ? "rgba(1,198,201,0.18)" : "rgba(165,144,221,0.18)",
            color: mode === "search" ? "var(--color-cyan)" : "var(--color-lavender)",
          }}>
            {mode}
          </span>
          {data.metadata.total_errors != null && data.metadata.total_errors > 0 && (
            <span className="brand-badge" style={{ background: "rgba(255,79,94,0.12)", color: "var(--color-coral)" }}>
              {data.metadata.total_errors} errors
            </span>
          )}
          {data.metadata.total_misaligned != null && data.metadata.total_misaligned > 0 && (
            <span className="brand-badge" style={{ background: "rgba(255,199,44,0.15)", color: "var(--color-warn)" }}>
              {data.metadata.total_misaligned} misaligned
            </span>
          )}
        </div>
        {data.failed_query_ids && data.failed_query_ids.length > 0 && (
          <p className="text-xs mt-2" style={{ color: "var(--color-coral)", fontFamily: "var(--font-mono)" }}>
            Failed IDs: {data.failed_query_ids.join(", ")}
          </p>
        )}
        {data.misaligned_query_ids && data.misaligned_query_ids.length > 0 && (
          <p className="text-xs mt-1" style={{ color: "var(--color-warn)", fontFamily: "var(--font-mono)" }}>
            Misaligned IDs: {data.misaligned_query_ids.join(", ")}
          </p>
        )}
      </div>

      {/* ── Filters ──────────────────────────────────────────────────────── */}
      <div className="space-y-3 mb-6">
        {isAsk && (
          <div
            className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
          >
            <span className="eyebrow mr-2">Filter</span>
            {(["all", "correct", "incorrect", "errors"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className="brand-btn-secondary"
                style={filter === f ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
              >
                {f}
              </button>
            ))}
          </div>
        )}
        {isAsk && questionTypes.length > 0 && (
          <div
            className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
          >
            <span className="eyebrow mr-2">Type</span>
            <button
              onClick={() => setTypeFilter(null)}
              className="brand-btn-secondary"
              style={!typeFilter ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
            >
              all
            </button>
            {questionTypes.map((t) => (
              <button
                key={t}
                onClick={() => setTypeFilter(t === typeFilter ? null : t)}
                className="brand-btn-secondary"
                style={typeFilter === t ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        {isAsk && tenantIds.length > 1 && (
          <div
            className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
          >
            <span className="eyebrow mr-2">Tenant</span>
            <input
              type="text"
              value={tenantFilter ?? ""}
              onChange={(e) => setTenantFilter(e.target.value || null)}
              placeholder="Filter by tenant ID..."
              className="rounded-md px-2.5 py-1 text-xs"
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--border-default)",
                color: "var(--text-primary)",
                fontFamily: "var(--font-mono)",
                width: "200px",
              }}
            />
            {tenantFilter && (
              <button
                onClick={() => setTenantFilter(null)}
                className="brand-btn-secondary"
                style={{ padding: "2px 8px", fontSize: "0.7rem" }}
              >
                clear
              </button>
            )}
          </div>
        )}
        <div
          className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
        >
          <span className="eyebrow mr-2">Sort</span>
          {(["id", "time"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className="brand-btn-secondary"
              style={sortBy === s ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
            >
              {s === "id" ? "Query ID" : "Time (desc)"}
            </button>
          ))}
          <span className="ml-auto text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {filteredQueries.length} of {data.queries.length}
          </span>
        </div>
      </div>

      {/* ── Query data ───────────────────────────────────────────────────── */}
      {isAsk ? (
        <AskQueriesView queries={filteredQueries as AskQuery[]} />
      ) : (
        <SearchQueriesView queries={filteredQueries as SearchQuery[]} />
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */

function RetrievedContextPanel({ context }: { context: Record<string, unknown> }) {
  // Support the Engram format: { memories: [{memory, time_added}], n_memories_retrieved }
  // and fall back to raw JSON for any other agent format.
  const memories = Array.isArray(context.memories) ? context.memories as { memory: string; time_added?: string }[] : null;

  return (
    <details className="mt-3">
      <summary
        className="text-xs cursor-pointer inline-flex items-center gap-1.5"
        style={{ color: "var(--color-cyan)", fontFamily: "var(--font-mono)" }}
      >
        <svg width="14" height="14" viewBox="0 0 32 32" fill="none" className="shrink-0">
          <path d="M16 2L28 9V23L16 30L4 23V9L16 2Z" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
        Retrieved Context
        {typeof context.n_memories_retrieved === "number" && (
          <span className="brand-badge ml-1" style={{ background: "rgba(1,198,201,0.15)", color: "var(--color-cyan)" }}>
            {context.n_memories_retrieved}
          </span>
        )}
      </summary>
      <div className="mt-2 space-y-2">
        {memories ? (
          memories.map((m, i) => (
            <div
              key={i}
              className="rounded-md p-3 text-sm"
              style={{
                background: "var(--bg-card)",
                border: "1px solid var(--border-subtle)",
                color: "var(--text-primary)",
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="eyebrow">Memory {i + 1}</span>
                {m.time_added && (
                  <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {m.time_added}
                  </span>
                )}
              </div>
              <p style={{ whiteSpace: "pre-wrap" }}>{m.memory}</p>
            </div>
          ))
        ) : (
          <pre
            className="rounded-md p-3 text-xs overflow-x-auto"
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border-subtle)",
              fontFamily: "var(--font-mono)",
              color: "var(--text-secondary)",
              whiteSpace: "pre-wrap",
            }}
          >
            {JSON.stringify(context, null, 2)}
          </pre>
        )}
      </div>
    </details>
  );
}

function AskQueriesView({ queries }: { queries: AskQuery[] }) {
  return (
    <div className="space-y-4">
      {queries.map((q) => {
        const isError = q.is_error === true;
        const isCorrect = q.score === 1 && !isError;
        const borderColor = isError
          ? "var(--color-coral)"
          : q.score === undefined
            ? "var(--border-default)"
            : isCorrect
              ? "var(--color-green)"
              : "var(--color-coral)";
        const bgTint = isError
          ? "var(--bg-card-error)"
          : isCorrect
            ? "var(--bg-card-success)"
            : q.score === 0
              ? "var(--bg-card-error)"
              : "var(--bg-card)";

        const statusLabel = isError ? "Error" : isCorrect ? "Correct" : q.score === 0 ? "Incorrect" : undefined;
        const statusBg = isError
          ? "rgba(255,79,94,0.12)"
          : isCorrect
            ? "rgba(1,245,122,0.15)"
            : "rgba(255,79,94,0.12)";
        const statusFg = isError
          ? "var(--color-coral)"
          : isCorrect
            ? "var(--color-green)"
            : "var(--color-coral)";

        return (
          <div
            key={q.query_id}
            className="rounded-lg p-5"
            style={{ border: `1px solid ${borderColor}`, background: bgTint }}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  {q.query_id}
                </span>
                {q.tenant_id && (
                  <span
                    className="brand-badge"
                    style={{ background: "rgba(165,144,221,0.15)", color: "var(--color-lavender)", fontSize: "0.6875rem" }}
                  >
                    {q.tenant_id}
                  </span>
                )}
                {q.question_type && (
                  <span
                    className="brand-badge"
                    style={{ background: "rgba(1,198,201,0.12)", color: "var(--color-cyan)", fontSize: "0.6875rem" }}
                  >
                    {q.question_type}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                <CopyButton text={formatAskQueryForCopy(q)} />
                <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  {q.time_taken.toFixed(2)}s
                </span>
                {q.score !== undefined && (
                  <span
                    className="brand-badge"
                    style={{
                      background: statusBg,
                      color: statusFg,
                      fontWeight: 600,
                    }}
                  >
                    {statusLabel ?? `Score: ${q.score}`}
                  </span>
                )}
              </div>
            </div>
            <div className="mb-3">
              <div className="eyebrow mb-1">Question</div>
              <p className="text-sm" style={{ color: "var(--text-primary)" }}>{q.question}</p>
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              <div>
                <div className="eyebrow mb-1">Ground Truth</div>
                <p className="text-sm rounded-md p-3" style={{ background: "var(--bg-surface)", color: "var(--text-primary)" }}>
                  {q.ground_truth_answer}
                </p>
              </div>
              <div>
                <div className="eyebrow mb-1">System Answer</div>
                <p className="text-sm rounded-md p-3" style={{ background: "var(--bg-surface)", color: "var(--text-primary)" }}>
                  {q.system_answer}
                </p>
              </div>
            </div>
            {q.judge_reasoning && (
              <details className="mt-3">
                <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  Judge Reasoning
                </summary>
                <p
                  className="mt-2 text-sm rounded-md p-3"
                  style={{
                    background: "rgba(122,214,235,0.08)",
                    border: "1px solid rgba(122,214,235,0.2)",
                    color: "var(--text-secondary)",
                  }}
                >
                  {q.judge_reasoning}
                </p>
              </details>
            )}
            {q.retrieved_context && (
              <RetrievedContextPanel context={q.retrieved_context} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function SearchQueriesView({ queries }: { queries: SearchQuery[] }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const anyPlans = queries.some((q) => (q.num_searches ?? q.searches?.length ?? 0) > 0);

  return (
    <div className="brand-card overflow-x-auto">
      <table className="w-full text-sm text-left brand-table">
        <thead>
          <tr>
            <th style={{ width: "1%" }}></th>
            <th>ID</th>
            <th>Question</th>
            <th>Ground Truth</th>
            <th>Retrieved</th>
            {anyPlans && <th>Plan</th>}
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {queries.map((q) => {
            const overlap = q.retrieved_ids.filter((id) =>
              q.ground_truth_ids.includes(id)
            ).length;
            const hasOverlap = overlap > 0;
            const searches = q.searches ?? [];
            const numSearches = q.num_searches ?? searches.length;
            const canExpand = numSearches > 0;
            const isOpen = expanded.has(q.query_id);
            const colSpan = anyPlans ? 7 : 6;

            return (
              <Fragment key={q.query_id}>
                <tr
                  style={{
                    ...(!hasOverlap ? { background: "var(--bg-card-error)" } : {}),
                    cursor: canExpand ? "pointer" : "default",
                  }}
                  onClick={canExpand ? () => toggle(q.query_id) : undefined}
                >
                  <td style={{ textAlign: "center", color: "var(--text-muted)" }}>
                    {canExpand && (
                      <span
                        style={{
                          display: "inline-block",
                          transition: "transform 0.15s",
                          transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        ▶
                      </span>
                    )}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>{q.query_id}</td>
                  <td className="max-w-xs truncate">{q.question}</td>
                  <td className="text-xs">{q.num_ground_truth} IDs</td>
                  <td className="text-xs">
                    <span>{q.num_retrieved} IDs</span>
                    <span
                      className="ml-1 brand-badge"
                      style={{
                        background: hasOverlap ? "rgba(1,245,122,0.15)" : "rgba(255,79,94,0.12)",
                        color: hasOverlap ? "var(--color-green)" : "var(--color-coral)",
                      }}
                    >
                      {overlap} match
                    </span>
                  </td>
                  {anyPlans && (
                    <td className="text-xs">
                      {numSearches > 0 ? (
                        <span
                          className="brand-badge"
                          style={{ background: "rgba(165,144,221,0.15)", color: "var(--color-lavender)" }}
                        >
                          {numSearches} {numSearches === 1 ? "search" : "searches"}
                        </span>
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>—</span>
                      )}
                    </td>
                  )}
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>{q.time_taken.toFixed(2)}s</td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={colSpan} style={{ padding: 0 }}>
                      <div
                        className="px-4 py-3"
                        style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-subtle)" }}
                      >
                        <div className="eyebrow mb-2">Agent Search Plan</div>
                        <SearchPlan searches={searches} />
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
