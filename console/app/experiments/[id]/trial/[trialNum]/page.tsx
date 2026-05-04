"use client";

import { useEffect, useState, useCallback } from "react";
import type { TrialResultFile, SearchQuery, AskQuery } from "@/lib/results";

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
  return `Question: ${q.question}

Ground Truth: ${q.ground_truth_answer}

System Answer: ${q.system_answer}

Judge Reasoning: ${q.judge_reasoning ?? "N/A"}`;
}

export default function TrialQueriesPage({
  params: paramsPromise,
}: {
  params: Promise<{ id: string; trialNum: string }>;
}) {
  const [params, setParams] = useState<{ id: string; trialNum: string } | null>(null);
  const [data, setData] = useState<TrialResultFile | null>(null);
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect" | "errors">("all");
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
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <a
          href={`/experiments/${params.id}`}
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
            background: mode === "search" ? "rgba(122,199,192,0.18)" : "rgba(165,144,221,0.18)",
            color: mode === "search" ? "var(--color-teal)" : "var(--color-lavender)",
          }}>
            {mode}
          </span>
          {data.metadata.total_errors != null && data.metadata.total_errors > 0 && (
            <span className="brand-badge" style={{ background: "rgba(244,64,78,0.12)", color: "var(--color-coral)" }}>
              {data.metadata.total_errors} errors
            </span>
          )}
          {data.metadata.total_misaligned != null && data.metadata.total_misaligned > 0 && (
            <span className="brand-badge" style={{ background: "rgba(249,241,93,0.15)", color: "#b8a900" }}>
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
          <p className="text-xs mt-1" style={{ color: "#b8a900", fontFamily: "var(--font-mono)" }}>
            Misaligned IDs: {data.misaligned_query_ids.join(", ")}
          </p>
        )}
      </div>

      {/* ── Filters ──────────────────────────────────────────────────────── */}
      <div className="flex gap-4 mb-6 items-center flex-wrap">
        {isAsk && (
          <div className="flex gap-1 items-center">
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
        <div className="flex gap-1 items-center">
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
        </div>
        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {filteredQueries.length} of {data.queries.length}
        </span>
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
          ? "rgba(244,64,78,0.12)"
          : isCorrect
            ? "rgba(97,189,115,0.15)"
            : "rgba(244,64,78,0.12)";
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
              <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                {q.query_id}
              </span>
              <div className="flex items-center gap-3">
                <CopyButton text={formatAskQueryForCopy(q)} />
                <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  {q.time_taken.toFixed(2)}s
                </span>
                {q.tenant_id && (
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>tenant: {q.tenant_id}</span>
                )}
                {statusLabel && (
                  <span
                    className="brand-badge"
                    style={{ background: statusBg, color: statusFg, fontWeight: 600 }}
                  >
                    {statusLabel}
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
          </div>
        );
      })}
    </div>
  );
}

function SearchQueriesView({ queries }: { queries: SearchQuery[] }) {
  return (
    <div className="brand-card overflow-x-auto">
      <table className="w-full text-sm text-left brand-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Question</th>
            <th>Ground Truth</th>
            <th>Retrieved</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {queries.map((q) => {
            const overlap = q.retrieved_ids.filter((id) =>
              q.ground_truth_ids.includes(id)
            ).length;
            const hasOverlap = overlap > 0;

            return (
              <tr
                key={q.query_id}
                style={!hasOverlap ? { background: "var(--bg-card-error)" } : {}}
              >
                <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>{q.query_id}</td>
                <td className="max-w-xs truncate">{q.question}</td>
                <td className="text-xs">{q.num_ground_truth} IDs</td>
                <td className="text-xs">
                  <span>{q.num_retrieved} IDs</span>
                  <span
                    className="ml-1 brand-badge"
                    style={{
                      background: hasOverlap ? "rgba(97,189,115,0.15)" : "rgba(244,64,78,0.12)",
                      color: hasOverlap ? "var(--color-green)" : "var(--color-coral)",
                    }}
                  >
                    {overlap} match
                  </span>
                </td>
                <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>{q.time_taken.toFixed(2)}s</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
