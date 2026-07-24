"use client";

import { useEffect, useState, useCallback } from "react";
import type { TrialResultFile, SearchQuery, AskQuery } from "@/lib/results";

interface TrialSummary {
  trialNumber: number;
  hasResults: boolean;
  totalQueries: number | null;
  avgQueryTime: number | null;
  metrics: Record<string, unknown> | null;
}

interface ExperimentDetail {
  id: string;
  dataset: string;
  agent_name: string;
  mode: "search" | "ask" | "unknown";
  num_trials: number;
  timestamp: string;
  metricEntries: { key: string; value: string }[];
  trials: TrialSummary[];
}

const POLL_INTERVAL_MS = 5_000;

const KEY_SCORE_KEYS = [
  "avg_alignment_score",
  "avg_exact_match_accuracy",
  "avg_recall_at_5",
  "avg_nDCG_at_10",
];

function parseMetricValue(value: string): number | null {
  const num = parseFloat(value);
  return isNaN(num) ? null : num;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Markdown report export — single experiment
   ═══════════════════════════════════════════════════════════════════════════ */

/** Escape pipe characters so values can't break markdown table rows. */
function escPipe(s: string): string {
  return s.replace(/\|/g, "\\|");
}

/** Collapse whitespace and truncate for compact display in a table cell. */
function truncateCell(s: string, max = 100): string {
  const clean = escPipe(s.replace(/\s+/g, " ").trim());
  return clean.length > max ? `${clean.slice(0, max)}…` : clean;
}

/** Count how many retrieved IDs are relevant (present in the ground truth). */
function relevantHits(q: SearchQuery): number {
  const gt = new Set(q.ground_truth_ids);
  return q.retrieved_ids.filter((id) => gt.has(id)).length;
}

/** A representative trial whose per-query results are embedded in the report. */
interface ReportTrial {
  trialNumber: number;
  data: TrialResultFile;
}

/**
 * Append a per-query results section for one trial: a compact overview table
 * followed by full details for every query. Mirrors the comparison report's
 * per-query layout. No-op if the trial has no queries.
 */
function appendPerQueryResults(
  L: (s?: string) => void,
  trial: ReportTrial,
  totalTrials: number,
): void {
  const mode = trial.data.metadata.mode;
  const queries = trial.data.queries;
  if (queries.length === 0) return;

  L(`## Per-Query Results`);
  L();
  L(`_All ${queries.length} queries from Trial ${trial.trialNumber} (mode: ${mode})._`);
  if (totalTrials > 1) {
    L();
    L(
      `_This experiment ran ${totalTrials} trials; the queries below are from a single representative trial. ` +
        `Per-trial aggregate scores are in the Trials table above._`,
    );
  }
  L();

  if (mode === "ask") {
    const asks = queries as AskQuery[];
    const askStatus = (q: AskQuery): string =>
      q.is_error ? "error" : q.score === 1 ? "✓ correct" : q.score === 0 ? "✗ wrong" : "? unscored";

    L(`### Results Overview`);
    L();
    L(`| # | Status | Time | Question |`);
    L(`| ---: | :--- | ---: | :--- |`);
    asks.forEach((q, i) => {
      const status = q.is_error ? "error" : q.score === 1 ? "✓" : q.score === 0 ? "✗" : "?";
      L(`| ${i + 1} | ${status} | ${q.time_taken.toFixed(2)}s | ${truncateCell(q.question)} |`);
    });
    L();
    L(`_Status: ✓ correct · ✗ wrong · ? unscored · error._`);
    L();

    L(`### Query Details`);
    L();
    asks.forEach((q, i) => {
      L(`#### ${i + 1}. [${askStatus(q)}] \`${escPipe(q.query_id)}\``);
      L();
      const meta: string[] = [];
      if (q.question_type) meta.push(`type: ${q.question_type}`);
      if (q.tenant_id) meta.push(`tenant: ${q.tenant_id}`);
      meta.push(`${q.time_taken.toFixed(2)}s`);
      L(`_${meta.join(" · ")}_`);
      L();
      L(`- **Question:** ${q.question}`);
      L(`- **Ground truth:** ${q.ground_truth_answer || "_(empty)_"}`);
      L(`- **System answer:** ${q.system_answer || "_(empty)_"}`);
      if (q.judge_reasoning) L(`- **Judge:** ${q.judge_reasoning}`);
      L();
    });
  } else {
    const searches = queries as SearchQuery[];

    L(`### Results Overview`);
    L();
    L(`| # | Hits | Retrieved | Time | Question |`);
    L(`| ---: | ---: | ---: | ---: | :--- |`);
    searches.forEach((q, i) => {
      const hits = relevantHits(q);
      L(
        `| ${i + 1} | ${hits}/${q.num_ground_truth} | ${q.num_retrieved} | ${q.time_taken.toFixed(2)}s | ${truncateCell(q.question)} |`,
      );
    });
    L();
    L(`_Hits = relevant documents retrieved of total ground-truth relevant._`);
    L();

    L(`### Query Details`);
    L();
    searches.forEach((q, i) => {
      const hits = relevantHits(q);
      const status = hits > 0 ? "✓ hit" : "✗ miss";
      L(`#### ${i + 1}. [${status}] \`${escPipe(q.query_id)}\``);
      L();
      L(`- **Question:** ${q.question}`);
      L(`- **Ground truth IDs (${q.num_ground_truth}):** ${q.ground_truth_ids.join(", ") || "—"}`);
      L(
        `- **Retrieved (${q.num_retrieved}) · ${hits} relevant · ${q.time_taken.toFixed(2)}s:**`,
      );
      const gt = new Set(q.ground_truth_ids);
      const shown = q.retrieved_ids.slice(0, 20).map((id) => (gt.has(id) ? `**${id}**` : id));
      const more = q.retrieved_ids.length > 20 ? ` … +${q.retrieved_ids.length - 20} more` : "";
      L(`  - Retrieved: ${shown.join(", ") || "—"}${more}`);
      const plan = q.searches ?? [];
      if (plan.length > 0) {
        const planStr = plan
          .map((s) => {
            const qq = s.query ? `"${s.query}"` : s.uuid_value ? `uuid:${s.uuid_value}` : "—";
            const filt = s.filters ? " +filter" : "";
            const sort = s.sort_property ? ` +sort(${s.sort_property.property_name})` : "";
            return `${s.collection}:${qq}${filt}${sort}`;
          })
          .join("; ");
        L(`  - Search plan (${plan.length}): ${planStr}`);
      }
      L();
    });
  }
}

/**
 * Build a detailed markdown report for a single experiment. When a representative
 * trial is provided, its per-query results are embedded.
 */
function buildExperimentReport(
  exp: ExperimentDetail,
  trial: ReportTrial | null,
  generatedAt: string,
): string {
  const lines: string[] = [];
  const L = (s = "") => lines.push(s);

  L(`# Experiment Report: ${exp.dataset}`);
  L();
  L(`_Generated ${generatedAt}_`);
  L();

  // ── Overview ───────────────────────────────────────────────────────────────
  L(`## Overview`);
  L();
  L(`- **Dataset:** ${exp.dataset}`);
  L(`- **Agent:** \`${exp.agent_name}\``);
  L(`- **Mode:** ${exp.mode}`);
  L(`- **Trials:** ${exp.num_trials}`);
  L(`- **Timestamp:** ${exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "—"}`);
  L(`- **ID:** \`${decodeURIComponent(exp.id)}\``);
  L();

  // ── Aggregated metrics ───────────────────────────────────────────────────────
  L(`## Aggregated Metrics`);
  L();
  if (exp.metricEntries.length === 0) {
    L(`_No aggregated metrics available._`);
    L();
  } else {
    L(`| Metric | Value |`);
    L(`| :--- | ---: |`);
    for (const { key, value } of exp.metricEntries) {
      L(`| ${escPipe(key.replace(/_/g, " "))} | ${escPipe(value)} |`);
    }
    L();
  }

  // ── Per-trial summary ─────────────────────────────────────────────────────────
  L(`## Trials`);
  L();
  if (exp.trials.length === 0) {
    L(`_No trial data available._`);
    L();
  } else {
    L(`| Trial | Queries | Avg Query Time | Key Score |`);
    L(`| :--- | ---: | ---: | ---: |`);
    for (const trial of exp.trials) {
      const totalQueries = trial.totalQueries ?? "—";
      const avgTime = trial.avgQueryTime != null ? `${trial.avgQueryTime.toFixed(2)}s` : "—";

      let keyScore = "—";
      if (trial.metrics) {
        for (const k of KEY_SCORE_KEYS) {
          if (typeof trial.metrics[k] === "number") {
            keyScore = `${((trial.metrics[k] as number) * 100).toFixed(1)}%`;
            break;
          }
        }
      }
      L(`| Trial ${trial.trialNumber} | ${totalQueries} | ${avgTime} | ${keyScore} |`);
    }
    L();
  }

  // ── Per-query results (representative trial) ───────────────────────────────
  if (trial) {
    appendPerQueryResults(L, trial, exp.num_trials);
  }

  L(`---`);
  L(`_Report generated by the Query Agent Benchmarking console._`);
  L();

  return lines.join("\n");
}

/** Trigger a client-side download of a markdown file. */
function downloadMarkdown(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Slugify a string for use in a filename. */
function slugify(s: string): string {
  return s.replace(/[^a-z0-9]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() || "experiment";
}

export default function ExperimentDetailPage({
  params: paramsPromise,
}: {
  params: Promise<{ id: string }>;
}) {
  const [id, setId] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    if (!experiment || exporting) return;
    setExporting(true);
    try {
      // Fetch the first trial with results so the report can embed per-query
      // detail. Best-effort — if it fails we still export the metrics + summary.
      let trial: ReportTrial | null = null;
      const firstWithResults = experiment.trials.find((t) => t.hasResults);
      if (firstWithResults) {
        try {
          const r = await fetch(
            `/api/trial?id=${experiment.id}&trial=${firstWithResults.trialNumber}`,
          );
          if (r.ok) {
            const data = (await r.json()) as TrialResultFile;
            trial = { trialNumber: firstWithResults.trialNumber, data };
          }
        } catch {
          // per-query detail is best-effort
        }
      }

      const now = new Date();
      const md = buildExperimentReport(experiment, trial, now.toLocaleString());
      const stamp = now.toISOString().slice(0, 10);
      downloadMarkdown(md, `experiment-${slugify(experiment.dataset)}-${stamp}.md`);
    } finally {
      setExporting(false);
    }
  };

  useEffect(() => {
    paramsPromise.then((p) => setId(p.id));
  }, [paramsPromise]);

  const fetchExperiment = useCallback(() => {
    if (!id) return;
    fetch(`/api/experiments/${id}`)
      .then((r) => {
        if (r.status === 404) { setNotFound(true); return null; }
        return r.json();
      })
      .then((data) => { if (data) setExperiment(data); })
      .catch(() => {});
  }, [id]);

  useEffect(() => {
    fetchExperiment();
    const interval = setInterval(fetchExperiment, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchExperiment]);

  if (notFound) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        Experiment not found.
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        <div className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3" style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }} />
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div>
      {/* ── Breadcrumb ───────────────────────────────────────────────────── */}
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/results" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>{experiment.dataset}</span>
      </nav>

      {/* ── Hero header ──────────────────────────────────────────────────── */}
      <div className="brand-card p-6 mb-8" style={{ background: "var(--gradient-navy)", color: "#fff" }}>
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
              {experiment.dataset}
            </h1>
            <div className="flex items-center gap-4 text-sm opacity-80">
              <span>
                Agent: <code style={{ fontFamily: "var(--font-mono)" }}>{experiment.agent_name}</code>
              </span>
              <span className="brand-badge" style={{
                background: experiment.mode === "search" ? "rgba(1,198,201,0.25)" : "rgba(165,144,221,0.25)",
                color: "#fff",
              }}>
                {experiment.mode}
              </span>
              <span>{experiment.num_trials} trial{experiment.num_trials !== 1 ? "s" : ""}</span>
            </div>
          </div>
          <div className="flex flex-col items-end gap-3">
            <button
              onClick={handleExport}
              disabled={exporting}
              className="brand-btn-primary"
              style={{ padding: "8px 18px", opacity: exporting ? 0.6 : 1 }}
              title="Download a detailed markdown report for this experiment"
            >
              {exporting ? "Exporting…" : "↓ Export Report"}
            </button>
            {experiment.timestamp && (
              <span className="text-xs opacity-50" style={{ fontFamily: "var(--font-mono)" }}>
                {new Date(experiment.timestamp).toLocaleString()}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── Aggregated metrics ───────────────────────────────────────────── */}
      {experiment.metricEntries.length > 0 && (
        <section className="mb-10">
          <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
            Aggregated Metrics
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {experiment.metricEntries.map(({ key, value }) => {
              const num = parseMetricValue(value);
              const isTime = key.toLowerCase().includes("time");
              const isPercentage = num !== null && !isTime && num <= 1.0;
              return (
                <div key={key} className="brand-card p-4">
                  <div className="eyebrow mb-2">
                    {key.replace(/_/g, " ")}
                  </div>
                  <div className="text-xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
                    {value}
                  </div>
                  {isPercentage && num !== null && (
                    <div className="mt-2 h-1.5 rounded-full" style={{ background: "var(--border-subtle)" }}>
                      <div
                        className="h-1.5 rounded-full transition-all"
                        style={{
                          width: `${Math.min(num * 100, 100)}%`,
                          background: num >= 0.8 ? "var(--color-green)" : num >= 0.5 ? "var(--color-cyan)" : "var(--color-coral)",
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Per-trial table ──────────────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
          Trials
        </h2>
        {experiment.trials.length === 0 ? (
          <p style={{ color: "var(--text-muted)" }}>No trial data available.</p>
        ) : (
          <div className="brand-card overflow-x-auto">
            <table className="w-full text-sm text-left brand-table">
              <thead>
                <tr>
                  <th>Trial</th>
                  <th>Queries</th>
                  <th>Avg Query Time</th>
                  <th>Key Score</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {experiment.trials.map((trial) => {
                  const totalQueries = trial.totalQueries ?? "--";
                  const avgTime =
                    trial.avgQueryTime != null
                      ? `${trial.avgQueryTime.toFixed(2)}s`
                      : "--";

                  let keyScore = "--";
                  if (trial.metrics) {
                    const m = trial.metrics;
                    for (const k of KEY_SCORE_KEYS) {
                      if (typeof m[k] === "number") {
                        keyScore = `${((m[k] as number) * 100).toFixed(1)}%`;
                        break;
                      }
                    }
                  }

                  return (
                    <tr key={trial.trialNumber}>
                      <td className="font-semibold">Trial {trial.trialNumber}</td>
                      <td>{totalQueries}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{avgTime}</td>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{keyScore}</td>
                      <td>
                        {trial.hasResults ? (
                          <a
                            href={`/results/experiments/${experiment.id}/trial/${trial.trialNumber}`}
                            className="text-xs font-semibold"
                            style={{ color: "var(--color-green)" }}
                          >
                            View queries
                          </a>
                        ) : (
                          <span className="text-xs" style={{ color: "var(--text-muted)" }}>No query data</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
