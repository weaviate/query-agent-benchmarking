"use client";

import { Fragment, Suspense, useEffect, useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import type {
  QueryComparison,
  QueryComparisonRow,
  ComparisonCell,
  ComparisonOutcome,
} from "@/lib/results";
import { SearchPlan } from "@/app/components/SearchPlan";

interface CompareExperiment {
  id: string;
  label: string;
  dataset: string;
  agent_name: string;
  mode: string;
  num_trials: number;
  timestamp: string;
  metrics: Record<string, number | null>;
}

interface CompareData {
  metricKeys: string[];
  experiments: CompareExperiment[];
}

function formatMetricName(key: string): string {
  let name = key.replace("avg_", "").replace("_mean", "");
  name = name.replace(/recall_at_(\d+)/g, "Recall@$1");
  name = name.replace(/nDCG_at_(\d+)/g, "nDCG@$1");
  name = name.replace("nDCG_at_k", "nDCG@10");
  name = name.replace(/alpha_nDCG_at_(\d+)/g, "alpha-nDCG@$1");
  name = name.replace(/coverage_at_(\d+)/g, "Coverage@$1");
  name = name.replace(/success_at_(\d+)/g, "Success@$1");
  name = name.replace("alignment_score", "Alignment");
  name = name.replace("exact_match_accuracy", "Exact Match");
  name = name.replace("officeqa_accuracy", "OfficeQA");
  name = name.replace("query_time", "Avg Time (s)");
  return name;
}

function isTimeMetric(key: string): boolean {
  return key.toLowerCase().includes("time");
}

function displayName(exp: CompareExperiment): string {
  return exp.label || `${exp.dataset} / ${exp.agent_name}`;
}

/* Brand-aligned experiment colors using secondary palette */
const EXP_COLORS = [
  { fg: "var(--color-teal)", bg: "rgba(122,199,192,0.12)" },
  { fg: "var(--color-lavender)", bg: "rgba(165,144,221,0.12)" },
  { fg: "var(--color-sky)", bg: "rgba(122,214,235,0.12)" },
  { fg: "var(--color-mint)", bg: "rgba(188,240,167,0.15)" },
  { fg: "var(--color-periwinkle)", bg: "rgba(185,200,222,0.15)" },
];

/* ═══════════════════════════════════════════════════════════════════════════
   Markdown report export
   ═══════════════════════════════════════════════════════════════════════════ */

type CompareAnalysis = {
  metricAnalysis: Record<
    string,
    { bestIdx: number | null; worstIdx: number | null; spread: number | null }
  >;
  wins: number[];
};

/** Escape pipe characters so user labels can't break markdown table rows. */
function escPipe(s: string): string {
  return s.replace(/\|/g, "\\|");
}

function formatMetricValue(key: string, val: number | null): string {
  if (val === null) return "—";
  return isTimeMetric(key) ? `${val.toFixed(2)}s` : `${(val * 100).toFixed(2)}%`;
}

/** Short column header for an experiment in the report (label preferred). */
function reportColLabel(exp: { label: string; agent_name: string }, i: number): string {
  return `[${i + 1}] ${escPipe(exp.label || exp.agent_name)}`;
}

/**
 * Build a detailed markdown comparison report from the aggregated compare data,
 * the derived analysis, and (optionally) the per-query comparison. Always
 * surfaces each experiment's user-assigned label prominently.
 */
function buildMarkdownReport(
  data: CompareData,
  analysis: CompareAnalysis | null,
  queries: QueryComparison | null,
  generatedAt: string,
): string {
  const { metricKeys, experiments } = data;
  const lines: string[] = [];
  const L = (s = "") => lines.push(s);

  L(`# Experiment Comparison Report`);
  L();
  L(`_Generated ${generatedAt} · comparing ${experiments.length} experiments_`);
  L();

  // ── Experiments ──────────────────────────────────────────────────────────
  L(`## Experiments`);
  L();
  experiments.forEach((exp, i) => {
    L(`### [${i + 1}] ${exp.label || `${exp.dataset} / ${exp.agent_name}`}`);
    L();
    L(`- **Label:** ${exp.label ? exp.label : "_(none assigned)_"}`);
    L(`- **Dataset:** ${exp.dataset}`);
    L(`- **Agent:** \`${exp.agent_name}\``);
    L(`- **Mode:** ${exp.mode}`);
    L(`- **Trials:** ${exp.num_trials}`);
    L(`- **Timestamp:** ${exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "—"}`);
    L(`- **ID:** \`${decodeURIComponent(exp.id)}\``);
    L();
  });

  // ── Metrics table ────────────────────────────────────────────────────────
  L(`## Metrics`);
  L();
  if (metricKeys.length === 0) {
    L(`_No aggregated metrics available._`);
    L();
  } else {
    const twoWay = experiments.length === 2;
    const header = ["Metric", ...experiments.map((e, i) => reportColLabel(e, i))];
    if (twoWay) header.push("Delta");
    L(`| ${header.join(" | ")} |`);
    L(`| ${header.map((_, i) => (i === 0 ? ":---" : "---:")).join(" | ")} |`);

    for (const key of metricKeys) {
      const isTime = isTimeMetric(key);
      const best = analysis?.metricAnalysis[key]?.bestIdx ?? null;
      const cells = [escPipe(formatMetricName(key))];

      experiments.forEach((exp, i) => {
        const val = exp.metrics[key];
        let cell = formatMetricValue(key, val);
        if (val !== null && i === best && experiments.length > 1) cell = `**${cell}** ⭐`;
        cells.push(cell);
      });

      if (twoWay) {
        const v0 = experiments[0].metrics[key];
        const v1 = experiments[1].metrics[key];
        if (v0 === null || v1 === null) {
          cells.push("—");
        } else {
          const delta = v1 - v0;
          if (Math.abs(delta) < 1e-9) {
            cells.push("0");
          } else {
            const isImprovement = isTime ? delta < 0 : delta > 0;
            const sign = delta > 0 ? "+" : "";
            const formatted = isTime
              ? `${sign}${delta.toFixed(2)}s`
              : `${sign}${(delta * 100).toFixed(2)}%`;
            cells.push(`${formatted} ${isImprovement ? "▲" : "▼"}`);
          }
        }
      }

      L(`| ${cells.join(" | ")} |`);
    }
    L();
    L(`⭐ marks the best value for each metric. ${twoWay ? "Delta = [2] − [1] (▲ improvement, ▼ regression)." : ""}`);
    L();
  }

  // ── Analysis ─────────────────────────────────────────────────────────────
  if (analysis) {
    L(`## Analysis`);
    L();

    const nonTime = metricKeys.filter((k) => !isTimeMetric(k));
    L(`### Metric Wins (excluding time)`);
    L();
    experiments.forEach((exp, i) => {
      const w = analysis.wins[i];
      const pct = nonTime.length > 0 ? ((w / nonTime.length) * 100).toFixed(0) : "0";
      L(`- **${reportColLabel(exp, i)}:** ${w} of ${nonTime.length} (${pct}%)`);
    });
    L();

    // Speed comparison (only meaningful for a 2-way comparison).
    if (experiments.length === 2) {
      const timeKey = metricKeys.find((k) => isTimeMetric(k));
      const t0 = timeKey ? experiments[0].metrics[timeKey] : null;
      const t1 = timeKey ? experiments[1].metrics[timeKey] : null;
      if (t0 !== null && t1 !== null) {
        const faster = t0 < t1 ? 0 : 1;
        const slower = 1 - faster;
        const speedup = ((Math.max(t0, t1) - Math.min(t0, t1)) / Math.max(t0, t1)) * 100;
        if (speedup >= 0.1) {
          L(`### Speed`);
          L();
          L(
            `**${reportColLabel(experiments[faster], faster)}** is **${speedup.toFixed(1)}% faster** than ` +
              `**${reportColLabel(experiments[slower], slower)}** ` +
              `(${Math.min(t0, t1).toFixed(2)}s vs ${Math.max(t0, t1).toFixed(2)}s).`,
          );
          L();
        }
      }
    }

    // Largest differences.
    const diffs = nonTime
      .map((k) => ({ key: k, spread: analysis.metricAnalysis[k]?.spread ?? null }))
      .filter((d) => d.spread !== null && d.spread > 0)
      .sort((a, b) => (b.spread ?? 0) - (a.spread ?? 0))
      .slice(0, 5);
    if (diffs.length > 0) {
      L(`### Largest Differences`);
      L();
      for (const d of diffs) {
        L(`- **${escPipe(formatMetricName(d.key))}:** ${(d.spread! * 100).toFixed(2)}% spread`);
      }
      L();
    }
  }

  // ── Per-query analysis ─────────────────────────────────────────────────────
  if (queries && !queries.warning && queries.rows.length > 0) {
    const { counts, sharedByAll, totalQuestions, mode } = queries;
    const qExps = queries.experiments;

    // Outcome label + sort weight (disagreements first, then all-wrong, then all-correct).
    const outcomeLabel: Record<ComparisonOutcome, string> = {
      mixed: "Disagreement",
      all_wrong: "All wrong",
      all_correct: "All correct",
    };

    /** Compact one-cell status used in the overview table. */
    const cellStatus = (cell: ComparisonCell | null): string => {
      if (!cell) return "—";
      if (mode === "search") return cell.correct ? `${cell.overlap ?? 0} hit` : "0 hit";
      if (cell.is_error) return "err";
      if (cell.correct === null) return "?";
      return cell.correct ? "✓" : "✗";
    };

    L(`## Per-Query Analysis`);
    L();
    L(`Queries aligned by question text (mode: ${mode}).`);
    L();
    L(`- **Comparable** (present in ≥2 experiments): ${counts.comparable}`);
    L(`- **Shared by all:** ${sharedByAll}`);
    L(`- **Total distinct questions:** ${totalQuestions}`);
    L(`- **All correct:** ${counts.allCorrect}`);
    L(`- **All wrong:** ${counts.allWrong}`);
    L(`- **Disagreements:** ${counts.mixed}`);
    L();

    // Comparable rows are already sorted disagreements-first by buildQueryComparison.
    const comparable = queries.rows.filter((r) => r.presentCount >= 2);
    const singleOnly = queries.rows.length - comparable.length;

    if (comparable.length > 0) {
      // ── Overview table — every comparable query at a glance ────────────────
      L(`### Results Overview`);
      L();
      const head = ["#", "Outcome", ...qExps.map((e, i) => reportColLabel(e, i)), "Question"];
      L(`| ${head.join(" | ")} |`);
      L(`| ${head.map((_, i) => (i === 0 ? "---:" : ":---")).join(" | ")} |`);
      comparable.forEach((row, idx) => {
        const statusCells = row.cells.map((c) => cellStatus(c));
        const q = escPipe(row.question.replace(/\s+/g, " ").trim());
        const qShort = q.length > 100 ? `${q.slice(0, 100)}…` : q;
        L(`| ${idx + 1} | ${outcomeLabel[row.outcome]} | ${statusCells.join(" | ")} | ${qShort} |`);
      });
      L();
      if (mode === "search") {
        L(`_Cells show relevant-document hits per experiment._`);
      } else {
        L(`_Cells: ✓ correct · ✗ wrong · ? unscored · err error._`);
      }
      L();

      // ── Full detail for every comparable query ─────────────────────────────
      L(`### Query Details`);
      L();
      comparable.forEach((row, idx) => {
        L(`#### ${idx + 1}. [${outcomeLabel[row.outcome]}] ${row.question}`);
        L();
        if (mode === "ask" && row.ground_truth_answer) {
          L(`**Ground truth:** ${row.ground_truth_answer}`);
          L();
        }
        if (mode === "search" && row.ground_truth_ids) {
          L(
            `**Ground truth IDs (${row.ground_truth_ids.length}):** ${row.ground_truth_ids.join(", ") || "—"}`,
          );
          L();
        }

        row.cells.forEach((cell, i) => {
          const name = reportColLabel(qExps[i], i);
          if (!cell) {
            L(`- ${name}: _not present in this experiment_`);
            return;
          }
          const time = cell.time_taken != null ? ` · ${cell.time_taken.toFixed(2)}s` : "";

          if (mode === "ask") {
            const status = cell.is_error
              ? "error"
              : cell.correct === true
                ? "✓ correct"
                : cell.correct === null
                  ? "? unscored"
                  : "✗ wrong";
            L(`- **${name}** — ${status}${time}`);
            L(`  - Answer: ${cell.system_answer ? cell.system_answer : "_(empty)_"}`);
            if (cell.judge_reasoning) L(`  - Judge: ${cell.judge_reasoning}`);
          } else {
            const status = cell.correct ? "✓ hit" : "✗ miss";
            const gt = new Set(row.ground_truth_ids ?? []);
            const retrieved = cell.retrieved_ids ?? [];
            L(
              `- **${name}** — ${status} · ${cell.overlap ?? 0} relevant of ${cell.num_retrieved ?? retrieved.length} retrieved${time}`,
            );
            if (retrieved.length > 0) {
              const shown = retrieved
                .slice(0, 20)
                .map((id) => (gt.has(id) ? `**${id}**` : id));
              const more = retrieved.length > 20 ? ` … +${retrieved.length - 20} more` : "";
              L(`  - Retrieved: ${shown.join(", ")}${more}`);
            }
            if (cell.searches && cell.searches.length > 0) {
              const plan = cell.searches
                .map((s) => {
                  const q = s.query ? `"${s.query}"` : s.uuid_value ? `uuid:${s.uuid_value}` : "—";
                  const filt = s.filters ? " +filter" : "";
                  const sort = s.sort_property ? ` +sort(${s.sort_property.property_name})` : "";
                  return `${s.collection}:${q}${filt}${sort}`;
                })
                .join("; ");
              L(`  - Search plan (${cell.searches.length}): ${plan}`);
            }
          }
        });
        L();
      });

      if (singleOnly > 0) {
        L(
          `_${singleOnly} additional ${singleOnly === 1 ? "query was" : "queries were"} present in only one experiment and omitted from the comparison._`,
        );
        L();
      }
    }
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

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>Loading comparison...</div>
      }
    >
      <ComparePageInner />
    </Suspense>
  );
}

function ComparePageInner() {
  const searchParams = useSearchParams();
  const ids = searchParams.get("ids") ?? "";
  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (!ids) {
      setError("No experiment IDs provided.");
      setLoading(false);
      return;
    }
    fetch(`/api/compare?ids=${ids}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load comparison data");
        return r.json();
      })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [ids]);

  const analysis = useMemo(() => {
    if (!data) return null;
    const { metricKeys, experiments } = data;
    const metricAnalysis: Record<
      string,
      { bestIdx: number | null; worstIdx: number | null; spread: number | null }
    > = {};

    for (const key of metricKeys) {
      const values = experiments.map((e) => e.metrics[key]);
      const numeric = values
        .map((v, i) => (v !== null ? { v, i } : null))
        .filter(Boolean) as { v: number; i: number }[];

      if (numeric.length < 2) {
        metricAnalysis[key] = { bestIdx: null, worstIdx: null, spread: null };
        continue;
      }

      const isTime = isTimeMetric(key);
      numeric.sort((a, b) => (isTime ? a.v - b.v : b.v - a.v));
      metricAnalysis[key] = {
        bestIdx: numeric[0].i,
        worstIdx: numeric[numeric.length - 1].i,
        spread: Math.abs(numeric[0].v - numeric[numeric.length - 1].v),
      };
    }

    const wins = experiments.map(() => 0);
    for (const key of metricKeys) {
      if (isTimeMetric(key)) continue;
      const best = metricAnalysis[key].bestIdx;
      if (best !== null) wins[best]++;
    }

    return { metricAnalysis, wins };
  }, [data]);

  const handleExport = async () => {
    if (!data || exporting) return;
    setExporting(true);
    try {
      // Fetch per-query data so the report is complete regardless of whether
      // the user expanded the per-query section. It's optional — if it fails
      // we still export the metrics + analysis.
      let queries: QueryComparison | null = null;
      try {
        const r = await fetch(`/api/compare/queries?ids=${ids}`);
        if (r.ok) queries = (await r.json()) as QueryComparison;
      } catch {
        // per-query analysis is best-effort
      }

      const now = new Date();
      const md = buildMarkdownReport(data, analysis, queries, now.toLocaleString());
      const stamp = now.toISOString().slice(0, 10);
      downloadMarkdown(md, `experiment-comparison-${stamp}.md`);
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        <div className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3" style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }} />
        <p>Loading comparison...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-20 text-center">
        <p className="mb-4" style={{ color: "var(--color-coral)" }}>{error ?? "Unknown error"}</p>
        <a href="/results" className="text-sm font-semibold" style={{ color: "var(--color-green)" }}>
          &larr; Back to dashboard
        </a>
      </div>
    );
  }

  const { metricKeys, experiments } = data;

  return (
    <div>
      {/* ── Breadcrumb ───────────────────────────────────────────────────── */}
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/results" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Compare</span>
      </nav>

      <div className="flex items-start justify-between gap-4 mb-6 flex-wrap">
        <h1 className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
          Compare Experiments
        </h1>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="brand-btn-primary"
          style={{ padding: "8px 18px", opacity: exporting ? 0.6 : 1 }}
          title="Download a detailed markdown comparison report"
        >
          {exporting ? "Exporting…" : "↓ Export Report"}
        </button>
      </div>

      {/* ── Experiment legend ─────────────────────────────────────────────── */}
      <div
        className="grid gap-4 mb-8"
        style={{ gridTemplateColumns: `repeat(${Math.min(experiments.length, 3)}, 1fr)` }}
      >
        {experiments.map((exp, i) => {
          const c = EXP_COLORS[i % EXP_COLORS.length];
          return (
            <div
              key={exp.id}
              className="brand-card p-5"
              style={{ borderLeft: `3px solid ${c.fg.replace("var(", "").replace(")", "")}` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="brand-badge"
                  style={{ background: c.bg, color: c.fg, fontWeight: 700 }}
                >
                  {i + 1}
                </span>
                <span className="font-semibold text-sm" style={{ fontFamily: "var(--font-display)" }}>
                  {displayName(exp)}
                </span>
              </div>
              {exp.label && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {exp.dataset} &middot;{" "}
                  <span style={{ fontFamily: "var(--font-mono)" }}>{exp.agent_name}</span>
                </p>
              )}
              {!exp.label && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Agent: <span style={{ fontFamily: "var(--font-mono)" }}>{exp.agent_name}</span>
                </p>
              )}
              <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                {exp.num_trials} trials &middot;{" "}
                {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
              </p>
            </div>
          );
        })}
      </div>

      {/* ── Metrics comparison table ─────────────────────────────────────── */}
      <section className="mb-10">
        <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
          Metrics
        </h2>
        <div className="brand-card overflow-x-auto">
          <table className="w-full text-sm text-left brand-table">
            <thead>
              <tr>
                <th>Metric</th>
                {experiments.map((exp, i) => {
                  const c = EXP_COLORS[i % EXP_COLORS.length];
                  return (
                    <th key={exp.id} className="text-right">
                      <span className="brand-badge mr-1" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                        {i + 1}
                      </span>
                      {exp.label || exp.agent_name}
                    </th>
                  );
                })}
                {experiments.length === 2 && <th className="text-right">Delta</th>}
              </tr>
            </thead>
            <tbody>
              {metricKeys.map((key) => {
                const isTime = isTimeMetric(key);
                const best = analysis?.metricAnalysis[key].bestIdx ?? null;

                return (
                  <tr key={key}>
                    <td className="font-semibold">{formatMetricName(key)}</td>
                    {experiments.map((exp, i) => {
                      const val = exp.metrics[key];
                      if (val === null) {
                        return (
                          <td key={i} className="text-right" style={{ color: "var(--text-muted)" }}>
                            --
                          </td>
                        );
                      }
                      const isBest = i === best && experiments.length > 1;
                      const formatted = isTime ? val.toFixed(2) : (val * 100).toFixed(2) + "%";
                      return (
                        <td
                          key={i}
                          className="text-right"
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontWeight: isBest ? 700 : 400,
                            color: isBest ? "var(--color-green)" : "var(--text-primary)",
                          }}
                        >
                          {formatted}
                        </td>
                      );
                    })}
                    {experiments.length === 2 &&
                      (() => {
                        const v0 = experiments[0].metrics[key];
                        const v1 = experiments[1].metrics[key];
                        if (v0 === null || v1 === null) {
                          return (
                            <td className="text-right" style={{ color: "var(--text-muted)" }}>
                              --
                            </td>
                          );
                        }
                        const delta = v1 - v0;
                        if (Math.abs(delta) < 1e-9) {
                          return (
                            <td
                              className="text-right"
                              style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}
                            >
                              0
                            </td>
                          );
                        }
                        const isImprovement = isTime ? delta < 0 : delta > 0;
                        const sign = delta > 0 ? "+" : "";
                        const formatted = isTime
                          ? `${sign}${delta.toFixed(2)}s`
                          : `${sign}${(delta * 100).toFixed(2)}%`;
                        return (
                          <td
                            className="text-right"
                            style={{
                              fontFamily: "var(--font-mono)",
                              fontWeight: 600,
                              color: isImprovement ? "var(--color-green)" : "var(--color-coral)",
                            }}
                          >
                            {formatted}
                          </td>
                        );
                      })()}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Analysis section ──────────────────────────────────────────────── */}
      {analysis && (
        <section>
          <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
            Analysis
          </h2>

          {/* Win counts */}
          <div className="brand-card p-6 mb-6">
            <h3 className="eyebrow mb-4">Metric Wins (excluding time)</h3>
            <div className="flex gap-6">
              {experiments.map((exp, i) => {
                const c = EXP_COLORS[i % EXP_COLORS.length];
                const total = metricKeys.filter((k) => !isTimeMetric(k)).length;
                const winPct = total > 0 ? (analysis.wins[i] / total) * 100 : 0;
                return (
                  <div key={exp.id} className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className="brand-badge"
                        style={{ background: c.bg, color: c.fg, fontWeight: 700 }}
                      >
                        {i + 1}
                      </span>
                      <span className="text-sm font-semibold">{displayName(exp)}</span>
                    </div>
                    <div className="text-3xl font-bold" style={{ fontFamily: "var(--font-display)", color: c.fg }}>
                      {analysis.wins[i]}
                    </div>
                    <div className="w-full h-2 rounded-full mt-2" style={{ background: "var(--border-subtle)" }}>
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{ width: `${winPct}%`, background: c.fg }}
                      />
                    </div>
                    <div className="text-xs mt-1" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {winPct.toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Speed comparison */}
          {experiments.length === 2 &&
            (() => {
              const timeKey = metricKeys.find((k) => isTimeMetric(k));
              if (!timeKey) return null;
              const t0 = experiments[0].metrics[timeKey];
              const t1 = experiments[1].metrics[timeKey];
              if (t0 === null || t1 === null) return null;

              const faster = t0 < t1 ? 0 : 1;
              const slower = 1 - faster;
              const speedup =
                ((Math.max(t0, t1) - Math.min(t0, t1)) / Math.max(t0, t1)) * 100;

              if (speedup < 0.1) return null;

              const cFaster = EXP_COLORS[faster % EXP_COLORS.length];
              const cSlower = EXP_COLORS[slower % EXP_COLORS.length];

              return (
                <div className="brand-card p-6 mb-6">
                  <h3 className="eyebrow mb-3">Speed</h3>
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    <span className="font-bold" style={{ color: cFaster.fg }}>
                      [{faster + 1}] {displayName(experiments[faster])}
                    </span>{" "}
                    is{" "}
                    <span className="font-bold" style={{ color: "var(--color-green)" }}>
                      {speedup.toFixed(1)}% faster
                    </span>{" "}
                    than{" "}
                    <span className="font-bold" style={{ color: cSlower.fg }}>
                      [{slower + 1}] {displayName(experiments[slower])}
                    </span>{" "}
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      ({Math.min(t0, t1).toFixed(2)}s vs {Math.max(t0, t1).toFixed(2)}s)
                    </span>
                  </p>
                </div>
              );
            })()}

          {/* Biggest differences */}
          {(() => {
            const diffs = metricKeys
              .filter((k) => !isTimeMetric(k))
              .map((k) => ({
                key: k,
                spread: analysis.metricAnalysis[k].spread,
              }))
              .filter((d) => d.spread !== null && d.spread > 0)
              .sort((a, b) => (b.spread ?? 0) - (a.spread ?? 0));

            if (diffs.length === 0) return null;

            return (
              <div className="brand-card p-6">
                <h3 className="eyebrow mb-4">Largest Differences</h3>
                <div className="space-y-3">
                  {diffs.slice(0, 5).map((d) => (
                    <div key={d.key} className="flex items-center gap-3">
                      <span className="text-sm font-semibold w-40">{formatMetricName(d.key)}</span>
                      <div className="flex-1 h-2 rounded-full" style={{ background: "var(--border-subtle)" }}>
                        <div
                          className="h-2 rounded-full"
                          style={{
                            width: `${Math.min((d.spread! / 0.5) * 100, 100)}%`,
                            background: "var(--gradient-teal-green)",
                          }}
                        />
                      </div>
                      <span
                        className="text-xs w-20 text-right"
                        style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}
                      >
                        {(d.spread! * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </section>
      )}

      {/* ── Per-query analysis ────────────────────────────────────────────── */}
      <PerQuerySection ids={ids} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Per-query analysis: what each experiment predicted for individual queries.
   Loaded on demand (trial files are large) and aligned by question text.
   ═══════════════════════════════════════════════════════════════════════════ */

type QueryFilter = "all" | "mixed" | "all_correct" | "all_wrong";

const OUTCOME_META: Record<
  ComparisonOutcome,
  { label: string; fg: string; bg: string }
> = {
  mixed: { label: "Disagreement", fg: "var(--color-coral)", bg: "rgba(244,64,78,0.12)" },
  all_correct: { label: "All correct", fg: "var(--color-green)", bg: "rgba(97,189,115,0.15)" },
  all_wrong: { label: "All wrong", fg: "#b8a900", bg: "rgba(249,241,93,0.15)" },
};

function PerQuerySection({ ids }: { ids: string }) {
  const [data, setData] = useState<QueryComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const [filter, setFilter] = useState<QueryFilter>("all");
  const [search, setSearch] = useState("");

  const load = () => {
    setLoading(true);
    setError(null);
    fetch(`/api/compare/queries?ids=${ids}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load per-query comparison");
        return r.json();
      })
      .then((d: QueryComparison) => {
        setData(d);
        setLoaded(true);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  };

  const filteredRows = useMemo(() => {
    if (!data) return [];
    const term = search.trim().toLowerCase();
    return data.rows.filter((row) => {
      if (row.presentCount < 2) return false; // only comparable rows
      if (filter !== "all" && row.outcome !== filter) return false;
      if (term && !row.question.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [data, filter, search]);

  return (
    <section className="mt-10">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h2 className="text-lg font-bold" style={{ fontFamily: "var(--font-display)" }}>
          Per-Query Analysis
        </h2>
        {!loaded && !loading && (
          <button onClick={load} className="brand-btn-primary" style={{ padding: "6px 16px" }}>
            Load per-query analysis
          </button>
        )}
      </div>

      {!loaded && !loading && (
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Compare what each experiment predicted for individual queries, aligned by question text.
          Surfaces disagreements where one experiment succeeds and another fails.
        </p>
      )}

      {loading && (
        <div className="py-10 text-center" style={{ color: "var(--text-muted)" }}>
          <div
            className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3"
            style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
          />
          <p>Loading per-query data...</p>
        </div>
      )}

      {error && (
        <p className="text-sm" style={{ color: "var(--color-coral)" }}>
          {error}
        </p>
      )}

      {data && data.warning && (
        <div
          className="brand-card p-4 mb-4 text-sm"
          style={{ color: "#b8a900", background: "rgba(249,241,93,0.08)" }}
        >
          {data.warning}
        </div>
      )}

      {data && !data.warning && data.rows.length === 0 && (
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          No overlapping queries found across these experiments (they may use different datasets).
        </p>
      )}

      {data && !data.warning && data.rows.length > 0 && (
        <PerQueryBody
          data={data}
          filteredRows={filteredRows}
          filter={filter}
          setFilter={setFilter}
          search={search}
          setSearch={setSearch}
        />
      )}
    </section>
  );
}

function PerQueryBody({
  data,
  filteredRows,
  filter,
  setFilter,
  search,
  setSearch,
}: {
  data: QueryComparison;
  filteredRows: QueryComparisonRow[];
  filter: QueryFilter;
  setFilter: (f: QueryFilter) => void;
  search: string;
  setSearch: (s: string) => void;
}) {
  const { experiments, counts, sharedByAll, totalQuestions, mode } = data;

  const summary: { key: QueryFilter; label: string; value: number; fg: string }[] = [
    { key: "mixed", label: "Disagreements", value: counts.mixed, fg: "var(--color-coral)" },
    { key: "all_correct", label: "All correct", value: counts.allCorrect, fg: "var(--color-green)" },
    { key: "all_wrong", label: "All wrong", value: counts.allWrong, fg: "#b8a900" },
  ];

  return (
    <div>
      {/* Summary strip */}
      <div className="grid gap-4 mb-5" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="brand-card p-4">
          <div className="eyebrow mb-1">Comparable</div>
          <div className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
            {counts.comparable}
          </div>
          <div className="text-xs mt-1" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {sharedByAll} in all · {totalQuestions} total
          </div>
        </div>
        {summary.map((s) => (
          <button
            key={s.key}
            onClick={() => setFilter(filter === s.key ? "all" : s.key)}
            className="brand-card p-4 text-left transition-all"
            style={{
              cursor: "pointer",
              outline: filter === s.key ? `2px solid ${s.fg}` : "none",
            }}
          >
            <div className="eyebrow mb-1">{s.label}</div>
            <div className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)", color: s.fg }}>
              {s.value}
            </div>
            <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
              {filter === s.key ? "filtering" : "click to filter"}
            </div>
          </button>
        ))}
      </div>

      {/* Controls */}
      <div
        className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3 mb-4"
        style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
      >
        <span className="eyebrow mr-2">Filter</span>
        {(["all", "mixed", "all_correct", "all_wrong"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className="brand-btn-secondary"
            style={
              filter === f
                ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" }
                : {}
            }
          >
            {f === "all" ? "all" : f === "mixed" ? "disagreements" : f === "all_correct" ? "all correct" : "all wrong"}
          </button>
        ))}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search question..."
          className="rounded-md px-2.5 py-1 text-xs ml-2"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-default)",
            color: "var(--text-primary)",
            width: "220px",
          }}
        />
        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {filteredRows.length} of {counts.comparable}
        </span>
      </div>

      {/* Comparison table */}
      <div className="brand-card overflow-x-auto">
        <table className="w-full text-sm text-left brand-table">
          <thead>
            <tr>
              <th style={{ width: "1%" }}></th>
              <th>Question</th>
              {experiments.map((exp, i) => {
                const c = EXP_COLORS[i % EXP_COLORS.length];
                return (
                  <th key={exp.id} className="text-center whitespace-nowrap">
                    <span className="brand-badge mr-1" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                      {i + 1}
                    </span>
                    {exp.label || exp.agent_name}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row, ri) => (
              <ComparisonRowView key={ri} row={row} mode={mode} experiments={experiments} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonRowView({
  row,
  mode,
  experiments,
}: {
  row: QueryComparisonRow;
  mode: QueryComparison["mode"];
  experiments: QueryComparison["experiments"];
}) {
  const [open, setOpen] = useState(false);
  const meta = OUTCOME_META[row.outcome];
  const colSpan = 2 + experiments.length;

  return (
    <Fragment>
      <tr
        onClick={() => setOpen((v) => !v)}
        style={{
          cursor: "pointer",
          background: row.outcome === "mixed" ? "rgba(244,64,78,0.05)" : undefined,
        }}
      >
        <td style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <span
            style={{
              display: "inline-block",
              transition: "transform 0.15s",
              transform: open ? "rotate(90deg)" : "rotate(0deg)",
              fontFamily: "var(--font-mono)",
            }}
          >
            ▶
          </span>
        </td>
        <td className="max-w-md">
          <div className="flex items-center gap-2">
            <span
              className="brand-badge shrink-0"
              style={{ background: meta.bg, color: meta.fg, fontSize: "0.65rem" }}
              title={meta.label}
            >
              {row.outcome === "mixed" ? "≠" : row.outcome === "all_correct" ? "✓" : "✗"}
            </span>
            <span className="truncate">{row.question}</span>
          </div>
        </td>
        {row.cells.map((cell, i) => (
          <td key={i} className="text-center">
            <CellBadge cell={cell} mode={mode} />
          </td>
        ))}
      </tr>
      {open && (
        <tr>
          <td colSpan={colSpan} style={{ padding: 0 }}>
            <div
              className="px-4 py-4"
              style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-subtle)" }}
            >
              <RowDetail row={row} mode={mode} experiments={experiments} />
            </div>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

/** Compact per-experiment cell shown in the main comparison row. */
function CellBadge({ cell, mode }: { cell: ComparisonCell | null; mode: QueryComparison["mode"] }) {
  if (!cell) {
    return <span style={{ color: "var(--text-muted)" }}>—</span>;
  }
  const ok = cell.correct === true;
  const unknown = cell.correct === null;
  const fg = unknown ? "var(--text-muted)" : ok ? "var(--color-green)" : "var(--color-coral)";
  const bg = unknown ? "var(--border-subtle)" : ok ? "rgba(97,189,115,0.15)" : "rgba(244,64,78,0.12)";

  let label: string;
  if (mode === "search") {
    const gt = cell.overlap ?? 0;
    label = unknown ? "?" : `${gt} hit`;
  } else {
    label = cell.is_error ? "err" : unknown ? "?" : ok ? "✓" : "✗";
  }

  return (
    <span className="brand-badge" style={{ background: bg, color: fg, fontWeight: 600 }}>
      {label}
    </span>
  );
}

/** Expanded full-width detail comparing each experiment's prediction. */
function RowDetail({
  row,
  mode,
  experiments,
}: {
  row: QueryComparison["rows"][number];
  mode: QueryComparison["mode"];
  experiments: QueryComparison["experiments"];
}) {
  return (
    <div className="space-y-4">
      {/* Ground truth */}
      <div>
        <div className="eyebrow mb-1">Question</div>
        <p className="text-sm mb-3" style={{ color: "var(--text-primary)" }}>{row.question}</p>
        {mode === "ask" && row.ground_truth_answer && (
          <>
            <div className="eyebrow mb-1">Ground Truth Answer</div>
            <p className="text-sm rounded-md p-3" style={{ background: "var(--bg-surface)", color: "var(--text-primary)" }}>
              {row.ground_truth_answer}
            </p>
          </>
        )}
        {mode === "search" && row.ground_truth_ids && (
          <div className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            Ground truth ({row.ground_truth_ids.length}): {row.ground_truth_ids.join(", ")}
          </div>
        )}
      </div>

      {/* Per-experiment predictions */}
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(row.cells.length, 2)}, 1fr)` }}>
        {row.cells.map((cell, i) => {
          const c = EXP_COLORS[i % EXP_COLORS.length];
          return (
            <div
              key={i}
              className="rounded-lg p-4"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)" }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="brand-badge" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                  {i + 1}
                </span>
                <span className="text-xs font-semibold" style={{ fontFamily: "var(--font-display)" }}>
                  {experiments[i]?.label || experiments[i]?.agent_name}
                </span>
                {cell ? (
                  <CellBadge cell={cell} mode={mode} />
                ) : (
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>not present</span>
                )}
                {cell?.time_taken != null && (
                  <span className="text-xs ml-auto" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {cell.time_taken.toFixed(2)}s
                  </span>
                )}
              </div>
              {cell && <CellDetail cell={cell} mode={mode} groundTruthIds={row.ground_truth_ids} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CellDetail({
  cell,
  mode,
  groundTruthIds,
}: {
  cell: ComparisonCell;
  mode: QueryComparison["mode"];
  groundTruthIds?: string[];
}) {
  if (mode === "ask") {
    return (
      <div>
        <div className="eyebrow mb-1">Answer</div>
        <p className="text-sm rounded-md p-2 mb-2" style={{ background: "var(--bg-card)", color: "var(--text-primary)" }}>
          {cell.system_answer || <em style={{ color: "var(--text-muted)" }}>(empty)</em>}
        </p>
        {cell.judge_reasoning && (
          <details>
            <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              Judge reasoning
            </summary>
            <p className="mt-1 text-xs rounded-md p-2" style={{ background: "var(--bg-card)", color: "var(--text-secondary)" }}>
              {cell.judge_reasoning}
            </p>
          </details>
        )}
      </div>
    );
  }

  // search mode
  const gt = new Set(groundTruthIds ?? []);
  const retrieved = cell.retrieved_ids ?? [];
  return (
    <div className="space-y-3">
      <div>
        <div className="eyebrow mb-1">
          Retrieved ({cell.num_retrieved ?? retrieved.length}) · {cell.overlap ?? 0} relevant
        </div>
        <div className="flex flex-wrap gap-1">
          {retrieved.length === 0 && (
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>none</span>
          )}
          {retrieved.slice(0, 20).map((id, i) => {
            const hit = gt.has(id);
            return (
              <span
                key={i}
                className="inline-flex items-center rounded px-1.5 py-0.5"
                style={{
                  background: hit ? "rgba(97,189,115,0.18)" : "var(--bg-card)",
                  color: hit ? "var(--color-green)" : "var(--text-muted)",
                  border: `1px solid ${hit ? "rgba(97,189,115,0.4)" : "var(--border-subtle)"}`,
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.65rem",
                }}
                title={hit ? "relevant" : undefined}
              >
                {id}
              </span>
            );
          })}
          {retrieved.length > 20 && (
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>+{retrieved.length - 20} more</span>
          )}
        </div>
      </div>
      {cell.searches != null && cell.searches.length > 0 && (
        <div>
          <div className="eyebrow mb-1">Search Plan</div>
          <SearchPlan searches={cell.searches} />
        </div>
      )}
    </div>
  );
}
