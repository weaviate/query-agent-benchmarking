"use client";

import { Suspense, useEffect, useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";

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
        <a href="/" className="text-sm font-semibold" style={{ color: "var(--color-green)" }}>
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
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Compare</span>
      </nav>

      <h1 className="text-2xl font-bold mb-6" style={{ fontFamily: "var(--font-display)" }}>
        Compare Experiments
      </h1>

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
    </div>
  );
}
