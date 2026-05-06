"use client";

import { useEffect, useState, useCallback } from "react";

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
  "avg_recall@5",
  "avg_nDCG@10",
];

function parseMetricValue(value: string): number | null {
  const num = parseFloat(value);
  return isNaN(num) ? null : num;
}

export default function ExperimentDetailPage({
  params: paramsPromise,
}: {
  params: Promise<{ id: string }>;
}) {
  const [id, setId] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [notFound, setNotFound] = useState(false);

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
                background: experiment.mode === "search" ? "rgba(122,199,192,0.25)" : "rgba(165,144,221,0.25)",
                color: "#fff",
              }}>
                {experiment.mode}
              </span>
              <span>{experiment.num_trials} trial{experiment.num_trials !== 1 ? "s" : ""}</span>
            </div>
          </div>
          {experiment.timestamp && (
            <span className="text-xs opacity-50" style={{ fontFamily: "var(--font-mono)" }}>
              {new Date(experiment.timestamp).toLocaleString()}
            </span>
          )}
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
                          background: num >= 0.8 ? "var(--color-green)" : num >= 0.5 ? "var(--color-teal)" : "var(--color-coral)",
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
