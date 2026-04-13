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

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="py-20 text-center text-gray-500">Loading comparison...</div>}>
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
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [ids]);

  // Compute analysis for each metric
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

    // Win counts (excluding time)
    const wins = experiments.map(() => 0);
    for (const key of metricKeys) {
      if (isTimeMetric(key)) continue;
      const best = metricAnalysis[key].bestIdx;
      if (best !== null) wins[best]++;
    }

    return { metricAnalysis, wins };
  }, [data]);

  if (loading) {
    return <div className="py-20 text-center text-gray-500">Loading comparison...</div>;
  }

  if (error || !data) {
    return (
      <div className="py-20 text-center">
        <p className="text-red-500 mb-4">{error ?? "Unknown error"}</p>
        <a href="/" className="text-blue-600 dark:text-blue-400 hover:underline text-sm">
          &larr; Back to dashboard
        </a>
      </div>
    );
  }

  const { metricKeys, experiments } = data;
  const colors = [
    "text-blue-600 dark:text-blue-400",
    "text-purple-600 dark:text-purple-400",
    "text-emerald-600 dark:text-emerald-400",
    "text-orange-600 dark:text-orange-400",
    "text-pink-600 dark:text-pink-400",
  ];

  return (
    <div>
      <div className="mb-6">
        <a href="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
          &larr; Back to dashboard
        </a>
      </div>

      <h1 className="text-2xl font-bold mb-6">Compare Experiments</h1>

      {/* Experiment legend */}
      <div className="grid gap-3 mb-8" style={{ gridTemplateColumns: `repeat(${Math.min(experiments.length, 3)}, 1fr)` }}>
        {experiments.map((exp, i) => (
          <div
            key={exp.id}
            className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className={`font-bold text-lg ${colors[i % colors.length]}`}>
                [{i + 1}]
              </span>
              <span className="font-semibold text-sm">{displayName(exp)}</span>
            </div>
            {exp.label && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {exp.dataset} &middot; <span className="font-mono">{exp.agent_name}</span>
              </p>
            )}
            {!exp.label && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Agent: <span className="font-mono">{exp.agent_name}</span>
              </p>
            )}
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Trials: {exp.num_trials} &middot;{" "}
              {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
            </p>
          </div>
        ))}
      </div>

      {/* Metrics comparison table */}
      <section className="mb-10">
        <h2 className="text-lg font-semibold mb-4">Metrics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
              <tr>
                <th className="py-3 pr-4">Metric</th>
                {experiments.map((exp, i) => (
                  <th key={exp.id} className="py-3 pr-4 text-right">
                    <span className={colors[i % colors.length]}>[{i + 1}]</span>{" "}
                    {exp.label || exp.agent_name}
                  </th>
                ))}
                {experiments.length === 2 && (
                  <th className="py-3 pr-4 text-right">Delta</th>
                )}
              </tr>
            </thead>
            <tbody>
              {metricKeys.map((key) => {
                const isTime = isTimeMetric(key);
                const best = analysis?.metricAnalysis[key].bestIdx ?? null;

                return (
                  <tr
                    key={key}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3 pr-4 font-medium">{formatMetricName(key)}</td>
                    {experiments.map((exp, i) => {
                      const val = exp.metrics[key];
                      if (val === null) {
                        return (
                          <td key={i} className="py-3 pr-4 text-right text-gray-400">
                            --
                          </td>
                        );
                      }
                      const isBest = i === best && experiments.length > 1;
                      const formatted = isTime ? val.toFixed(2) : (val * 100).toFixed(2) + "%";
                      return (
                        <td
                          key={i}
                          className={`py-3 pr-4 text-right font-mono ${
                            isBest
                              ? "text-green-600 dark:text-green-400 font-bold"
                              : ""
                          }`}
                        >
                          {formatted}
                        </td>
                      );
                    })}
                    {experiments.length === 2 && (() => {
                      const v0 = experiments[0].metrics[key];
                      const v1 = experiments[1].metrics[key];
                      if (v0 === null || v1 === null) {
                        return <td className="py-3 pr-4 text-right text-gray-400">--</td>;
                      }
                      const delta = v1 - v0;
                      if (Math.abs(delta) < 1e-9) {
                        return (
                          <td className="py-3 pr-4 text-right text-gray-400 font-mono">
                            0
                          </td>
                        );
                      }
                      // For time: negative is better. For metrics: positive is better.
                      const isImprovement = isTime ? delta < 0 : delta > 0;
                      const sign = delta > 0 ? "+" : "";
                      const formatted = isTime
                        ? `${sign}${delta.toFixed(2)}s`
                        : `${sign}${(delta * 100).toFixed(2)}%`;
                      return (
                        <td
                          className={`py-3 pr-4 text-right font-mono ${
                            isImprovement
                              ? "text-green-600 dark:text-green-400"
                              : "text-red-600 dark:text-red-400"
                          }`}
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

      {/* Analysis section */}
      {analysis && (
        <section>
          <h2 className="text-lg font-semibold mb-4">Analysis</h2>

          {/* Win counts */}
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 mb-6">
            <h3 className="text-sm font-semibold mb-3 text-gray-500 uppercase">
              Metric Wins (excluding time)
            </h3>
            <div className="flex gap-6">
              {experiments.map((exp, i) => {
                const total = metricKeys.filter((k) => !isTimeMetric(k)).length;
                const winPct = total > 0 ? (analysis.wins[i] / total) * 100 : 0;
                return (
                  <div key={exp.id} className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`font-bold ${colors[i % colors.length]}`}>
                        [{i + 1}]
                      </span>
                      <span className="text-sm">{displayName(exp)}</span>
                    </div>
                    <div className="text-2xl font-bold">{analysis.wins[i]}</div>
                    <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-2 mt-2">
                      <div
                        className="bg-green-500 h-2 rounded-full transition-all"
                        style={{ width: `${winPct}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-400 mt-1">{winPct.toFixed(0)}%</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Speed comparison (if time metric exists) */}
          {experiments.length === 2 && (() => {
            const timeKey = metricKeys.find((k) => isTimeMetric(k));
            if (!timeKey) return null;
            const t0 = experiments[0].metrics[timeKey];
            const t1 = experiments[1].metrics[timeKey];
            if (t0 === null || t1 === null) return null;

            const faster = t0 < t1 ? 0 : 1;
            const slower = 1 - faster;
            const speedup = ((Math.max(t0, t1) - Math.min(t0, t1)) / Math.max(t0, t1)) * 100;

            if (speedup < 0.1) return null;

            return (
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 mb-6">
                <h3 className="text-sm font-semibold mb-3 text-gray-500 uppercase">
                  Speed
                </h3>
                <p className="text-sm">
                  <span className={`font-bold ${colors[faster % colors.length]}`}>
                    [{faster + 1}] {displayName(experiments[faster])}
                  </span>{" "}
                  is{" "}
                  <span className="font-bold text-green-600 dark:text-green-400">
                    {speedup.toFixed(1)}% faster
                  </span>{" "}
                  than{" "}
                  <span className={`font-bold ${colors[slower % colors.length]}`}>
                    [{slower + 1}] {displayName(experiments[slower])}
                  </span>{" "}
                  ({Math.min(t0, t1).toFixed(2)}s vs {Math.max(t0, t1).toFixed(2)}s avg query time)
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
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5">
                <h3 className="text-sm font-semibold mb-3 text-gray-500 uppercase">
                  Largest Differences
                </h3>
                <div className="space-y-2">
                  {diffs.slice(0, 5).map((d) => (
                    <div key={d.key} className="flex items-center gap-3">
                      <span className="text-sm font-medium w-40">
                        {formatMetricName(d.key)}
                      </span>
                      <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{
                            width: `${Math.min((d.spread! / 0.5) * 100, 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono text-gray-500 w-20 text-right">
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
