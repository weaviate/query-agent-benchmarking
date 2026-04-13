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
        if (r.status === 404) {
          setNotFound(true);
          return null;
        }
        return r.json();
      })
      .then((data) => {
        if (data) setExperiment(data);
      })
      .catch(() => {});
  }, [id]);

  useEffect(() => {
    fetchExperiment();
    const interval = setInterval(fetchExperiment, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchExperiment]);

  if (notFound) {
    return <div className="py-20 text-center text-gray-500">Experiment not found.</div>;
  }

  if (!experiment) {
    return <div className="py-20 text-center text-gray-500">Loading...</div>;
  }

  return (
    <div>
      <div className="mb-6">
        <a href="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
          &larr; Back to dashboard
        </a>
      </div>

      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">{experiment.dataset}</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Agent: <span className="font-mono">{experiment.agent_name}</span> &middot;
          Mode: <span className="capitalize">{experiment.mode}</span> &middot;
          {experiment.num_trials} trial{experiment.num_trials !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Aggregated metrics */}
      {experiment.metricEntries.length > 0 && (
        <section className="mb-10">
          <h2 className="text-lg font-semibold mb-4">Aggregated Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {experiment.metricEntries.map(({ key, value }) => (
              <div
                key={key}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
              >
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="text-lg font-semibold">{value}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-trial table */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Trials</h2>
        {experiment.trials.length === 0 ? (
          <p className="text-gray-500">No trial data available.</p>
        ) : (
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
              <tr>
                <th className="py-3 pr-4">Trial</th>
                <th className="py-3 pr-4">Queries</th>
                <th className="py-3 pr-4">Avg Query Time</th>
                <th className="py-3 pr-4">Key Score</th>
                <th className="py-3"></th>
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
                  <tr
                    key={trial.trialNumber}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3 pr-4 font-medium">Trial {trial.trialNumber}</td>
                    <td className="py-3 pr-4">{totalQueries}</td>
                    <td className="py-3 pr-4">{avgTime}</td>
                    <td className="py-3 pr-4">{keyScore}</td>
                    <td className="py-3">
                      {trial.hasResults ? (
                        <a
                          href={`/experiments/${experiment.id}/trial/${trial.trialNumber}`}
                          className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
                        >
                          View queries
                        </a>
                      ) : (
                        <span className="text-xs text-gray-400">No query data</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
