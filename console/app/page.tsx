"use client";

import { useEffect, useState, useCallback } from "react";

interface ExperimentSummary {
  id: string;
  dataset: string;
  agent_name: string;
  mode: "search" | "ask" | "unknown";
  num_trials: number;
  timestamp: string;
  keyMetric: { name: string; value: number | null };
}

const POLL_INTERVAL_MS = 5_000;

function ScoreBadge({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-400">--</span>;
  const pct = value * 100;
  const color =
    pct >= 80
      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
      : pct >= 50
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-sm font-medium ${color}`}>
      {pct.toFixed(1)}%
    </span>
  );
}

function ModeBadge({ mode }: { mode: string }) {
  const color =
    mode === "search"
      ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
      : mode === "ask"
        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
        : "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {mode}
    </span>
  );
}

export default function Dashboard() {
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ExperimentSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchExperiments = useCallback(() => {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((data) => setExperiments(data))
      .catch(() => {});
  }, []);

  const handleDelete = useCallback(async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await fetch(`/api/experiments/${deleteTarget.id}`, { method: "DELETE" });
      setExperiments((prev) => prev?.filter((e) => e.id !== deleteTarget.id) ?? null);
    } catch {
      // ignore
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }, [deleteTarget]);

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchExperiments]);

  if (experiments === null) {
    return <div className="py-20 text-center text-gray-500">Loading...</div>;
  }

  if (experiments.length === 0) {
    return (
      <div className="text-center py-20">
        <h2 className="text-xl font-semibold mb-2">No results found</h2>
        <p className="text-gray-500 dark:text-gray-400">
          Run a benchmark to generate results in{" "}
          <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm">
            console/results/
          </code>
        </p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Experiments</h1>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
            <tr>
              <th className="py-3 pr-4">Dataset</th>
              <th className="py-3 pr-4">Agent</th>
              <th className="py-3 pr-4">Mode</th>
              <th className="py-3 pr-4">Key Metric</th>
              <th className="py-3 pr-4">Trials</th>
              <th className="py-3 pr-4">Timestamp</th>
              <th className="py-3"></th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr
                key={exp.id}
                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900"
              >
                <td className="py-3 pr-4 font-medium">{exp.dataset}</td>
                <td className="py-3 pr-4 font-mono text-xs">{exp.agent_name}</td>
                <td className="py-3 pr-4">
                  <ModeBadge mode={exp.mode} />
                </td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <ScoreBadge value={exp.keyMetric.value} />
                    <span className="text-xs text-gray-400">{exp.keyMetric.name}</span>
                  </div>
                </td>
                <td className="py-3 pr-4">{exp.num_trials}</td>
                <td className="py-3 pr-4 text-xs text-gray-500">
                  {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
                </td>
                <td className="py-3 flex items-center gap-3">
                  <a
                    href={`/experiments/${exp.id}`}
                    className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
                  >
                    View details
                  </a>
                  <button
                    onClick={() => setDeleteTarget(exp)}
                    className="text-xs text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold mb-2">Delete experiment?</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              This will permanently delete all JSON files for:
            </p>
            <p className="text-sm font-medium mb-4">
              {deleteTarget.dataset} &mdash; {deleteTarget.agent_name}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="px-4 py-2 text-sm rounded bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="px-4 py-2 text-sm rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
              >
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
