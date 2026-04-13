"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";

interface ExperimentSummary {
  id: string;
  label: string;
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

function InlineLabel({
  experimentId,
  initialLabel,
  onSaved,
}: {
  experimentId: string;
  initialLabel: string;
  onSaved: (id: string, label: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(initialLabel);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setValue(initialLabel);
  }, [initialLabel]);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  const save = useCallback(() => {
    setEditing(false);
    const trimmed = value.trim();
    if (trimmed === initialLabel) return;
    fetch(`/api/experiments/${experimentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: trimmed }),
    }).catch(() => {});
    onSaved(experimentId, trimmed);
  }, [experimentId, value, initialLabel, onSaved]);

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") {
            setValue(initialLabel);
            setEditing(false);
          }
        }}
        placeholder="e.g. prod v1.2"
        className="text-xs px-1.5 py-0.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 w-32 focus:outline-none focus:ring-1 focus:ring-blue-500"
      />
    );
  }

  if (initialLabel) {
    return (
      <button
        onClick={() => setEditing(true)}
        className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer"
        title="Click to edit label"
      >
        {initialLabel}
      </button>
    );
  }

  return (
    <button
      onClick={() => setEditing(true)}
      className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 cursor-pointer"
      title="Add a label"
    >
      + label
    </button>
  );
}

export default function Dashboard() {
  const router = useRouter();
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
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
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(deleteTarget.id);
        return next;
      });
    } catch {
      // ignore
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }, [deleteTarget]);

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const handleCompare = useCallback(() => {
    if (selected.size < 2) return;
    const ids = [...selected].join(",");
    router.push(`/compare?ids=${ids}`);
  }, [selected, router]);

  const handleLabelSaved = useCallback((id: string, label: string) => {
    setExperiments((prev) =>
      prev?.map((e) => (e.id === id ? { ...e, label } : e)) ?? null
    );
  }, []);

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
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Experiments</h1>
        {selected.size >= 2 && (
          <button
            onClick={handleCompare}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
          >
            Compare {selected.size} experiments
          </button>
        )}
        {selected.size === 1 && (
          <span className="text-sm text-gray-400">
            Select at least one more to compare
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
            <tr>
              <th className="py-3 pr-2 w-8"></th>
              <th className="py-3 pr-4">Label</th>
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
            {experiments.map((exp) => {
              const isSelected = selected.has(exp.id);
              return (
                <tr
                  key={exp.id}
                  className={`border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900 ${
                    isSelected ? "bg-blue-50 dark:bg-blue-950/30" : ""
                  }`}
                >
                  <td className="py-3 pr-2">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSelect(exp.id)}
                      className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                    />
                  </td>
                  <td className="py-3 pr-4">
                    <InlineLabel
                      experimentId={exp.id}
                      initialLabel={exp.label}
                      onSaved={handleLabelSaved}
                    />
                  </td>
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
              );
            })}
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
