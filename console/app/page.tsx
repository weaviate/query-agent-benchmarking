"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
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

/* ── Score badge with brand colors ──────────────────────────────────────── */
function ScoreBadge({ value }: { value: number | null }) {
  if (value === null) return <span style={{ color: "var(--text-muted)" }}>--</span>;
  const pct = value * 100;
  const bg =
    pct >= 80
      ? "rgba(97,189,115,0.15)"
      : pct >= 50
        ? "rgba(249,241,93,0.2)"
        : "rgba(244,64,78,0.12)";
  const fg =
    pct >= 80
      ? "var(--color-green)"
      : pct >= 50
        ? "#b8a900"
        : "var(--color-coral)";
  return (
    <span
      className="brand-badge"
      style={{ background: bg, color: fg, fontWeight: 600 }}
    >
      {pct.toFixed(1)}%
    </span>
  );
}

/* ── Mode badge ─────────────────────────────────────────────────────────── */
function ModeBadge({ mode }: { mode: string }) {
  const styles: Record<string, { bg: string; fg: string }> = {
    search: { bg: "rgba(122,199,192,0.18)", fg: "var(--color-teal)" },
    ask: { bg: "rgba(165,144,221,0.18)", fg: "var(--color-lavender)" },
  };
  const s = styles[mode] || { bg: "rgba(136,150,171,0.15)", fg: "var(--text-muted)" };
  return (
    <span className="brand-badge" style={{ background: s.bg, color: s.fg }}>
      {mode}
    </span>
  );
}

/* ── Inline label editor ────────────────────────────────────────────────── */
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

  useEffect(() => setValue(initialLabel), [initialLabel]);
  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);

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
          if (e.key === "Escape") { setValue(initialLabel); setEditing(false); }
        }}
        placeholder="e.g. prod v1.2"
        className="text-xs px-2 py-1 rounded-md w-36 focus:outline-none"
        style={{
          border: "1px solid var(--color-green)",
          background: "var(--bg-card)",
          fontFamily: "var(--font-body)",
          color: "var(--text-primary)",
        }}
      />
    );
  }

  if (initialLabel) {
    return (
      <button
        onClick={() => setEditing(true)}
        className="brand-badge cursor-pointer"
        style={{ background: "rgba(97,189,115,0.12)", color: "var(--color-green)" }}
        title="Click to edit label"
      >
        {initialLabel}
      </button>
    );
  }

  return (
    <button
      onClick={() => setEditing(true)}
      className="text-xs cursor-pointer"
      style={{ color: "var(--text-muted)" }}
      title="Add a label"
    >
      + label
    </button>
  );
}

/* ── View toggle ────────────────────────────────────────────────────────── */
type ViewMode = "grouped" | "table";

export default function Dashboard() {
  const router = useRouter();
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleteTarget, setDeleteTarget] = useState<ExperimentSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [view, setView] = useState<ViewMode>("grouped");
  const [datasetFilter, setDatasetFilter] = useState<string | null>(null);

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
      setSelected((prev) => { const next = new Set(prev); next.delete(deleteTarget.id); return next; });
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
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const handleCompare = useCallback(() => {
    if (selected.size < 2) return;
    router.push(`/compare?ids=${[...selected].join(",")}`);
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

  /* ── Derived data ────────────────────────────────────────────────────── */
  const datasets = useMemo(() => {
    if (!experiments) return [];
    return [...new Set(experiments.map((e) => e.dataset))].sort();
  }, [experiments]);

  const filtered = useMemo(() => {
    if (!experiments) return [];
    if (!datasetFilter) return experiments;
    return experiments.filter((e) => e.dataset === datasetFilter);
  }, [experiments, datasetFilter]);

  const grouped = useMemo(() => {
    const map: Record<string, ExperimentSummary[]> = {};
    for (const exp of filtered) {
      (map[exp.dataset] ??= []).push(exp);
    }
    return Object.entries(map).sort(([a], [b]) => a.localeCompare(b));
  }, [filtered]);

  /* ── Loading / empty states ──────────────────────────────────────────── */
  if (experiments === null) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        <div className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3" style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }} />
        <p>Loading experiments...</p>
      </div>
    );
  }

  if (experiments.length === 0) {
    return (
      <div className="text-center py-20">
        <div className="mb-4">
          <svg width="48" height="48" viewBox="0 0 32 32" fill="none" className="mx-auto opacity-30">
            <path d="M16 1.07L29.86 9.07V24.93L16 30.93L2.14 24.93V9.07L16 1.07Z" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </div>
        <h2 className="text-xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
          No results yet
        </h2>
        <p style={{ color: "var(--text-secondary)" }}>
          Run a benchmark to generate results in{" "}
          <code
            className="px-2 py-0.5 rounded text-sm"
            style={{ background: "var(--bg-surface)", fontFamily: "var(--font-mono)" }}
          >
            console/results/
          </code>
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* ── Page header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-8 flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
            Experiments
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
            {experiments.length} experiment{experiments.length !== 1 ? "s" : ""} across {datasets.length} dataset{datasets.length !== 1 ? "s" : ""}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {selected.size >= 2 && (
            <button onClick={handleCompare} className="brand-btn-primary">
              Compare {selected.size} experiments
            </button>
          )}
          {selected.size === 1 && (
            <span className="text-sm" style={{ color: "var(--text-muted)" }}>
              Select one more to compare
            </span>
          )}
        </div>
      </div>

      {/* ── Toolbar: filters + view toggle ───────────────────────────────── */}
      <div className="flex items-center gap-3 mb-6 flex-wrap">
        <span className="eyebrow">Dataset</span>
        <div className="flex gap-1 flex-wrap">
          <button
            onClick={() => setDatasetFilter(null)}
            className="brand-btn-secondary"
            style={!datasetFilter ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
          >
            All
          </button>
          {datasets.map((ds) => (
            <button
              key={ds}
              onClick={() => setDatasetFilter(ds === datasetFilter ? null : ds)}
              className="brand-btn-secondary"
              style={datasetFilter === ds ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
            >
              {ds}
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setView("grouped")}
            className="brand-btn-secondary"
            style={view === "grouped" ? { background: "var(--bg-surface)", borderColor: "var(--color-green)" } : {}}
            title="Grouped view"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="1" y="1" width="6" height="6" rx="1"/><rect x="9" y="1" width="6" height="6" rx="1"/><rect x="1" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg>
          </button>
          <button
            onClick={() => setView("table")}
            className="brand-btn-secondary"
            style={view === "table" ? { background: "var(--bg-surface)", borderColor: "var(--color-green)" } : {}}
            title="Table view"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="1" y="2" width="14" height="2" rx="0.5"/><rect x="1" y="7" width="14" height="2" rx="0.5"/><rect x="1" y="12" width="14" height="2" rx="0.5"/></svg>
          </button>
        </div>
      </div>

      {/* ── Content area ─────────────────────────────────────────────────── */}
      {view === "grouped" ? (
        <GroupedView
          grouped={grouped}
          selected={selected}
          toggleSelect={toggleSelect}
          onLabelSaved={handleLabelSaved}
          onDelete={setDeleteTarget}
        />
      ) : (
        <TableView
          experiments={filtered}
          selected={selected}
          toggleSelect={toggleSelect}
          onLabelSaved={handleLabelSaved}
          onDelete={setDeleteTarget}
        />
      )}

      {/* ── Delete modal ─────────────────────────────────────────────────── */}
      {deleteTarget && (
        <div className="fixed inset-0 flex items-center justify-center z-50" style={{ background: "rgba(19,12,73,0.5)" }}>
          <div className="rounded-lg shadow-xl p-6 max-w-md mx-4" style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}>
            <h3 className="text-lg font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
              Delete experiment?
            </h3>
            <p className="text-sm mb-1" style={{ color: "var(--text-secondary)" }}>
              This will permanently delete all JSON files for:
            </p>
            <p className="text-sm font-semibold mb-4" style={{ fontFamily: "var(--font-mono)" }}>
              {deleteTarget.dataset} &mdash; {deleteTarget.agent_name}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="brand-btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="px-4 py-2 text-sm rounded-md font-semibold transition-colors disabled:opacity-50 cursor-pointer"
                style={{ background: "var(--color-coral)", color: "#fff" }}
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

/* ═══════════════════════════════════════════════════════════════════════════
   Grouped View — experiments organized by dataset
   ═══════════════════════════════════════════════════════════════════════════ */

function GroupedView({
  grouped,
  selected,
  toggleSelect,
  onLabelSaved,
  onDelete,
}: {
  grouped: [string, ExperimentSummary[]][];
  selected: Set<string>;
  toggleSelect: (id: string) => void;
  onLabelSaved: (id: string, label: string) => void;
  onDelete: (exp: ExperimentSummary) => void;
}) {
  return (
    <div className="space-y-10">
      {grouped.map(([dataset, exps]) => (
        <section key={dataset}>
          <div className="flex items-center gap-3 mb-4">
            <svg width="20" height="20" viewBox="0 0 32 32" fill="none" className="shrink-0">
              <path
                d="M16 2L28 9V23L16 30L4 23V9L16 2Z"
                stroke="var(--color-green)"
                strokeWidth="1.5"
                fill="rgba(97,189,115,0.08)"
              />
            </svg>
            <h2 className="text-lg font-bold" style={{ fontFamily: "var(--font-display)" }}>
              {dataset}
            </h2>
            <span className="eyebrow">{exps.length} run{exps.length !== 1 ? "s" : ""}</span>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {exps.map((exp) => (
              <ExperimentCard
                key={exp.id}
                exp={exp}
                isSelected={selected.has(exp.id)}
                toggleSelect={toggleSelect}
                onLabelSaved={onLabelSaved}
                onDelete={onDelete}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

/* ── Experiment Card ─────────────────────────────────────────────────────── */
function ExperimentCard({
  exp,
  isSelected,
  toggleSelect,
  onLabelSaved,
  onDelete,
}: {
  exp: ExperimentSummary;
  isSelected: boolean;
  toggleSelect: (id: string) => void;
  onLabelSaved: (id: string, label: string) => void;
  onDelete: (exp: ExperimentSummary) => void;
}) {
  return (
    <div
      className="brand-card p-5 relative"
      style={isSelected ? { borderColor: "var(--color-green)", boxShadow: "0 0 0 2px rgba(97,189,115,0.25)" } : {}}
    >
      {/* Selection checkbox */}
      <div className="absolute top-4 right-4">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => toggleSelect(exp.id)}
          className="w-4 h-4 rounded cursor-pointer accent-[#61BD73]"
        />
      </div>

      {/* Header row: mode badge + label */}
      <div className="flex items-center gap-2 mb-3">
        <ModeBadge mode={exp.mode} />
        <InlineLabel experimentId={exp.id} initialLabel={exp.label} onSaved={onLabelSaved} />
      </div>

      {/* Agent name */}
      <p
        className="text-xs mb-3 truncate"
        style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}
        title={exp.agent_name}
      >
        {exp.agent_name}
      </p>

      {/* Key metric — hero element */}
      <div className="flex items-end justify-between mb-4">
        <div>
          <div className="eyebrow mb-1">{exp.keyMetric.name}</div>
          <div className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
            {exp.keyMetric.value !== null
              ? `${(exp.keyMetric.value * 100).toFixed(1)}%`
              : "--"}
          </div>
        </div>
        <ScoreBadge value={exp.keyMetric.value} />
      </div>

      {/* Footer: trials, timestamp, actions */}
      <div className="flex items-center justify-between pt-3" style={{ borderTop: "1px solid var(--border-subtle)" }}>
        <div className="flex items-center gap-3">
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
            {exp.num_trials} trial{exp.num_trials !== 1 ? "s" : ""}
          </span>
          <span className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {exp.timestamp ? new Date(exp.timestamp).toLocaleDateString() : "--"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <a
            href={`/experiments/${exp.id}`}
            className="text-xs font-semibold transition-colors"
            style={{ color: "var(--color-green)" }}
          >
            View
          </a>
          <button
            onClick={() => onDelete(exp)}
            className="text-xs transition-colors cursor-pointer"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--color-coral)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Table View — classic flat table for dense scanning
   ═══════════════════════════════════════════════════════════════════════════ */

function TableView({
  experiments,
  selected,
  toggleSelect,
  onLabelSaved,
  onDelete,
}: {
  experiments: ExperimentSummary[];
  selected: Set<string>;
  toggleSelect: (id: string) => void;
  onLabelSaved: (id: string, label: string) => void;
  onDelete: (exp: ExperimentSummary) => void;
}) {
  return (
    <div className="overflow-x-auto brand-card">
      <table className="w-full text-sm text-left brand-table">
        <thead>
          <tr>
            <th className="w-8"></th>
            <th>Label</th>
            <th>Dataset</th>
            <th>Agent</th>
            <th>Mode</th>
            <th>Key Metric</th>
            <th>Trials</th>
            <th>Timestamp</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp) => {
            const isSelected = selected.has(exp.id);
            return (
              <tr
                key={exp.id}
                style={isSelected ? { background: "rgba(97,189,115,0.06)" } : {}}
              >
                <td>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleSelect(exp.id)}
                    className="w-4 h-4 rounded cursor-pointer accent-[#61BD73]"
                  />
                </td>
                <td>
                  <InlineLabel experimentId={exp.id} initialLabel={exp.label} onSaved={onLabelSaved} />
                </td>
                <td className="font-semibold">{exp.dataset}</td>
                <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>{exp.agent_name}</td>
                <td><ModeBadge mode={exp.mode} /></td>
                <td>
                  <div className="flex items-center gap-2">
                    <ScoreBadge value={exp.keyMetric.value} />
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>{exp.keyMetric.name}</span>
                  </div>
                </td>
                <td>{exp.num_trials}</td>
                <td className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
                </td>
                <td>
                  <div className="flex items-center gap-3">
                    <a
                      href={`/experiments/${exp.id}`}
                      className="text-xs font-semibold"
                      style={{ color: "var(--color-green)" }}
                    >
                      View
                    </a>
                    <button
                      onClick={() => onDelete(exp)}
                      className="text-xs cursor-pointer"
                      style={{ color: "var(--color-coral)" }}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
