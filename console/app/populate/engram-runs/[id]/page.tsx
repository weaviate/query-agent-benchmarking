"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams, useSearchParams } from "next/navigation";

interface EngramRunEntry {
  run_id: string;
  tenant_id: string;
  session_id: string;
  session_date: string;
  submitted_at: number;
  status?: string;
  created?: number;
  updated?: number;
  deleted?: number;
  run_duration_seconds?: number;
}

interface EngramTenantStats {
  tenant_id: string;
  num_sessions: number;
  elapsed_seconds: number;
  total_created: number;
  total_updated: number;
  total_deleted: number;
}

interface EngramManifest {
  type: "engram_ingest";
  timestamp: string;
  dataset: string;
  submit_elapsed_seconds: number;
  total_sessions: number;
  total_tenants: number;
  tenant_session_counts: Record<string, number>;
  runs: EngramRunEntry[];
  stats?: EngramTenantStats[];
}

interface MemoryEntry {
  memory_id: string;
  operation: "created" | "updated" | "deleted";
  content: string | null;
  created_at?: string;
  note?: string;
}

interface RunDetail {
  session_text: string | null;
  run_status: string;
  memories: MemoryEntry[];
}

const OP_COLORS: Record<string, { bg: string; fg: string }> = {
  created: { bg: "rgba(97,189,115,0.15)", fg: "var(--color-green)" },
  updated: { bg: "rgba(100,149,237,0.15)", fg: "cornflowerblue" },
  deleted: { bg: "rgba(244,64,78,0.12)", fg: "var(--color-coral)" },
};

function RunDetailPanel({
  run,
  dataset,
}: {
  run: EngramRunEntry;
  dataset: string;
}) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const params = new URLSearchParams({
      run_id: run.run_id,
      dataset_name: dataset,
      tenant_id: run.tenant_id,
      session_id: run.session_id,
    });
    fetch(`/api/backend/engram-run-detail?${params}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "error") throw new Error(data.error);
        setDetail(data);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [run, dataset]);

  if (loading) {
    return (
      <div className="py-4 text-center">
        <div
          className="inline-block w-4 h-4 border-2 border-t-transparent rounded-full animate-spin"
          style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
        />
        <span className="ml-2 text-xs" style={{ color: "var(--text-muted)" }}>Loading run detail...</span>
      </div>
    );
  }

  if (error) {
    return (
      <p className="py-3 text-xs" style={{ color: "var(--color-coral)" }}>
        Failed to load: {error}
      </p>
    );
  }

  if (!detail) return null;

  return (
    <div className="space-y-4">
      {/* Session input */}
      <div>
        <p className="eyebrow mb-2">Session Input</p>
        {detail.session_text ? (
          <div
            className="rounded-md p-3 text-xs overflow-auto max-h-64"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-default)",
              color: "var(--text-primary)",
              fontFamily: "var(--font-mono)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {detail.session_text}
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            Session text not found in dataset.
          </p>
        )}
      </div>

      {/* Memories */}
      <div>
        <p className="eyebrow mb-2">
          Memory Operations ({detail.memories.length})
        </p>
        {detail.memories.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>No memory operations for this run.</p>
        ) : (
          <div className="space-y-2">
            {detail.memories.map((m) => {
              const colors = OP_COLORS[m.operation] || OP_COLORS.created;
              return (
                <div
                  key={m.memory_id}
                  className="rounded-md p-3"
                  style={{ background: colors.bg, border: `1px solid ${colors.fg}22` }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                      style={{ background: colors.bg, color: colors.fg, border: `1px solid ${colors.fg}44` }}
                    >
                      {m.operation}
                    </span>
                    <span className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {m.memory_id.slice(0, 12)}...
                    </span>
                  </div>
                  {m.content ? (
                    <p className="text-xs" style={{ color: "var(--text-primary)", lineHeight: 1.5 }}>
                      {m.content}
                    </p>
                  ) : (
                    <p className="text-xs italic" style={{ color: "var(--text-muted)" }}>
                      {m.note || "Memory content unavailable"}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default function EngramRunDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const id = params.id as string;

  const [manifest, setManifest] = useState<EngramManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tenantFilter, setTenantFilter] = useState<string>("all");
  const [expandedRunId, setExpandedRunId] = useState<string | null>(
    searchParams.get("expand") || null
  );

  useEffect(() => {
    fetch(`/api/engram-runs/${id}`)
      .then((r) => {
        if (!r.ok) throw new Error("Not found");
        return r.json();
      })
      .then((data) => setManifest(data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  const toggleRun = useCallback((runId: string) => {
    setExpandedRunId((prev) => (prev === runId ? null : runId));
  }, []);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-10 text-center">
        <div
          className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin"
          style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
        />
      </div>
    );
  }

  if (error || !manifest) {
    return (
      <div className="max-w-4xl mx-auto py-10">
        <p className="text-sm" style={{ color: "var(--color-coral)" }}>
          {error || "Manifest not found"}
        </p>
        <a href="/populate/engram-runs" className="brand-btn-secondary mt-4 inline-block" style={{ fontSize: "0.75rem" }}>
          Back to History
        </a>
      </div>
    );
  }

  const tenantIds = [...new Set(manifest.runs.map((r) => r.tenant_id))].sort();
  const filteredRuns = tenantFilter === "all"
    ? manifest.runs
    : manifest.runs.filter((r) => r.tenant_id === tenantFilter);

  let totalCreated = 0, totalUpdated = 0, totalDeleted = 0;
  if (manifest.stats) {
    for (const s of manifest.stats) {
      totalCreated += s.total_created;
      totalUpdated += s.total_updated;
      totalDeleted += s.total_deleted;
    }
  } else {
    for (const r of manifest.runs) {
      totalCreated += r.created ?? 0;
      totalUpdated += r.updated ?? 0;
      totalDeleted += r.deleted ?? 0;
    }
  }

  const formatTimestamp = (ts: string) => {
    try { return new Date(ts).toLocaleString(); } catch { return ts; }
  };

  return (
    <div className="max-w-4xl mx-auto py-10">
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>Home</a>
        <span>/</span>
        <a href="/populate" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>Populate</a>
        <span>/</span>
        <a href="/populate/engram-runs" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>Engram Runs</a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Detail</span>
      </nav>

      <h1 className="text-2xl font-bold mb-1" style={{ fontFamily: "var(--font-display)" }}>
        Engram Ingestion Detail
      </h1>
      <p className="text-xs mb-6" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        {manifest.dataset} &mdash; {formatTimestamp(manifest.timestamp)}
      </p>

      {/* Summary cards */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <div className="brand-card p-3 text-center">
          <p className="eyebrow mb-1">Sessions</p>
          <p className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>{manifest.total_sessions}</p>
        </div>
        <div className="brand-card p-3 text-center">
          <p className="eyebrow mb-1">Tenants</p>
          <p className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>{manifest.total_tenants}</p>
        </div>
        <div className="brand-card p-3 text-center">
          <p className="eyebrow mb-1">Created</p>
          <p className="text-xl font-bold" style={{ color: "var(--color-green)" }}>{totalCreated}</p>
        </div>
        <div className="brand-card p-3 text-center">
          <p className="eyebrow mb-1">Updated</p>
          <p className="text-xl font-bold" style={{ color: "cornflowerblue" }}>{totalUpdated}</p>
        </div>
        <div className="brand-card p-3 text-center">
          <p className="eyebrow mb-1">Deleted</p>
          <p className="text-xl font-bold" style={{ color: "var(--color-coral)" }}>{totalDeleted}</p>
        </div>
      </div>

      {/* Per-tenant stats */}
      {manifest.stats && manifest.stats.length > 0 && (
        <div className="brand-card p-4 mb-6">
          <p className="eyebrow mb-3">Per-Tenant Stats</p>
          <div className="rounded-md overflow-hidden" style={{ border: "1px solid var(--border-default)" }}>
            <table className="w-full text-xs" style={{ fontFamily: "var(--font-mono)" }}>
              <thead>
                <tr style={{ background: "var(--bg-surface)" }}>
                  <th className="text-left px-3 py-2" style={{ color: "var(--text-muted)" }}>Tenant</th>
                  <th className="text-right px-3 py-2" style={{ color: "var(--text-muted)" }}>Sessions</th>
                  <th className="text-right px-3 py-2" style={{ color: "var(--text-muted)" }}>Created</th>
                  <th className="text-right px-3 py-2" style={{ color: "var(--text-muted)" }}>Updated</th>
                  <th className="text-right px-3 py-2" style={{ color: "var(--text-muted)" }}>Deleted</th>
                  <th className="text-right px-3 py-2" style={{ color: "var(--text-muted)" }}>Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {manifest.stats.map((s) => (
                  <tr key={s.tenant_id} style={{ borderTop: "1px solid var(--border-default)" }}>
                    <td className="px-3 py-1.5" style={{ color: "var(--text-primary)" }}>{s.tenant_id}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: "var(--text-primary)" }}>{s.num_sessions}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: "var(--color-green)" }}>{s.total_created}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: "cornflowerblue" }}>{s.total_updated}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: "var(--color-coral)" }}>{s.total_deleted}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: "var(--text-muted)" }}>{s.elapsed_seconds.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Run records */}
      <div className="brand-card p-4">
        <div className="flex items-center justify-between mb-1">
          <p className="eyebrow">Run Records ({filteredRuns.length})</p>
          {tenantIds.length > 1 && (
            <select
              value={tenantFilter}
              onChange={(e) => setTenantFilter(e.target.value)}
              className="rounded-md px-2 py-1 text-xs"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
            >
              <option value="all">All tenants ({manifest.runs.length})</option>
              {tenantIds.map((tid) => (
                <option key={tid} value={tid}>
                  {tid} ({manifest.runs.filter((r) => r.tenant_id === tid).length})
                </option>
              ))}
            </select>
          )}
        </div>
        <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
          Click a row to view the session input and memory operations.
        </p>

        <div className="space-y-0">
          {filteredRuns.map((r) => {
            const isExpanded = expandedRunId === r.run_id;
            const ops = (r.created ?? 0) + (r.updated ?? 0) + (r.deleted ?? 0);
            return (
              <div key={r.run_id}>
                <button
                  type="button"
                  onClick={() => toggleRun(r.run_id)}
                  className="w-full text-left px-3 py-2 flex items-center gap-3 transition-colors"
                  style={{
                    background: isExpanded ? "var(--bg-surface)" : "transparent",
                    borderTop: "1px solid var(--border-default)",
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.75rem",
                  }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      transform: isExpanded ? "rotate(90deg)" : "none",
                      transition: "transform 0.15s",
                      color: "var(--text-muted)",
                      fontSize: "0.6rem",
                    }}
                  >
                    &#9654;
                  </span>
                  <span style={{ color: "var(--text-primary)", minWidth: "80px" }}>{r.tenant_id}</span>
                  <span style={{ color: "var(--text-primary)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.session_id}>
                    {r.session_id}
                  </span>
                  <span style={{ color: "var(--text-muted)", minWidth: "70px", textAlign: "right" }}>
                    {ops} ops
                  </span>
                  <span style={{ minWidth: "55px", textAlign: "right" }}>
                    {r.status && (
                      <span
                        className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                        style={{
                          background: r.status === "completed" ? "rgba(97,189,115,0.15)" : "rgba(244,64,78,0.15)",
                          color: r.status === "completed" ? "var(--color-green)" : "var(--color-coral)",
                        }}
                      >
                        {r.status}
                      </span>
                    )}
                  </span>
                </button>

                {isExpanded && (
                  <div
                    className="px-4 py-4"
                    style={{
                      background: "var(--bg-surface)",
                      borderTop: "1px solid var(--border-default)",
                    }}
                  >
                    <RunDetailPanel run={r} dataset={manifest.dataset} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
