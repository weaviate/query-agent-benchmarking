"use client";

import { useState, useEffect } from "react";

interface EngramManifestSummary {
  id: string;
  filename: string;
  timestamp: string;
  dataset: string;
  total_sessions: number;
  total_tenants: number;
  submit_elapsed_seconds: number;
  total_created: number;
  total_updated: number;
  total_deleted: number;
}

export default function EngramRunsPage() {
  const [manifests, setManifests] = useState<EngramManifestSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/engram-runs")
      .then((r) => r.json())
      .then((data) => setManifests(data))
      .catch(() => setManifests([]))
      .finally(() => setLoading(false));
  }, []);

  const formatTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-10">
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Home
        </a>
        <span>/</span>
        <a href="/populate" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Populate
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Engram Run History</span>
      </nav>

      <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
        Engram Run History
      </h1>
      <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
        Browse past Engram ingestion runs. Each entry maps run IDs back to the sessions that produced them.
      </p>

      {loading && (
        <div className="text-center py-12">
          <div
            className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
          />
        </div>
      )}

      {!loading && manifests.length === 0 && (
        <div
          className="rounded-lg p-8 text-center"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
        >
          <p className="text-sm mb-2" style={{ color: "var(--text-muted)" }}>
            No Engram ingestion runs found.
          </p>
          <a href="/populate" className="brand-btn-primary" style={{ fontSize: "0.75rem" }}>
            Run a Population
          </a>
        </div>
      )}

      {!loading && manifests.length > 0 && (
        <div className="space-y-3">
          {manifests.map((m) => (
            <a
              key={m.id}
              href={`/populate/engram-runs/${m.id}`}
              className="brand-card p-4 block transition-shadow hover:shadow-md"
              style={{ textDecoration: "none" }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block rounded px-2 py-0.5 text-xs font-medium"
                    style={{ background: "rgba(1,245,122,0.12)", color: "var(--color-green)" }}
                  >
                    {m.dataset}
                  </span>
                  <span className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {formatTimestamp(m.timestamp)}
                  </span>
                </div>
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {m.submit_elapsed_seconds.toFixed(0)}s
                </span>
              </div>

              <div className="flex gap-6 text-xs" style={{ fontFamily: "var(--font-mono)" }}>
                <span>
                  <span style={{ color: "var(--text-muted)" }}>sessions: </span>
                  <span style={{ color: "var(--text-primary)" }}>{m.total_sessions}</span>
                </span>
                <span>
                  <span style={{ color: "var(--text-muted)" }}>tenants: </span>
                  <span style={{ color: "var(--text-primary)" }}>{m.total_tenants}</span>
                </span>
                <span>
                  <span style={{ color: "var(--text-muted)" }}>created: </span>
                  <span style={{ color: "var(--color-green)" }}>{m.total_created}</span>
                </span>
                {m.total_updated > 0 && (
                  <span>
                    <span style={{ color: "var(--text-muted)" }}>updated: </span>
                    <span style={{ color: "var(--text-primary)" }}>{m.total_updated}</span>
                  </span>
                )}
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
