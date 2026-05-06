"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { SEARCH_DATASETS, ASK_DATASETS } from "@/lib/datasets";

const POPULATE_DATASETS = [...new Set([...SEARCH_DATASETS, ...ASK_DATASETS])].sort();

const STORAGE_KEY = "populate-form";
const JOB_KEY = "populate-active-job";

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

interface ProgressState {
  phase: "loading" | "submit" | "poll" | "poll_wait";
  message?: string;
  submitted?: number;
  completed?: number;
  total?: number;
  skipped?: number;
  tenant_id?: string;
  session_id?: string;
  run_id?: string;
  run_status?: string;
  poll_checks?: number;
  memories_created?: number;
  memories_updated?: number;
  memories_deleted?: number;
}

interface FormState {
  datasetName: string;
  databaseTarget: "weaviate" | "engram";
  tag: string;
  recreate: boolean;
  textEmbeddingModel: string;
  imageEmbeddingModel: string;
  useMuvera: boolean;
  ksim: number;
  dprojections: number;
  repetitions: number;
  ef: number;
  showSubset: boolean;
  subsetStart: number;
  subsetEnd: number;
  tenantId: string;
}

interface ActiveJob {
  jobId: string;
  startedAt: number; // Date.now()
}

// ---------------------------------------------------------------------------
// sessionStorage helpers
// ---------------------------------------------------------------------------

function loadFormState(): Partial<FormState> {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveFormState(state: FormState) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* ignore */ }
}

function loadActiveJob(): ActiveJob | null {
  try {
    const raw = sessionStorage.getItem(JOB_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveActiveJob(job: ActiveJob) {
  try {
    sessionStorage.setItem(JOB_KEY, JSON.stringify(job));
  } catch { /* ignore */ }
}

function clearActiveJob() {
  try {
    sessionStorage.removeItem(JOB_KEY);
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Progress event parser (shared between stream + poll)
// ---------------------------------------------------------------------------

function parseProgressEvent(event: Record<string, unknown>): {
  type: "loading" | "progress" | "complete" | "error" | "started" | "unknown";
  progress?: ProgressState;
  manifest?: EngramManifest;
  jobId?: string;
  errorMessage?: string;
} {
  if (event.event === "started") {
    return { type: "started", jobId: event.job_id as string };
  }
  if (event.event === "loading") {
    return { type: "loading", progress: { phase: "loading", message: event.message as string } };
  }
  // Handle both wrapped (event.event === "progress") and raw (no event key) progress dicts
  const isProgress = event.event === "progress" || (!event.event && typeof event.phase === "string");
  if (isProgress) {
    const phase = event.phase as string;
    if (phase === "submit") {
      return {
        type: "progress",
        progress: {
          phase: "submit",
          submitted: event.submitted as number,
          total: event.total as number,
          skipped: event.skipped as number,
          tenant_id: event.tenant_id as string,
          session_id: event.session_id as string,
        },
      };
    }
    if (phase === "poll") {
      return {
        type: "progress",
        progress: {
          phase: "poll",
          completed: event.completed as number,
          total: event.total as number,
          tenant_id: event.tenant_id as string,
          memories_created: event.memories_created as number,
          memories_updated: event.memories_updated as number,
          memories_deleted: event.memories_deleted as number,
        },
      };
    }
    if (phase === "poll_wait") {
      return {
        type: "progress",
        progress: {
          phase: "poll_wait",
          completed: event.completed as number,
          total: event.total as number,
          tenant_id: event.tenant_id as string,
          run_id: event.run_id as string,
          run_status: event.run_status as string,
          poll_checks: event.poll_checks as number,
          memories_created: event.memories_created as number,
          memories_updated: event.memories_updated as number,
          memories_deleted: event.memories_deleted as number,
        },
      };
    }
  }
  if (event.event === "complete") {
    return { type: "complete", manifest: event.manifest as EngramManifest };
  }
  if (event.event === "error") {
    return { type: "error", errorMessage: event.message as string };
  }
  return { type: "unknown" };
}


export default function PopulatePage() {
  // Restore form state from sessionStorage on initial render
  const [initialized, setInitialized] = useState(false);
  const saved = useRef<Partial<FormState>>({});

  // Core fields
  const [datasetName, setDatasetName] = useState("longmemeval-s");
  const [databaseTarget, setDatabaseTarget] = useState<"weaviate" | "engram">("engram");
  const [tag, setTag] = useState("Default");
  const [recreate, setRecreate] = useState(true);

  // Embedding models
  const [textEmbeddingModel, setTextEmbeddingModel] = useState("weaviate/Snowflake/snowflake-arctic-embed-l-v2.0");
  const [imageEmbeddingModel, setImageEmbeddingModel] = useState("weaviate/ModernVBERT/colmodernvbert");

  // MUVERA
  const [showMuvera, setShowMuvera] = useState(false);
  const [useMuvera, setUseMuvera] = useState(false);
  const [ksim, setKsim] = useState(4);
  const [dprojections, setDprojections] = useState(16);
  const [repetitions, setRepetitions] = useState(10);
  const [ef, setEf] = useState(500);

  // LongMemEval subset
  const [showSubset, setShowSubset] = useState(false);
  const [subsetStart, setSubsetStart] = useState(0);
  const [subsetEnd, setSubsetEnd] = useState(5);
  const [tenantId, setTenantId] = useState("");

  // Status
  const [status, setStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Live progress for Engram streaming
  const [progress, setProgress] = useState<ProgressState | null>(null);

  // Engram manifest from last successful population
  const [engramManifest, setEngramManifest] = useState<EngramManifest | null>(null);
  const [showRuns, setShowRuns] = useState(false);
  const [tenantFilter, setTenantFilter] = useState<string>("all");

  // Track whether we're polling (reconnected to existing job)
  const pollRef = useRef(false);
  const [isPolling, setIsPolling] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);

  // -------------------------------------------------------------------------
  // Persist form state to sessionStorage on every change
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!initialized) return;
    saveFormState({
      datasetName, databaseTarget, tag, recreate,
      textEmbeddingModel, imageEmbeddingModel,
      useMuvera, ksim, dprojections, repetitions, ef,
      showSubset, subsetStart, subsetEnd, tenantId,
    });
  }, [
    initialized, datasetName, databaseTarget, tag, recreate,
    textEmbeddingModel, imageEmbeddingModel,
    useMuvera, ksim, dprojections, repetitions, ef,
    showSubset, subsetStart, subsetEnd, tenantId,
  ]);

  // -------------------------------------------------------------------------
  // Poll for job status (used when reconnecting after navigation)
  // -------------------------------------------------------------------------
  const pollJobStatus = useCallback(async (jobId: string, startedAt: number) => {
    pollRef.current = true;
    setIsPolling(true);
    setStatus("running");
    setElapsed(Math.round((Date.now() - startedAt) / 1000));

    // Start elapsed timer from original start time
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setElapsed(Math.round((Date.now() - startedAt) / 1000));
    }, 1000);

    try {
      while (pollRef.current) {
        setLastCheckedAt(Date.now());
        const resp = await fetch(`/api/backend/populate-db-job/${jobId}`);
        const data = await resp.json();

        if (data.status === "not_found") {
          clearActiveJob();
          setStatus("error");
          setErrorMsg("Job not found — the server may have restarted.");
          setProgress(null);
          setIsPolling(false);
          if (timerRef.current) clearInterval(timerRef.current);
          return;
        }

        // Update progress from latest event
        if (data.progress) {
          const parsed = parseProgressEvent(data.progress);
          if (parsed.progress) setProgress(parsed.progress);
        }

        if (data.status === "complete" && data.manifest) {
          setEngramManifest(data.manifest);
          setProgress(null);
          setStatus("success");
          setIsPolling(false);
          clearActiveJob();
          if (timerRef.current) clearInterval(timerRef.current);
          return;
        }

        if (data.status === "error") {
          setStatus("error");
          setErrorMsg(data.error || "An error occurred during ingestion.");
          setProgress(null);
          setIsPolling(false);
          clearActiveJob();
          if (timerRef.current) clearInterval(timerRef.current);
          return;
        }

        // Still running — wait and poll again
        await new Promise((r) => setTimeout(r, 30_000));
      }
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Failed to reconnect to job");
      setProgress(null);
      setIsPolling(false);
      clearActiveJob();
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, []);

  // -------------------------------------------------------------------------
  // On mount: restore form state + reconnect to active job
  // -------------------------------------------------------------------------
  useEffect(() => {
    saved.current = loadFormState();
    const s = saved.current;
    if (s.datasetName !== undefined) setDatasetName(s.datasetName);
    if (s.databaseTarget !== undefined) setDatabaseTarget(s.databaseTarget);
    if (s.tag !== undefined) setTag(s.tag);
    if (s.recreate !== undefined) setRecreate(s.recreate);
    if (s.textEmbeddingModel !== undefined) setTextEmbeddingModel(s.textEmbeddingModel);
    if (s.imageEmbeddingModel !== undefined) setImageEmbeddingModel(s.imageEmbeddingModel);
    if (s.useMuvera !== undefined) setUseMuvera(s.useMuvera);
    if (s.ksim !== undefined) setKsim(s.ksim);
    if (s.dprojections !== undefined) setDprojections(s.dprojections);
    if (s.repetitions !== undefined) setRepetitions(s.repetitions);
    if (s.ef !== undefined) setEf(s.ef);
    if (s.showSubset !== undefined) setShowSubset(s.showSubset);
    if (s.subsetStart !== undefined) setSubsetStart(s.subsetStart);
    if (s.subsetEnd !== undefined) setSubsetEnd(s.subsetEnd);
    if (s.tenantId !== undefined) setTenantId(s.tenantId);

    setInitialized(true);

    // Check for an active job to reconnect to
    const activeJob = loadActiveJob();
    if (activeJob) {
      pollJobStatus(activeJob.jobId, activeJob.startedAt);
    }

    return () => {
      pollRef.current = false;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [pollJobStatus]);

  // -------------------------------------------------------------------------
  // Stream handler for new Engram ingestion jobs
  // -------------------------------------------------------------------------
  const handleEngramStream = useCallback(async (body: Record<string, unknown>) => {
    const resp = await fetch("/api/backend/populate-db-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`Server returned ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let raw: Record<string, unknown>;
        try {
          raw = JSON.parse(trimmed);
        } catch {
          continue;
        }

        const parsed = parseProgressEvent(raw);

        if (parsed.type === "started" && parsed.jobId) {
          // Save job ID so we can reconnect if user navigates away
          saveActiveJob({ jobId: parsed.jobId, startedAt: Date.now() });
        } else if (parsed.type === "loading" || parsed.type === "progress") {
          if (parsed.progress) setProgress(parsed.progress);
        } else if (parsed.type === "complete" && parsed.manifest) {
          setEngramManifest(parsed.manifest);
          if (timerRef.current) clearInterval(timerRef.current);
          setProgress(null);
          setStatus("success");
          clearActiveJob();
          return;
        } else if (parsed.type === "error") {
          clearActiveJob();
          throw new Error(parsed.errorMessage || "Unknown error");
        }
      }
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("running");
    setElapsed(0);
    setErrorMsg("");
    setEngramManifest(null);
    setShowRuns(false);
    setProgress(null);

    timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);

    const body: Record<string, unknown> = {
      recreate,
      tag,
      dataset_name: datasetName,
      database_target: databaseTarget,
    };

    if (databaseTarget === "weaviate") {
      body.text_embedding_model = textEmbeddingModel;
      if (imageEmbeddingModel) body.image_embedding_model = imageEmbeddingModel;
      if (useMuvera) {
        body.use_MUVERA_encoding = true;
        body.ksim = ksim;
        body.dprojections = dprojections;
        body.repetitions = repetitions;
        body.ef = ef;
      }
    }

    if (datasetName.startsWith("longmemeval") && showSubset) {
      const trimmedTenant = tenantId.trim();
      if (trimmedTenant) {
        body.longmemeval_tenant_ids = trimmedTenant.split(/[,\s]+/).filter(Boolean);
      } else {
        body.longmemeval_subset_start = subsetStart;
        body.longmemeval_subset_end = subsetEnd;
      }
    }

    try {
      if (databaseTarget === "engram") {
        // Use streaming endpoint for Engram
        await handleEngramStream(body);
      } else {
        // Standard blocking request for Weaviate
        const resp = await fetch("/api/backend/populate-db", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (timerRef.current) clearInterval(timerRef.current);

        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || `Server returned ${resp.status}`);
        }

        const data = await resp.json();
        if (data.status === "error") throw new Error(data.error);
        setStatus("success");
      }
    } catch (err) {
      if (timerRef.current) clearInterval(timerRef.current);
      setProgress(null);
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  };

  const isLongMemEval = datasetName.startsWith("longmemeval");

  // Compute totals from manifest
  const manifestTotals = engramManifest ? (() => {
    let created = 0, updated = 0, deleted = 0;
    if (engramManifest.stats) {
      for (const s of engramManifest.stats) {
        created += s.total_created;
        updated += s.total_updated;
        deleted += s.total_deleted;
      }
    } else {
      for (const r of engramManifest.runs) {
        created += r.created ?? 0;
        updated += r.updated ?? 0;
        deleted += r.deleted ?? 0;
      }
    }
    return { created, updated, deleted };
  })() : null;

  // Get unique tenant IDs for filter
  const tenantIds = engramManifest
    ? [...new Set(engramManifest.runs.map((r) => r.tenant_id))].sort()
    : [];

  // Filtered runs
  const filteredRuns = engramManifest
    ? tenantFilter === "all"
      ? engramManifest.runs
      : engramManifest.runs.filter((r) => r.tenant_id === tenantFilter)
    : [];

  return (
    <div className="max-w-2xl mx-auto py-10">
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Home
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Populate Database</span>
      </nav>

      <div className="flex items-center justify-between mb-2">
        <h1 className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
          Populate Database
        </h1>
        <a
          href="/populate/engram-runs"
          className="brand-btn-secondary"
          style={{ fontSize: "0.75rem" }}
        >
          Engram Run History
        </a>
      </div>
      <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
        Configure and load datasets into Weaviate or Engram.
      </p>

      {/* Engram manifest success banner */}
      {status === "success" && engramManifest && manifestTotals && (
        <div
          className="rounded-lg p-5 mb-6"
          style={{ background: "rgba(97,189,115,0.08)", border: "1px solid var(--color-green)" }}
        >
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm font-semibold" style={{ color: "var(--color-green)" }}>
              Engram ingestion completed in {elapsed}s
            </span>
            <a href="/populate/engram-runs" className="brand-btn-secondary" style={{ fontSize: "0.75rem" }}>
              View All Runs
            </a>
          </div>

          {/* Summary stats grid */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="rounded-md p-3 text-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)" }}>
              <p className="eyebrow mb-1">Sessions</p>
              <p className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>{engramManifest.total_sessions}</p>
            </div>
            <div className="rounded-md p-3 text-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)" }}>
              <p className="eyebrow mb-1">Tenants</p>
              <p className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>{engramManifest.total_tenants}</p>
            </div>
            <div className="rounded-md p-3 text-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)" }}>
              <p className="eyebrow mb-1">Memories Created</p>
              <p className="text-lg font-bold" style={{ color: "var(--color-green)" }}>{manifestTotals.created}</p>
            </div>
          </div>

          {/* Per-tenant stats */}
          {engramManifest.stats && engramManifest.stats.length > 0 && (
            <div className="mb-4">
              <p className="eyebrow mb-2">Per-Tenant Summary</p>
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
                    {engramManifest.stats.map((s) => (
                      <tr key={s.tenant_id} style={{ borderTop: "1px solid var(--border-default)" }}>
                        <td className="px-3 py-1.5" style={{ color: "var(--text-primary)" }}>{s.tenant_id}</td>
                        <td className="text-right px-3 py-1.5" style={{ color: "var(--text-primary)" }}>{s.num_sessions}</td>
                        <td className="text-right px-3 py-1.5" style={{ color: "var(--color-green)" }}>{s.total_created}</td>
                        <td className="text-right px-3 py-1.5" style={{ color: "var(--text-primary)" }}>{s.total_updated}</td>
                        <td className="text-right px-3 py-1.5" style={{ color: "var(--text-muted)" }}>{s.total_deleted}</td>
                        <td className="text-right px-3 py-1.5" style={{ color: "var(--text-muted)" }}>{s.elapsed_seconds.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Expandable runs table */}
          <button
            type="button"
            onClick={() => setShowRuns(!showRuns)}
            className="text-xs cursor-pointer flex items-center gap-1"
            style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}
          >
            <span style={{ display: "inline-block", transform: showRuns ? "rotate(90deg)" : "none", transition: "transform 0.15s" }}>
              &#9654;
            </span>
            {showRuns ? "Hide" : "Show"} {engramManifest.runs.length} run records
          </button>

          {showRuns && (
            <div className="mt-3">
              {/* Tenant filter */}
              {tenantIds.length > 1 && (
                <div className="mb-2">
                  <select
                    value={tenantFilter}
                    onChange={(e) => setTenantFilter(e.target.value)}
                    className="rounded-md px-2 py-1 text-xs"
                    style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
                  >
                    <option value="all">All tenants ({engramManifest.runs.length})</option>
                    {tenantIds.map((tid) => (
                      <option key={tid} value={tid}>
                        {tid} ({engramManifest.runs.filter((r) => r.tenant_id === tid).length})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="rounded-md overflow-auto max-h-80" style={{ border: "1px solid var(--border-default)" }}>
                <table className="w-full text-xs" style={{ fontFamily: "var(--font-mono)" }}>
                  <thead className="sticky top-0">
                    <tr style={{ background: "var(--bg-surface)" }}>
                      <th className="text-left px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Tenant</th>
                      <th className="text-left px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Session ID</th>
                      <th className="text-left px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Date</th>
                      <th className="text-left px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Run ID</th>
                      <th className="text-center px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Status</th>
                      <th className="text-right px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Created</th>
                      <th className="text-right px-2 py-1.5" style={{ color: "var(--text-muted)" }}>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRuns.map((r) => (
                      <tr key={r.run_id} style={{ borderTop: "1px solid var(--border-default)" }}>
                        <td className="px-2 py-1" style={{ color: "var(--text-primary)" }}>{r.tenant_id}</td>
                        <td className="px-2 py-1" style={{ color: "var(--text-primary)", maxWidth: "120px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.session_id}</td>
                        <td className="px-2 py-1" style={{ color: "var(--text-muted)" }}>{r.session_date || "-"}</td>
                        <td className="px-2 py-1" style={{ color: "var(--text-muted)", maxWidth: "100px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.run_id}</td>
                        <td className="text-center px-2 py-1">
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
                        </td>
                        <td className="text-right px-2 py-1" style={{ color: "var(--color-green)" }}>{r.created ?? "-"}</td>
                        <td className="text-right px-2 py-1" style={{ color: "var(--text-muted)" }}>
                          {r.run_duration_seconds != null ? `${r.run_duration_seconds.toFixed(1)}s` : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Generic success banner (Weaviate) */}
      {status === "success" && !engramManifest && (
        <div
          className="rounded-lg p-4 mb-6 flex items-center justify-between"
          style={{ background: "rgba(97,189,115,0.12)", border: "1px solid var(--color-green)" }}
        >
          <span className="text-sm font-medium" style={{ color: "var(--color-green)" }}>
            Database populated successfully in {elapsed}s
          </span>
          <a href="/results" className="brand-btn-secondary" style={{ fontSize: "0.75rem" }}>
            View Results
          </a>
        </div>
      )}

      {status === "error" && (
        <div
          className="rounded-lg p-4 mb-6"
          style={{ background: "rgba(244,64,78,0.08)", border: "1px solid var(--color-coral)" }}
        >
          <p className="text-sm font-medium mb-1" style={{ color: "var(--color-coral)" }}>
            Population failed
          </p>
          <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", whiteSpace: "pre-wrap" }}>
            {errorMsg}
          </p>
          <button
            onClick={() => setStatus("idle")}
            className="brand-btn-secondary mt-3"
            style={{ fontSize: "0.75rem" }}
          >
            Try Again
          </button>
        </div>
      )}

      {/* Running status with live progress */}
      {status === "running" && (
        <div
          className="rounded-lg p-6 mb-6"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
        >
          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-5 h-5 border-2 border-t-transparent rounded-full animate-spin flex-shrink-0"
              style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
            />
            <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              {databaseTarget === "engram" ? "Populating Engram..." : "Populating database..."}
            </p>
            <span className="ml-auto text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              {elapsed}s
            </span>
          </div>

          {/* Poll-reconnect mode: waiting for first check */}
          {isPolling && !progress && (
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Reconnected to running job. Fetching latest status...
            </p>
          )}

          {/* Poll-reconnect mode: compact status snapshot */}
          {isPolling && progress && (
            <div className="space-y-2">
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Reconnected to running job. Checking every 30s for updates.
              </p>
              <div
                className="rounded-md p-3"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)" }}
              >
                <p className="eyebrow mb-2">Last status check</p>
                <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs" style={{ fontFamily: "var(--font-mono)" }}>
                  <span>
                    <span style={{ color: "var(--text-muted)" }}>phase: </span>
                    <span style={{ color: "var(--text-primary)" }}>
                      {progress.phase === "loading" && "loading dataset"}
                      {progress.phase === "submit" && `submitting ${progress.submitted ?? 0}/${progress.total ?? "?"}`}
                      {(progress.phase === "poll" || progress.phase === "poll_wait") && `processing ${progress.completed ?? 0}/${progress.total ?? "?"}`}
                    </span>
                  </span>
                  {progress.tenant_id && (
                    <span>
                      <span style={{ color: "var(--text-muted)" }}>tenant: </span>
                      <span style={{ color: "var(--text-primary)" }}>{progress.tenant_id}</span>
                    </span>
                  )}
                  {((progress.memories_created ?? 0) + (progress.memories_updated ?? 0) + (progress.memories_deleted ?? 0)) > 0 && (
                    <span>
                      <span style={{ color: "var(--text-muted)" }}>memory ops: </span>
                      <span style={{ color: "var(--color-green)" }}>
                        {(progress.memories_created ?? 0) + (progress.memories_updated ?? 0) + (progress.memories_deleted ?? 0)}
                      </span>
                      <span style={{ color: "var(--text-muted)" }}>
                        {" "}({progress.memories_created ?? 0}c {progress.memories_updated ?? 0}u {progress.memories_deleted ?? 0}d)
                      </span>
                    </span>
                  )}
                </div>
                {lastCheckedAt && (
                  <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
                    checked {new Date(lastCheckedAt).toLocaleTimeString()} — next check ~30s
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Live streaming mode: full progress bars */}
          {!isPolling && progress && (
            <div className="space-y-3">
              {progress.phase === "loading" && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {progress.message}
                </p>
              )}

              {progress.phase === "submit" && progress.total != null && (
                <>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span style={{ color: "var(--text-muted)" }}>
                        Submitting sessions
                      </span>
                      <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                        {progress.submitted}/{progress.total}
                        {progress.skipped ? ` (${progress.skipped} skipped)` : ""}
                      </span>
                    </div>
                    <div className="w-full rounded-full h-2" style={{ background: "var(--bg-surface)" }}>
                      <div
                        className="h-2 rounded-full transition-all duration-300"
                        style={{
                          background: "var(--color-green)",
                          width: `${((progress.submitted ?? 0) / progress.total) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                  <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    tenant: {progress.tenant_id}
                    {progress.session_id ? ` / session: ${progress.session_id}` : ""}
                  </p>
                </>
              )}

              {(progress.phase === "poll" || progress.phase === "poll_wait") && progress.total != null && (
                <>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span style={{ color: "var(--text-muted)" }}>
                        Processing memories
                      </span>
                      <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                        {progress.completed}/{progress.total} completed
                      </span>
                    </div>
                    <div className="w-full rounded-full h-2" style={{ background: "var(--bg-surface)" }}>
                      <div
                        className="h-2 rounded-full transition-all duration-300"
                        style={{
                          background: "var(--color-green)",
                          width: `${((progress.completed ?? 0) / progress.total) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs" style={{ fontFamily: "var(--font-mono)" }}>
                    <span>
                      <span style={{ color: "var(--text-muted)" }}>tenant: </span>
                      <span style={{ color: "var(--text-primary)" }}>{progress.tenant_id}</span>
                    </span>
                    <span>
                      <span style={{ color: "var(--text-muted)" }}>memory ops: </span>
                      <span style={{ color: "var(--color-green)" }}>
                        {(progress.memories_created ?? 0) + (progress.memories_updated ?? 0) + (progress.memories_deleted ?? 0)}
                      </span>
                      <span style={{ color: "var(--text-muted)" }}>
                        {" "}({progress.memories_created ?? 0}c {progress.memories_updated ?? 0}u {progress.memories_deleted ?? 0}d)
                      </span>
                    </span>
                  </div>
                  {progress.phase === "poll_wait" && (
                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      waiting on run {progress.run_id?.slice(0, 8)}... (status: {progress.run_status}, check #{progress.poll_checks})
                    </p>
                  )}
                </>
              )}
            </div>
          )}

          {/* Fallback for Weaviate (no streaming) */}
          {!progress && databaseTarget === "weaviate" && (
            <p className="text-xs text-center" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              {elapsed}s elapsed
            </p>
          )}
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="brand-card p-6 space-y-6">
        {/* Dataset & Target */}
        <div>
          <label className="eyebrow mb-2 block">Dataset</label>
          <select
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            disabled={status === "running"}
            className="w-full rounded-md px-3 py-2 text-sm"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-default)",
              color: "var(--text-primary)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {POPULATE_DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>

        <div>
          <label className="eyebrow mb-2 block">Database Target</label>
          <div className="flex gap-2">
            {(["weaviate", "engram"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setDatabaseTarget(t)}
                disabled={status === "running"}
                className="brand-btn-secondary flex-1"
                style={databaseTarget === t ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Weaviate-specific: tag, recreate, embedding models */}
        {databaseTarget === "weaviate" && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="eyebrow mb-2 block">Tag</label>
                <input
                  type="text"
                  value={tag}
                  onChange={(e) => setTag(e.target.value)}
                  disabled={status === "running"}
                  className="w-full rounded-md px-3 py-2 text-sm"
                  style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
                />
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Collection tag suffix</p>
              </div>
              <div className="flex flex-col justify-center">
                <div className="flex items-center justify-between">
                  <label className="eyebrow">Recreate</label>
                  <button
                    type="button"
                    onClick={() => setRecreate(!recreate)}
                    disabled={status === "running"}
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: recreate ? "var(--color-green)" : "var(--border-default)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 rounded-full transition-transform" style={{ background: "#fff", left: recreate ? "calc(100% - 18px)" : "2px" }} />
                  </button>
                </div>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Drop and recreate collection</p>
              </div>
            </div>

            <div>
              <label className="eyebrow mb-2 block">Text Embedding Model</label>
              <input
                type="text"
                value={textEmbeddingModel}
                onChange={(e) => setTextEmbeddingModel(e.target.value)}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
              />
            </div>

            <div>
              <label className="eyebrow mb-2 block">Image Embedding Model</label>
              <input
                type="text"
                value={imageEmbeddingModel}
                onChange={(e) => setImageEmbeddingModel(e.target.value)}
                disabled={status === "running"}
                placeholder="Leave empty if not needed"
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
              />
            </div>

            {/* MUVERA Section */}
            <details open={showMuvera} onToggle={(e) => setShowMuvera((e.target as HTMLDetailsElement).open)}>
              <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                MUVERA+HNSW Parameters
              </summary>
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <label className="eyebrow">Enable MUVERA Encoding</label>
                  <button
                    type="button"
                    onClick={() => setUseMuvera(!useMuvera)}
                    disabled={status === "running"}
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: useMuvera ? "var(--color-green)" : "var(--border-default)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 rounded-full transition-transform" style={{ background: "#fff", left: useMuvera ? "calc(100% - 18px)" : "2px" }} />
                  </button>
                </div>
                {useMuvera && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="eyebrow mb-2 block">ksim</label>
                      <input type="number" value={ksim} onChange={(e) => setKsim(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                    <div>
                      <label className="eyebrow mb-2 block">dprojections</label>
                      <input type="number" value={dprojections} onChange={(e) => setDprojections(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                    <div>
                      <label className="eyebrow mb-2 block">repetitions</label>
                      <input type="number" value={repetitions} onChange={(e) => setRepetitions(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                    <div>
                      <label className="eyebrow mb-2 block">ef</label>
                      <input type="number" value={ef} onChange={(e) => setEf(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                  </div>
                )}
              </div>
            </details>
          </>
        )}

        {/* LongMemEval Subset */}
        {isLongMemEval && (
          <details open={showSubset} onToggle={(e) => setShowSubset((e.target as HTMLDetailsElement).open)}>
            <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              LongMemEval Subset Selection
            </summary>
            <div className="mt-4 space-y-4">
              <div>
                <label className="eyebrow mb-2 block">Tenant ID</label>
                <input
                  type="text"
                  value={tenantId}
                  onChange={(e) => setTenantId(e.target.value)}
                  disabled={status === "running"}
                  placeholder="e.g. user_123 or user_1, user_2"
                  className="w-full rounded-md px-3 py-2 text-sm"
                  style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
                />
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  Load a specific tenant by ID. Comma or space-separated for multiple. Overrides index range if set.
                </p>
              </div>
              <div
                className="rounded-md p-3"
                style={{
                  background: tenantId.trim() ? "var(--bg-surface)" : "transparent",
                  opacity: tenantId.trim() ? 0.5 : 1,
                }}
              >
                <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                  Or filter tenants by sorted index range [start, end).
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="eyebrow mb-2 block">Start Index</label>
                    <input type="number" min={0} value={subsetStart} onChange={(e) => setSubsetStart(Number(e.target.value))} disabled={status === "running" || !!tenantId.trim()} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                  <div>
                    <label className="eyebrow mb-2 block">End Index</label>
                    <input type="number" min={0} value={subsetEnd} onChange={(e) => setSubsetEnd(Number(e.target.value))} disabled={status === "running" || !!tenantId.trim()} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                </div>
              </div>
            </div>
          </details>
        )}

        <button
          type="submit"
          disabled={status === "running"}
          className="w-full rounded-md px-4 py-2.5 text-sm font-semibold transition-opacity"
          style={{
            background: "var(--color-green)",
            color: "var(--color-navy)",
            opacity: status === "running" ? 0.5 : 1,
          }}
        >
          {status === "running" ? "Running..." : "Populate Database"}
        </button>
      </form>
    </div>
  );
}
