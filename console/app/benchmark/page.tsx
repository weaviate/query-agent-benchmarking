"use client";

import { useState, useRef, useEffect } from "react";
import { SEARCH_DATASETS, ASK_DATASETS } from "@/lib/datasets";

type Tab = "search" | "ask";
type Status = "idle" | "running" | "success" | "error";

export default function BenchmarkPage() {
  const [tab, setTab] = useState<Tab>("search");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Search form state
  const [searchDataset, setSearchDataset] = useState(SEARCH_DATASETS[0]);
  const [searchAgent, setSearchAgent] = useState("external_service");
  const [searchTrials, setSearchTrials] = useState(1);
  const [searchAdvanced, setSearchAdvanced] = useState(false);
  const [searchAsync, setSearchAsync] = useState(true);
  const [searchBatchSize, setSearchBatchSize] = useState(5);
  const [searchMaxConcurrent, setSearchMaxConcurrent] = useState(5);
  const [searchEmbeddingModel, setSearchEmbeddingModel] = useState("");
  const [searchTarget, setSearchTarget] = useState("");

  // Ask form state
  const [askDataset, setAskDataset] = useState(ASK_DATASETS[0]);
  const [askAgent, setAskAgent] = useState("engram");
  const [askTrials, setAskTrials] = useState(1);
  const [askAdvanced, setAskAdvanced] = useState(false);
  const [askJudgeModel, setAskJudgeModel] = useState("openai/gpt-5.4");
  const [askEnsembleK, setAskEnsembleK] = useState(3);
  const [askUseReasoning, setAskUseReasoning] = useState(true);
  const [askAsync, setAskAsync] = useState(true);
  const [askBatchSize, setAskBatchSize] = useState(5);
  const [askMaxConcurrent, setAskMaxConcurrent] = useState(5);
  const [askSubsetStart, setAskSubsetStart] = useState(0);
  const [askSubsetEnd, setAskSubsetEnd] = useState(20);
  const [askTenantId, setAskTenantId] = useState("");

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("running");
    setElapsed(0);
    setErrorMsg("");

    timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);

    try {
      let endpoint: string;
      let body: Record<string, unknown>;

      if (tab === "search") {
        endpoint = "/api/backend/run-search-benchmark";
        body = {
          search_dataset: searchDataset,
          agent_name: searchAgent,
          num_trials: searchTrials,
          use_async: searchAsync,
          batch_size: searchBatchSize,
          max_concurrent: searchMaxConcurrent,
        };
        if (searchEmbeddingModel) body.embedding_model = searchEmbeddingModel;
        if (searchTarget) body.search_target = searchTarget;
      } else {
        endpoint = "/api/backend/run-ask-benchmark";
        body = {
          queries: askDataset,
          agent_name: askAgent,
          num_trials: askTrials,
          judge_model: askJudgeModel,
          ensemble_k: askEnsembleK,
          use_reasoning: askUseReasoning,
          use_async: askAsync,
          batch_size: askBatchSize,
          max_concurrent: askMaxConcurrent,
        };
        if (askDataset.startsWith("longmemeval")) {
          const trimmedTenant = askTenantId.trim();
          if (trimmedTenant) {
            body.longmemeval_tenant_ids = trimmedTenant.split(/[,\s]+/).filter(Boolean);
          } else {
            body.longmemeval_subset_start = askSubsetStart;
            body.longmemeval_subset_end = askSubsetEnd;
          }
        }
      }

      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (timerRef.current) clearInterval(timerRef.current);

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `Server returned ${resp.status}`);
      }

      setStatus("success");
    } catch (err) {
      if (timerRef.current) clearInterval(timerRef.current);
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  };

  return (
    <div className="max-w-2xl mx-auto py-10">
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Home
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Run Benchmark</span>
      </nav>

      <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
        Run Benchmark
      </h1>
      <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
        Execute search or ask benchmarks against your populated collections.
      </p>

      {/* Status banners */}
      {status === "success" && (
        <div
          className="rounded-lg p-4 mb-6 flex items-center justify-between"
          style={{ background: "rgba(97,189,115,0.12)", border: "1px solid var(--color-green)" }}
        >
          <span className="text-sm font-medium" style={{ color: "var(--color-green)" }}>
            Benchmark completed in {elapsed}s
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
            Benchmark failed
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

      {status === "running" && (
        <div
          className="rounded-lg p-6 mb-6 text-center"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
        >
          <div
            className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3"
            style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
          />
          <p className="text-sm mb-1" style={{ color: "var(--text-primary)" }}>Running {tab} benchmark...</p>
          <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {elapsed}s elapsed
          </p>
        </div>
      )}

      {/* Tab switcher */}
      <div
        className="flex gap-1.5 items-center rounded-lg px-4 py-3 mb-6"
        style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
      >
        {(["search", "ask"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            disabled={status === "running"}
            className="brand-btn-secondary"
            style={tab === t ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" } : {}}
          >
            {t === "search" ? "Search" : "Ask"}
          </button>
        ))}
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="brand-card p-6 space-y-5">
        {tab === "search" ? (
          <>
            <div>
              <label className="eyebrow mb-2 block">Dataset</label>
              <select
                value={searchDataset}
                onChange={(e) => setSearchDataset(e.target.value)}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {SEARCH_DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div>
              <label className="eyebrow mb-2 block">Agent Name</label>
              <input
                type="text"
                value={searchAgent}
                onChange={(e) => setSearchAgent(e.target.value)}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            <div>
              <label className="eyebrow mb-2 block">Num Trials</label>
              <input
                type="number"
                min={1}
                value={searchTrials}
                onChange={(e) => setSearchTrials(Number(e.target.value))}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            {/* Advanced */}
            <details open={searchAdvanced} onToggle={(e) => setSearchAdvanced((e.target as HTMLDetailsElement).open)}>
              <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                Advanced Settings
              </summary>
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <label className="eyebrow">Async</label>
                  <button
                    type="button"
                    onClick={() => setSearchAsync(!searchAsync)}
                    disabled={status === "running"}
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: searchAsync ? "var(--color-green)" : "var(--border-default)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 rounded-full transition-transform" style={{ background: "#fff", left: searchAsync ? "calc(100% - 18px)" : "2px" }} />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="eyebrow mb-2 block">Batch Size</label>
                    <input type="number" min={1} value={searchBatchSize} onChange={(e) => setSearchBatchSize(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                  <div>
                    <label className="eyebrow mb-2 block">Max Concurrent</label>
                    <input type="number" min={1} value={searchMaxConcurrent} onChange={(e) => setSearchMaxConcurrent(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                </div>
                <div>
                  <label className="eyebrow mb-2 block">Embedding Model</label>
                  <input type="text" value={searchEmbeddingModel} onChange={(e) => setSearchEmbeddingModel(e.target.value)} disabled={status === "running"} placeholder="Leave empty for default" className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                </div>
                <div>
                  <label className="eyebrow mb-2 block">Search Target</label>
                  <input type="text" value={searchTarget} onChange={(e) => setSearchTarget(e.target.value)} disabled={status === "running"} placeholder="Leave empty for default" className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                </div>
              </div>
            </details>
          </>
        ) : (
          <>
            <div>
              <label className="eyebrow mb-2 block">Dataset</label>
              <select
                value={askDataset}
                onChange={(e) => setAskDataset(e.target.value)}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {ASK_DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div>
              <label className="eyebrow mb-2 block">Agent Name</label>
              <input
                type="text"
                value={askAgent}
                onChange={(e) => setAskAgent(e.target.value)}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            <div>
              <label className="eyebrow mb-2 block">Num Trials</label>
              <input
                type="number"
                min={1}
                value={askTrials}
                onChange={(e) => setAskTrials(Number(e.target.value))}
                disabled={status === "running"}
                className="w-full rounded-md px-3 py-2 text-sm"
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-default)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            {/* LongMemEval Subset (shown above advanced when relevant) */}
            {askDataset.startsWith("longmemeval") && (
              <div
                className="rounded-md p-4 space-y-3"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)" }}
              >
                <label className="eyebrow block">LongMemEval Subset</label>
                <div>
                  <label className="eyebrow mb-1 block">Tenant ID</label>
                  <input
                    type="text"
                    value={askTenantId}
                    onChange={(e) => setAskTenantId(e.target.value)}
                    disabled={status === "running"}
                    placeholder="e.g. user_123 or user_1, user_2"
                    className="w-full rounded-md px-3 py-2 text-sm"
                    style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}
                  />
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    Run benchmark for a specific tenant. Comma or space-separated for multiple. Overrides index range if set.
                  </p>
                </div>
                <div
                  style={{
                    opacity: askTenantId.trim() ? 0.5 : 1,
                  }}
                >
                  <p className="text-xs mb-2" style={{ color: "var(--text-muted)" }}>
                    Or use tenant index range [start, end) — must match what was ingested.
                  </p>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="eyebrow mb-1 block">Start</label>
                      <input type="number" min={0} value={askSubsetStart} onChange={(e) => setAskSubsetStart(Number(e.target.value))} disabled={status === "running" || !!askTenantId.trim()} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                    <div>
                      <label className="eyebrow mb-1 block">End</label>
                      <input type="number" min={0} value={askSubsetEnd} onChange={(e) => setAskSubsetEnd(Number(e.target.value))} disabled={status === "running" || !!askTenantId.trim()} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Advanced */}
            <details open={askAdvanced} onToggle={(e) => setAskAdvanced((e.target as HTMLDetailsElement).open)}>
              <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                Advanced Settings
              </summary>
              <div className="mt-4 space-y-4">
                <div>
                  <label className="eyebrow mb-2 block">Judge Model</label>
                  <input type="text" value={askJudgeModel} onChange={(e) => setAskJudgeModel(e.target.value)} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                </div>
                <div>
                  <label className="eyebrow mb-2 block">Ensemble K</label>
                  <input type="number" min={1} value={askEnsembleK} onChange={(e) => setAskEnsembleK(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                </div>
                <div className="flex items-center justify-between">
                  <label className="eyebrow">Use Reasoning</label>
                  <button
                    type="button"
                    onClick={() => setAskUseReasoning(!askUseReasoning)}
                    disabled={status === "running"}
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: askUseReasoning ? "var(--color-green)" : "var(--border-default)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 rounded-full transition-transform" style={{ background: "#fff", left: askUseReasoning ? "calc(100% - 18px)" : "2px" }} />
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <label className="eyebrow">Async</label>
                  <button
                    type="button"
                    onClick={() => setAskAsync(!askAsync)}
                    disabled={status === "running"}
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: askAsync ? "var(--color-green)" : "var(--border-default)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 rounded-full transition-transform" style={{ background: "#fff", left: askAsync ? "calc(100% - 18px)" : "2px" }} />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="eyebrow mb-2 block">Batch Size</label>
                    <input type="number" min={1} value={askBatchSize} onChange={(e) => setAskBatchSize(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                  <div>
                    <label className="eyebrow mb-2 block">Max Concurrent</label>
                    <input type="number" min={1} value={askMaxConcurrent} onChange={(e) => setAskMaxConcurrent(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                  </div>
                </div>
              </div>
            </details>
          </>
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
          {status === "running" ? "Running..." : `Run ${tab === "search" ? "Search" : "Ask"} Benchmark`}
        </button>
      </form>
    </div>
  );
}
