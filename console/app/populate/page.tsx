"use client";

import { useState, useRef, useEffect } from "react";
import { SEARCH_DATASETS, ASK_DATASETS } from "@/lib/datasets";

const POPULATE_DATASETS = [...new Set([...SEARCH_DATASETS, ...ASK_DATASETS])].sort();

export default function PopulatePage() {
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

  // Status
  const [status, setStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("running");
    setElapsed(0);
    setErrorMsg("");

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
      body.longmemeval_subset_start = subsetStart;
      body.longmemeval_subset_end = subsetEnd;
    }

    try {
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
    } catch (err) {
      if (timerRef.current) clearInterval(timerRef.current);
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  };

  const isLongMemEval = datasetName.startsWith("longmemeval");

  return (
    <div className="max-w-2xl mx-auto py-10">
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Home
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Populate Database</span>
      </nav>

      <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
        Populate Database
      </h1>
      <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
        Configure and load datasets into Weaviate or Engram.
      </p>

      {/* Status banners */}
      {status === "success" && (
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

      {status === "running" && (
        <div
          className="rounded-lg p-6 mb-6 text-center"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
        >
          <div
            className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3"
            style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
          />
          <p className="text-sm mb-1" style={{ color: "var(--text-primary)" }}>Populating database...</p>
          <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {elapsed}s elapsed
          </p>
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
            <div className="mt-4">
              <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                Filter tenants by sorted index range. Only sessions for tenants within [start, end) are loaded.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="eyebrow mb-2 block">Start Index</label>
                  <input type="number" min={0} value={subsetStart} onChange={(e) => setSubsetStart(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
                </div>
                <div>
                  <label className="eyebrow mb-2 block">End Index</label>
                  <input type="number" min={0} value={subsetEnd} onChange={(e) => setSubsetEnd(Number(e.target.value))} disabled={status === "running"} className="w-full rounded-md px-3 py-2 text-sm" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-default)", color: "var(--text-primary)", fontFamily: "var(--font-mono)" }} />
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
