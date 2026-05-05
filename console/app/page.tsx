import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="py-12">
      <div className="text-center mb-12">
        <h1
          className="text-3xl font-bold mb-3"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Query Agent Benchmarking
        </h1>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Populate databases, run benchmarks, and view results
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
        {/* Result Viewer */}
        <Link href="/results" className="brand-card p-6 hover:border-[var(--color-green)] transition-colors group">
          <div className="mb-4">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--color-green)" }}>
              <path d="M3 3v18h18" />
              <path d="M7 16l4-4 4 4 5-5" />
            </svg>
          </div>
          <h2 className="text-lg font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
            Result Viewer
          </h2>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Browse, compare, and analyze benchmark results across experiments and trials.
          </p>
        </Link>

        {/* Populate Database */}
        <Link href="/populate" className="brand-card p-6 hover:border-[var(--color-teal)] transition-colors group">
          <div className="mb-4">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--color-teal)" }}>
              <ellipse cx="12" cy="5" rx="9" ry="3" />
              <path d="M3 5v7c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
              <path d="M3 12v7c0 1.66 4.03 3 9 3s9-1.34 9-3v-7" />
            </svg>
          </div>
          <h2 className="text-lg font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
            Populate Database
          </h2>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Load datasets into Weaviate or Engram for benchmarking.
          </p>
        </Link>

        {/* Run Benchmark */}
        <Link href="/benchmark" className="brand-card p-6 hover:border-[var(--color-lavender)] transition-colors group">
          <div className="mb-4">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--color-lavender)" }}>
              <polygon points="5,3 19,12 5,21" />
            </svg>
          </div>
          <h2 className="text-lg font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
            Run Benchmark
          </h2>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Execute search or ask benchmarks and evaluate agent performance.
          </p>
        </Link>
      </div>
    </div>
  );
}
