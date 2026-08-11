import Link from "next/link";
import AgentMascot from "./components/AgentMascot";

/* Flat-top hexagon tile with a line icon — the "capability icon" pattern.
   The tile stays navy in both themes; icon strokes come from the spectrum. */
function HexTile({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: 64, height: 56 }}>
      <svg width="64" height="56" viewBox="-32 -28 64 56" fill="none" className="absolute inset-0">
        <polygon
          points="30,0 15,26 -15,26 -30,0 -15,-26 15,-26"
          fill="var(--color-navy-darkest)"
          stroke={color}
          strokeOpacity="0.55"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
      <span className="relative" style={{ color }}>
        {children}
      </span>
    </div>
  );
}

const steps = [
  {
    step: "01",
    href: "/populate",
    title: "Populate Database",
    body: "Load benchmark datasets into Weaviate or Engram collections.",
    color: "var(--color-cyan)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v7c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
        <path d="M3 12v7c0 1.66 4.03 3 9 3s9-1.34 9-3v-7" />
      </svg>
    ),
  },
  {
    step: "02",
    href: "/benchmark",
    title: "Run Benchmark",
    body: "Execute search or ask benchmarks and evaluate agent performance.",
    color: "var(--color-blue-green)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
        <polygon points="6,4 20,12 6,20" />
      </svg>
    ),
  },
  {
    step: "03",
    href: "/results",
    title: "View Results",
    body: "Browse, compare, and analyze results across experiments and trials.",
    color: "var(--color-green)",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 3v18h18" />
        <path d="M7 16l4-4 4 4 5-5" />
      </svg>
    ),
  },
];

export default function LandingPage() {
  return (
    <div className="py-8">
      {/* Hero — always on navy; agent gradients don't leave dark surfaces. */}
      <section
        className="relative overflow-hidden rounded-2xl px-8 py-14 mb-10 text-center"
        style={{
          background: "var(--gradient-navy)",
          border: "1px solid rgba(1, 245, 122, 0.18)",
        }}
      >
        {/* Dotted orbit arcs + ambient glyphs (agent activity motif) */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 800 380"
          preserveAspectRatio="xMidYMid slice"
          fill="none"
          aria-hidden="true"
        >
          <circle cx="400" cy="120" r="105" stroke="#DDEBF2" strokeOpacity="0.25" strokeWidth="1.5" strokeDasharray="1 8" strokeLinecap="round" />
          <circle cx="400" cy="120" r="150" stroke="#01F57A" strokeOpacity="0.3" strokeWidth="1.5" strokeDasharray="1 10" strokeLinecap="round" />
          <path d="M120 300 H 300" stroke="#DDEBF2" strokeOpacity="0.2" strokeWidth="1.5" strokeDasharray="1 7" strokeLinecap="round" />
          <path d="M500 300 H 680" stroke="#01F57A" strokeOpacity="0.25" strokeWidth="1.5" strokeDasharray="1 7" strokeLinecap="round" />
          <path d="M130 80 l8 8 m0 -8 l-8 8" stroke="#DDEBF2" strokeOpacity="0.3" strokeWidth="1.5" strokeLinecap="round" />
          <path d="M660 200 l8 8 m0 -8 l-8 8" stroke="#DDEBF2" strokeOpacity="0.25" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="700" cy="70" r="5" stroke="#01C6C9" strokeOpacity="0.4" strokeWidth="1.5" />
          <rect x="90" y="190" width="9" height="9" stroke="#DDEBF2" strokeOpacity="0.25" strokeWidth="1.5" />
        </svg>

        <div className="relative flex flex-col items-center">
          <AgentMascot size={88} frameColor="#ffffff" className="mb-6" />
          <p className="eyebrow mb-3" style={{ color: "rgba(221, 235, 242, 0.6)" }}>
            Weaviate Query Agent
          </p>
          <h1
            className="text-4xl sm:text-5xl font-bold mb-4 tracking-tight"
            style={{ fontFamily: "var(--font-display)", color: "#ffffff" }}
          >
            Benchmark the <span className="agent-gradient-text">Query Agent</span>
          </h1>
          <p className="text-base max-w-xl" style={{ color: "#DDEBF2" }}>
            Populate databases, run search and ask benchmarks, and compare
            agent performance across experiments.
          </p>
        </div>
      </section>

      {/* The benchmarking workflow, in order */}
      <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
        {steps.map((s) => (
          <Link key={s.href} href={s.href} className="brand-card p-6 group">
            <div className="flex items-start justify-between mb-4">
              <HexTile color={s.color}>{s.icon}</HexTile>
              <span className="eyebrow" style={{ color: s.color }}>
                {s.step}
              </span>
            </div>
            <h2 className="text-lg font-bold mb-2" style={{ fontFamily: "var(--font-display)" }}>
              {s.title}
            </h2>
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              {s.body}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
