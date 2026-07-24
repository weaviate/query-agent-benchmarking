/** Badge for the Query Agent search-mode compute effort level, or the
 *  hybrid-search baseline included in an effort sweep. */
export function EffortBadge({ effort }: { effort: string | null | undefined }) {
  if (!effort) return null;
  // Effort levels walk the agent spectrum: low (cyan) → high (green).
  // Baselines sit outside the sweep, so they take the theme-aware baseline
  // color (white on dark, blue on light).
  const styles: Record<string, { bg: string; fg: string }> = {
    low: { bg: "rgba(1,198,201,0.14)", fg: "var(--color-cyan)" },
    medium: { bg: "rgba(1,222,160,0.14)", fg: "var(--color-blue-green)" },
    high: { bg: "rgba(1,245,122,0.14)", fg: "var(--color-green)" },
    hybrid: { bg: "color-mix(in srgb, var(--color-baseline) 15%, transparent)", fg: "var(--color-baseline)" },
  };
  const labels: Record<string, string> = { hybrid: "Hybrid Search" };
  const isEffortLevel = effort === "low" || effort === "medium" || effort === "high";
  const s = styles[effort] || { bg: "rgba(136,150,171,0.15)", fg: "var(--text-muted)" };
  return (
    <span
      className="brand-badge"
      style={{ background: s.bg, color: s.fg }}
      title={isEffortLevel ? "Search-mode compute effort" : "Baseline agent"}
    >
      {isEffortLevel ? `effort: ${effort}` : labels[effort] ?? effort}
    </span>
  );
}
