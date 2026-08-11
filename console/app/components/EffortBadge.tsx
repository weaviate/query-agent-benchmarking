/** Badge for the Query Agent search-mode compute effort level, or the
 *  hybrid-search baseline included in an effort sweep. */
export function EffortBadge({ effort }: { effort: string | null | undefined }) {
  if (!effort) return null;
  // Effort levels: medium (cyan) → high (blue-green) → ultrahigh (sky),
  // matching the sweep chart bars. Baselines sit outside the sweep, so they
  // take the theme-aware navy baseline color.
  const styles: Record<string, { bg: string; fg: string }> = {
    medium: { bg: "rgba(1,198,201,0.14)", fg: "var(--color-cyan)" },
    high: { bg: "rgba(1,222,160,0.14)", fg: "var(--color-blue-green)" },
    ultrahigh: { bg: "rgba(122,214,235,0.14)", fg: "var(--color-sky)" },
    hybrid: { bg: "color-mix(in srgb, var(--color-baseline) 15%, transparent)", fg: "var(--color-baseline)" },
  };
  const labels: Record<string, string> = { hybrid: "Hybrid Search" };
  const isEffortLevel = effort === "medium" || effort === "high" || effort === "ultrahigh";
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
