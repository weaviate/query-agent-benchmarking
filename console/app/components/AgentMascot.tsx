/**
 * The Query Agent mascot: a flat-top hexagon with a face — agent-gradient
 * fill, two rounded dark eyes, a thin outlined hexagon frame, and an antenna
 * dot at the top. Construction rules live in STYLES.md ("The agent mascot").
 *
 * The frame color follows --mascot-frame (white on dark surfaces, navy on
 * light). Never rotate to pointy-top; never remove the face.
 */
export default function AgentMascot({
  size = 64,
  className,
  frameColor,
}: {
  size?: number;
  className?: string;
  /** Override the outline color (defaults to var(--mascot-frame)). */
  frameColor?: string;
}) {
  const frame = frameColor ?? "var(--mascot-frame)";
  return (
    <svg
      width={size}
      height={size * (124 / 120)}
      viewBox="-60 -64 120 124"
      fill="none"
      className={className}
      role="img"
      aria-label="Query Agent"
    >
      <defs>
        {/* Lower-left cyan → upper-right green (--gradient-agent) */}
        <linearGradient id="qa-agent-grad" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#01C6C9" />
          <stop offset="100%" stopColor="#01F57A" />
        </linearGradient>
      </defs>

      {/* Antenna dot, just off the top edge of the frame */}
      <circle cx="0" cy="-53" r="4" fill="var(--color-green, #01F57A)" />

      {/* Thin outline frame (flat-top hexagon) */}
      <polygon
        points="52,0 26,45 -26,45 -52,0 -26,-45 26,-45"
        stroke={frame}
        strokeWidth="2.5"
        strokeLinejoin="round"
      />

      {/* Gradient-filled body (same-color stroke rounds the corners) */}
      <polygon
        points="40,0 20,34.6 -20,34.6 -40,0 -20,-34.6 20,-34.6"
        fill="url(#qa-agent-grad)"
        stroke="url(#qa-agent-grad)"
        strokeWidth="6"
        strokeLinejoin="round"
      />

      {/* Eyes */}
      <rect x="-19" y="-14" width="12" height="27" rx="6" fill="#000335" />
      <rect x="7" y="-14" width="12" height="27" rx="6" fill="#000335" />
    </svg>
  );
}
