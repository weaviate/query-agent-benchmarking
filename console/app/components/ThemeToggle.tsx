"use client";

import { useEffect, useState } from "react";

/** Dark is the brand default; the toggle opts into the light surface by
 *  putting a `light` class on <html> (mirrors the init script in layout). */
export default function ThemeToggle() {
  const [light, setLight] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setLight(localStorage.getItem("theme") === "light");
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const root = document.documentElement;
    if (light) {
      root.classList.add("light");
      localStorage.setItem("theme", "light");
    } else {
      root.classList.remove("light");
      localStorage.setItem("theme", "dark");
    }
  }, [light, mounted]);

  // Avoid hydration mismatch — render a placeholder until mounted
  if (!mounted) {
    return <div className="w-9 h-9" />;
  }

  // The button lives in the always-navy header, so both states style for dark.
  return (
    <button
      onClick={() => setLight((l) => !l)}
      className="w-9 h-9 flex items-center justify-center rounded-md transition-colors cursor-pointer"
      style={{
        background: light ? "rgba(255,255,255,0.12)" : "rgba(1,245,122,0.14)",
        color: light ? "rgba(255,255,255,0.85)" : "#01F57A",
      }}
      title={light ? "Switch to dark mode" : "Switch to light mode"}
      aria-label={light ? "Switch to dark mode" : "Switch to light mode"}
    >
      {light ? (
        /* Moon icon — switch back to dark */
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : (
        /* Sun icon — switch to light */
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      )}
    </button>
  );
}
