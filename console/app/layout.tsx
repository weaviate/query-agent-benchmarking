import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Inter } from "next/font/google";
import "./globals.css";
import ThemeToggle from "./components/ThemeToggle";

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-display-loaded",
  subsets: ["latin"],
  weight: ["500", "700"],
  display: "swap",
});

const inter = Inter({
  variable: "--font-body-loaded",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "query-agent-benchmarking",
  description: "Dashboard for Weaviate Query Agent benchmark results",
};

// Inline script to apply dark class before first paint, preventing FOUC
const themeInitScript = `
(function() {
  try {
    var t = localStorage.getItem('theme');
    if (t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }
  } catch(e) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className={`${plusJakarta.variable} ${inter.variable} antialiased hex-pattern`}>
        <header className="brand-header px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-3 group">
            <img
              src="/weaviate-logo.png"
              alt="Weaviate"
              className="shrink-0 block"
              style={{ height: 32, width: "auto", marginTop: 4 }}
            />
            <span
              className="text-lg font-bold tracking-tight leading-none"
              style={{ fontFamily: "var(--font-display)" }}
            >
              query-agent-benchmarking
            </span>
            <span
              className="hidden sm:inline text-xs leading-none opacity-50"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              v0.7
            </span>
          </a>
          <nav className="flex items-center gap-4">
            <a
              href="/results"
              className="text-sm opacity-70 hover:opacity-100 transition-opacity"
              style={{ fontFamily: "var(--font-body)" }}
            >
              Results
            </a>
            <a
              href="/populate"
              className="text-sm opacity-70 hover:opacity-100 transition-opacity"
              style={{ fontFamily: "var(--font-body)" }}
            >
              Populate
            </a>
            <a
              href="/benchmark"
              className="text-sm opacity-70 hover:opacity-100 transition-opacity"
              style={{ fontFamily: "var(--font-body)" }}
            >
              Benchmark
            </a>
            <ThemeToggle />
          </nav>
        </header>
        <main className="px-6 py-8 max-w-7xl mx-auto min-h-[calc(100vh-64px)]">
          {children}
        </main>
      </body>
    </html>
  );
}
