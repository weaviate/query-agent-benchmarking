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
  title: "Weaviate Benchmark Console",
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
      <body className={`${plusJakarta.variable} ${inter.variable} antialiased`}>
        <header className="brand-header px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-3 group">
            {/* Hexagonal W mark */}
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" className="shrink-0">
              <path
                d="M16 1.07L29.86 9.07V24.93L16 30.93L2.14 24.93V9.07L16 1.07Z"
                fill="none"
                stroke="#61BD73"
                strokeWidth="1.5"
              />
              <path
                d="M8 12L11.5 22L16 14L20.5 22L24 12"
                stroke="#61BD73"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
            <div>
              <span className="text-lg font-bold tracking-tight" style={{ fontFamily: "var(--font-display)" }}>
                Benchmark Console
              </span>
              <span className="hidden sm:inline text-xs ml-2 opacity-50" style={{ fontFamily: "var(--font-mono)" }}>
                v0.1
              </span>
            </div>
          </a>
          <nav className="flex items-center gap-4">
            <a
              href="/"
              className="text-sm opacity-70 hover:opacity-100 transition-opacity"
              style={{ fontFamily: "var(--font-body)" }}
            >
              Experiments
            </a>
            <ThemeToggle />
          </nav>
        </header>
        <main className="px-6 py-8 max-w-7xl mx-auto hex-pattern min-h-[calc(100vh-64px)]">
          {children}
        </main>
      </body>
    </html>
  );
}
