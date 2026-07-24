import type { Metadata } from "next";
import { Montserrat, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import ThemeToggle from "./components/ThemeToggle";
import AgentMascot from "./components/AgentMascot";

const montserrat = Montserrat({
  variable: "--font-display-loaded",
  subsets: ["latin"],
  weight: ["500", "700"],
  display: "swap",
});

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-body-loaded",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "query-agent-benchmarking",
  description: "Dashboard for Weaviate Query Agent benchmark results",
};

// Dark is the brand default; apply the light class before first paint only
// if the user explicitly chose it. Prevents FOUC.
const themeInitScript = `
(function() {
  try {
    if (localStorage.getItem('theme') === 'light') {
      document.documentElement.classList.add('light');
    }
  } catch(e) {}
})();
`;

const navLinks = [
  { href: "/populate", label: "Populate" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/results", label: "Results" },
];

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
      <body className={`${montserrat.variable} ${plusJakarta.variable} antialiased hex-pattern`}>
        <header className="brand-header px-6 py-3 flex items-center justify-between">
          <a href="/" className="flex items-center gap-3 group">
            {/* Header is always navy, so the frame is always white here */}
            <AgentMascot size={30} frameColor="#ffffff" className="shrink-0" />
            <span
              className="text-lg font-bold tracking-tight leading-none"
              style={{ fontFamily: "var(--font-display)" }}
            >
              query-agent-benchmarking
            </span>
            <span
              className="hidden sm:inline text-xs leading-none"
              style={{ fontFamily: "var(--font-mono)", color: "rgba(221, 235, 242, 0.5)" }}
            >
              v0.7
            </span>
          </a>
          <nav className="flex items-center gap-5">
            {navLinks.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="text-sm opacity-70 hover:opacity-100 transition-opacity"
                style={{ fontFamily: "var(--font-body)", color: "#DDEBF2" }}
              >
                {l.label}
              </a>
            ))}
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
