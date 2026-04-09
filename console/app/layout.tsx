import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Benchmark Result Visualizer",
  description: "Dashboard for Weaviate Query Agent benchmark results",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <header className="border-b border-gray-200 dark:border-gray-800 px-6 py-4">
          <a href="/" className="text-lg font-semibold font-[family-name:var(--font-geist-sans)]">
            Benchmark Results
          </a>
        </header>
        <main className="px-6 py-6 max-w-7xl mx-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
