"use client";

import { Fragment, Suspense, useEffect, useState, useMemo, useRef } from "react";
import { useSearchParams } from "next/navigation";
import type {
  QueryComparison,
  QueryComparisonRow,
  ComparisonCell,
  ComparisonOutcome,
} from "@/lib/results";
import { SearchPlan } from "@/app/components/SearchPlan";
import { EffortBadge } from "@/app/components/EffortBadge";

interface CompareExperiment {
  id: string;
  label: string;
  dataset: string;
  agent_name: string;
  mode: string;
  num_trials: number;
  timestamp: string;
  effort: string | null;
  metrics: Record<string, number | null>;
  metricsStd: Record<string, number | null>;
  // In aggregate mode each "trial" is one constituent dataset (label set).
  metricsTrials: Record<string, { trial: number; label?: string; value: number | null }[]>;
  // Aggregate mode only: the experiments averaged into this group.
  constituents?: { id: string; dataset: string; label: string }[];
}

interface CompareData {
  metricKeys: string[];
  experiments: CompareExperiment[];
  isEffortSweep: boolean;
  // True when experiments are per-effort averages across datasets.
  isAggregate?: boolean;
  // Aggregate mode only: set when the groups don't cover identical dataset sets.
  warning?: string;
}

/** Dataset family names with canonical casing for display titles. */
const DATASET_FAMILY_NAMES: Record<string, string> = {
  bright: "BRIGHT",
  beir: "BEIR",
  lotte: "LoTTe",
  freshstack: "FreshStack",
};

/** "bright/biology" -> "BRIGHT Biology"; "bright/earth_science" -> "BRIGHT Earth Science". */
function datasetTitle(dataset: string): string {
  const [family, ...rest] = dataset.split("/");
  const familyName = DATASET_FAMILY_NAMES[family.toLowerCase()] ?? family;
  const subset = rest
    .join(" ")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return subset ? `${familyName} ${subset}` : familyName;
}

/** Compact subset abbreviations that fit under narrow chart bars. */
const SUBSET_ABBREV: Record<string, string> = {
  biology: "Bio",
  earth: "Earth",
  economics: "Econ",
  psychology: "Psych",
  robotics: "Rob",
  stackoverflow: "SO",
  sustainable: "Sust",
  leetcode: "Leet",
  aops: "AoPS",
  theoremqa: "ThmQA",
};

/** "Earth Science" -> "Earth"; unknown names fall back to the first word. */
function shortSubsetLabel(label: string): string {
  const first = label.split(/\s+/)[0];
  return SUBSET_ABBREV[first.toLowerCase()] ?? first.slice(0, 6);
}

/** Chart title for an aggregate view: "BRIGHT — Average of 5 Subsets". */
function aggregateTitle(experiments: CompareExperiment[]): string {
  const datasets = new Set(
    experiments.flatMap((e) => (e.constituents ?? []).map((c) => c.dataset))
  );
  const families = new Set([...datasets].map((d) => d.split("/")[0].toLowerCase()));
  const n = datasets.size;
  if (families.size === 1) {
    const family = DATASET_FAMILY_NAMES[[...families][0]] ?? [...families][0];
    return `${family} — Average of ${n} Subset${n !== 1 ? "s" : ""}`;
  }
  return `Average of ${n} Dataset${n !== 1 ? "s" : ""}`;
}

function formatMetricName(key: string): string {
  let name = key.replace("avg_", "").replace("_mean", "");
  name = name.replace(/recall_at_(\d+)/g, "Recall@$1");
  name = name.replace(/nDCG_at_(\d+)/g, "nDCG@$1");
  name = name.replace("nDCG_at_k", "nDCG@10");
  name = name.replace(/alpha_ndcg_at_(\d+)/gi, "alpha-nDCG@$1");
  name = name.replace(/coverage_at_(\d+)/g, "Coverage@$1");
  name = name.replace(/success_at_(\d+)/g, "Success@$1");
  name = name.replace("alignment_score", "Alignment");
  name = name.replace("exact_match_accuracy", "Exact Match");
  name = name.replace("officeqa_accuracy", "OfficeQA");
  name = name.replace("query_time", "Avg Time (s)");
  return name;
}

function isTimeMetric(key: string): boolean {
  return key.toLowerCase().includes("time");
}

/** True effort levels; other effort tags (hybrid/vector/bm25) are sweep baselines. */
const CORE_EFFORTS = new Set(["low", "medium", "high"]);

/** Friendly names for baseline tags. */
const BASELINE_LABELS: Record<string, string> = {
  hybrid: "Hybrid Search",
  vector: "Vector Search",
  bm25: "BM25 Search",
};

/** "low" -> "effort=low"; baseline tags get their friendly name ("Hybrid Search"). */
function effortDisplay(effort: string | null | undefined): string {
  if (!effort) return "";
  if (CORE_EFFORTS.has(effort)) return `effort=${effort}`;
  return BASELINE_LABELS[effort] ?? effort;
}

function displayName(exp: CompareExperiment): string {
  if (exp.label) return exp.label;
  // In an effort sweep all runs share dataset + agent; effort is the differentiator.
  if (exp.effort) return effortDisplay(exp.effort);
  return `${exp.dataset} / ${exp.agent_name}`;
}

/* Brand-aligned experiment colors using secondary palette */
const EXP_COLORS = [
  { fg: "var(--color-cyan)", bg: "rgba(1,198,201,0.12)" },
  { fg: "var(--color-lavender)", bg: "rgba(165,144,221,0.12)" },
  { fg: "var(--color-sky)", bg: "rgba(122,214,235,0.12)" },
  { fg: "var(--color-blue-green)", bg: "rgba(1,222,160,0.15)" },
  { fg: "var(--text-muted)", bg: "rgba(221,235,242,0.15)" },
];

/* ═══════════════════════════════════════════════════════════════════════════
   Markdown report export
   ═══════════════════════════════════════════════════════════════════════════ */

type CompareAnalysis = {
  metricAnalysis: Record<
    string,
    { bestIdx: number | null; worstIdx: number | null; spread: number | null }
  >;
  wins: number[];
};

/** Escape pipe characters so user labels can't break markdown table rows. */
function escPipe(s: string): string {
  return s.replace(/\|/g, "\\|");
}

function formatMetricValue(key: string, val: number | null, std?: number | null): string {
  if (val === null) return "—";
  const base = isTimeMetric(key) ? `${val.toFixed(2)}s` : `${(val * 100).toFixed(2)}%`;
  if (std != null && std > 0) {
    return `${base} ±${isTimeMetric(key) ? `${std.toFixed(2)}s` : `${(std * 100).toFixed(2)}%`}`;
  }
  return base;
}

/** Short column header for an experiment in the report (label preferred). */
function reportColLabel(
  exp: { label: string; agent_name: string; effort?: string | null },
  i: number
): string {
  const name = exp.label || (exp.effort ? effortDisplay(exp.effort) : exp.agent_name);
  return `[${i + 1}] ${escPipe(name)}`;
}

/**
 * Build a detailed markdown comparison report from the aggregated compare data,
 * the derived analysis, and (optionally) the per-query comparison. Always
 * surfaces each experiment's user-assigned label prominently.
 */
function buildMarkdownReport(
  data: CompareData,
  analysis: CompareAnalysis | null,
  queries: QueryComparison | null,
  generatedAt: string,
): string {
  const { metricKeys, experiments } = data;
  const lines: string[] = [];
  const L = (s = "") => lines.push(s);

  L(`# Experiment Comparison Report`);
  L();
  L(`_Generated ${generatedAt} · comparing ${experiments.length} experiments_`);
  L();
  if (data.isAggregate) {
    L(`_Metrics are macro-averaged across datasets; ± values show the std of the cross-dataset average across trials._`);
    L();
    if (data.warning) {
      L(`> **Warning:** ${data.warning}`);
      L();
    }
  }

  // ── Experiments ──────────────────────────────────────────────────────────
  L(`## Experiments`);
  L();
  experiments.forEach((exp, i) => {
    L(`### [${i + 1}] ${exp.label || (exp.constituents && exp.effort ? effortDisplay(exp.effort) : `${exp.dataset} / ${exp.agent_name}`)}`);
    L();
    L(`- **Label:** ${exp.label ? exp.label : "_(none assigned)_"}`);
    if (exp.constituents) {
      L(`- **Datasets (${exp.constituents.length}):** ${exp.constituents.map((c) => c.dataset).join(", ")}`);
    } else {
      L(`- **Dataset:** ${exp.dataset}`);
    }
    L(`- **Agent:** \`${exp.agent_name}\``);
    L(`- **Mode:** ${exp.mode}`);
    if (exp.effort) L(`- **Effort:** ${exp.effort}`);
    L(`- **Trials:** ${exp.num_trials}`);
    L(`- **Timestamp:** ${exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "—"}`);
    L(`- **ID:** \`${decodeURIComponent(exp.id)}\``);
    L();
  });

  // ── Metrics table ────────────────────────────────────────────────────────
  L(`## Metrics`);
  L();
  if (metricKeys.length === 0) {
    L(`_No aggregated metrics available._`);
    L();
  } else {
    const twoWay = experiments.length === 2;
    // For a sweep, delta anchors on the actual low/high entries — baselines
    // (e.g. hybrid) sit before "low" in column order.
    const iLow = experiments.findIndex((e) => e.effort === "low");
    const iHigh = experiments.findIndex((e) => e.effort === "high");
    const sweepDelta = data.isEffortSweep && iLow !== -1 && iHigh !== -1;
    const showDelta = twoWay || sweepDelta;
    const deltaLabel = sweepDelta ? "Δ (high − low)" : "Delta";
    const header = ["Metric", ...experiments.map((e, i) => reportColLabel(e, i))];
    if (showDelta) header.push(deltaLabel);
    L(`| ${header.join(" | ")} |`);
    L(`| ${header.map((_, i) => (i === 0 ? ":---" : "---:")).join(" | ")} |`);

    for (const key of metricKeys) {
      const isTime = isTimeMetric(key);
      const best = analysis?.metricAnalysis[key]?.bestIdx ?? null;
      const cells = [escPipe(formatMetricName(key))];

      experiments.forEach((exp, i) => {
        const val = exp.metrics[key];
        let cell = formatMetricValue(key, val, exp.metricsStd?.[key]);
        if (val !== null && i === best && experiments.length > 1) cell = `**${cell}** ⭐`;
        cells.push(cell);
      });

      if (showDelta) {
        const v0 = experiments[sweepDelta ? iLow : 0].metrics[key];
        const v1 = experiments[sweepDelta ? iHigh : 1].metrics[key];
        if (v0 === null || v1 === null) {
          cells.push("—");
        } else {
          const delta = v1 - v0;
          if (Math.abs(delta) < 1e-9) {
            cells.push("0");
          } else {
            const isImprovement = isTime ? delta < 0 : delta > 0;
            const sign = delta > 0 ? "+" : "";
            const formatted = isTime
              ? `${sign}${delta.toFixed(2)}s`
              : `${sign}${(delta * 100).toFixed(2)}%`;
            cells.push(`${formatted} ${isImprovement ? "▲" : "▼"}`);
          }
        }
      }

      L(`| ${cells.join(" | ")} |`);
    }
    L();
    const deltaNote = sweepDelta
      ? "Δ = effort high − effort low (▲ improvement, ▼ regression). Values are mean ± std across trials."
      : twoWay
        ? "Delta = [2] − [1] (▲ improvement, ▼ regression)."
        : "";
    L(`⭐ marks the best value for each metric. ${deltaNote}`);
    L();
  }

  // ── Analysis ─────────────────────────────────────────────────────────────
  if (analysis) {
    L(`## Analysis`);
    L();

    const nonTime = metricKeys.filter((k) => !isTimeMetric(k));
    L(`### Metric Wins (excluding time)`);
    L();
    experiments.forEach((exp, i) => {
      const w = analysis.wins[i];
      const pct = nonTime.length > 0 ? ((w / nonTime.length) * 100).toFixed(0) : "0";
      L(`- **${reportColLabel(exp, i)}:** ${w} of ${nonTime.length} (${pct}%)`);
    });
    L();

    // Speed comparison — 2-way, or low-vs-high effort within a sweep
    // (mirrors the on-page Speed card's delta anchors).
    const iLowT = experiments.findIndex((e) => e.effort === "low");
    const iHighT = experiments.findIndex((e) => e.effort === "high");
    const sweepSpeed = data.isEffortSweep && iLowT !== -1 && iHighT !== -1;
    if (experiments.length === 2 || sweepSpeed) {
      const iA = sweepSpeed ? iLowT : 0;
      const iB = sweepSpeed ? iHighT : 1;
      const timeKey = metricKeys.find((k) => isTimeMetric(k));
      const t0 = timeKey ? experiments[iA].metrics[timeKey] : null;
      const t1 = timeKey ? experiments[iB].metrics[timeKey] : null;
      if (timeKey && t0 != null && t1 != null) {
        const faster = t0 < t1 ? iA : iB;
        const slower = faster === iA ? iB : iA;
        const speedup = ((Math.max(t0, t1) - Math.min(t0, t1)) / Math.max(t0, t1)) * 100;
        if (speedup >= 0.1) {
          L(`### Speed`);
          L();
          L(
            `**${reportColLabel(experiments[faster], faster)}** is **${speedup.toFixed(1)}% faster** than ` +
              `**${reportColLabel(experiments[slower], slower)}** ` +
              `(${Math.min(t0, t1).toFixed(2)}s vs ${Math.max(t0, t1).toFixed(2)}s).`,
          );
          L();
          // Sweeps with 3+ columns: list every column's timing so baselines and
          // mid-tier efforts aren't dropped from the report.
          if (experiments.length > 2) {
            experiments.forEach((exp, i) => {
              const t = exp.metrics[timeKey];
              L(`- ${reportColLabel(exp, i)}: ${t != null ? `${t.toFixed(2)}s` : "—"}`);
            });
            L();
          }
        }
      }
    }

    // Largest differences.
    const diffs = nonTime
      .map((k) => ({ key: k, spread: analysis.metricAnalysis[k]?.spread ?? null }))
      .filter((d) => d.spread !== null && d.spread > 0)
      .sort((a, b) => (b.spread ?? 0) - (a.spread ?? 0))
      .slice(0, 5);
    if (diffs.length > 0) {
      L(`### Largest Differences`);
      L();
      for (const d of diffs) {
        L(`- **${escPipe(formatMetricName(d.key))}:** ${(d.spread! * 100).toFixed(2)}% spread`);
      }
      L();
    }
  }

  // ── Per-query analysis ─────────────────────────────────────────────────────
  if (queries && !queries.warning && queries.rows.length > 0) {
    const { counts, sharedByAll, totalQuestions, mode } = queries;
    const qExps = queries.experiments;

    // Outcome label + sort weight (disagreements first, then all-wrong, then all-correct).
    const outcomeLabel: Record<ComparisonOutcome, string> = {
      mixed: "Disagreement",
      all_wrong: "All wrong",
      all_correct: "All correct",
    };

    /** Compact one-cell status used in the overview table. */
    const cellStatus = (cell: ComparisonCell | null): string => {
      if (!cell) return "—";
      if (mode === "search") return cell.correct ? `${cell.overlap ?? 0} hit` : "0 hit";
      if (cell.is_error) return "err";
      if (cell.correct === null) return "?";
      return cell.correct ? "✓" : "✗";
    };

    L(`## Per-Query Analysis`);
    L();
    L(`Queries aligned by question text (mode: ${mode}).`);
    L();
    L(`- **Comparable** (present in ≥2 experiments): ${counts.comparable}`);
    L(`- **Shared by all:** ${sharedByAll}`);
    L(`- **Total distinct questions:** ${totalQuestions}`);
    L(`- **All correct:** ${counts.allCorrect}`);
    L(`- **All wrong:** ${counts.allWrong}`);
    L(`- **Disagreements:** ${counts.mixed}`);
    L();

    // Comparable rows are already sorted disagreements-first by buildQueryComparison.
    const comparable = queries.rows.filter((r) => r.presentCount >= 2);
    const singleOnly = queries.rows.length - comparable.length;

    if (comparable.length > 0) {
      // ── Overview table — every comparable query at a glance ────────────────
      L(`### Results Overview`);
      L();
      const head = ["#", "Outcome", ...qExps.map((e, i) => reportColLabel(e, i)), "Question"];
      L(`| ${head.join(" | ")} |`);
      L(`| ${head.map((_, i) => (i === 0 ? "---:" : ":---")).join(" | ")} |`);
      comparable.forEach((row, idx) => {
        const statusCells = row.cells.map((c) => cellStatus(c));
        const q = escPipe(row.question.replace(/\s+/g, " ").trim());
        const qShort = q.length > 100 ? `${q.slice(0, 100)}…` : q;
        L(`| ${idx + 1} | ${outcomeLabel[row.outcome]} | ${statusCells.join(" | ")} | ${qShort} |`);
      });
      L();
      if (mode === "search") {
        L(`_Cells show relevant-document hits per experiment._`);
      } else {
        L(`_Cells: ✓ correct · ✗ wrong · ? unscored · err error._`);
      }
      L();

      // ── Full detail for every comparable query ─────────────────────────────
      L(`### Query Details`);
      L();
      comparable.forEach((row, idx) => {
        L(`#### ${idx + 1}. [${outcomeLabel[row.outcome]}] ${row.question}`);
        L();
        if (mode === "ask" && row.ground_truth_answer) {
          L(`**Ground truth:** ${row.ground_truth_answer}`);
          L();
        }
        if (mode === "search" && row.ground_truth_ids) {
          L(
            `**Ground truth IDs (${row.ground_truth_ids.length}):** ${row.ground_truth_ids.join(", ") || "—"}`,
          );
          L();
        }

        row.cells.forEach((cell, i) => {
          const name = reportColLabel(qExps[i], i);
          if (!cell) {
            L(`- ${name}: _not present in this experiment_`);
            return;
          }
          const time = cell.time_taken != null ? ` · ${cell.time_taken.toFixed(2)}s` : "";

          if (mode === "ask") {
            const status = cell.is_error
              ? "error"
              : cell.correct === true
                ? "✓ correct"
                : cell.correct === null
                  ? "? unscored"
                  : "✗ wrong";
            L(`- **${name}** — ${status}${time}`);
            L(`  - Answer: ${cell.system_answer ? cell.system_answer : "_(empty)_"}`);
            if (cell.judge_reasoning) L(`  - Judge: ${cell.judge_reasoning}`);
          } else {
            const status = cell.correct ? "✓ hit" : "✗ miss";
            const gt = new Set(row.ground_truth_ids ?? []);
            const retrieved = cell.retrieved_ids ?? [];
            L(
              `- **${name}** — ${status} · ${cell.overlap ?? 0} relevant of ${cell.num_retrieved ?? retrieved.length} retrieved${time}`,
            );
            if (retrieved.length > 0) {
              const shown = retrieved
                .slice(0, 20)
                .map((id) => (gt.has(id) ? `**${id}**` : id));
              const more = retrieved.length > 20 ? ` … +${retrieved.length - 20} more` : "";
              L(`  - Retrieved: ${shown.join(", ")}${more}`);
            }
            if (cell.searches && cell.searches.length > 0) {
              const plan = cell.searches
                .map((s) => {
                  const q = s.query ? `"${s.query}"` : s.uuid_value ? `uuid:${s.uuid_value}` : "—";
                  const filt = s.filters ? " +filter" : "";
                  const sort = s.sort_property ? ` +sort(${s.sort_property.property_name})` : "";
                  return `${s.collection}:${q}${filt}${sort}`;
                })
                .join("; ");
              L(`  - Search plan (${cell.searches.length}): ${plan}`);
            }
          }
        });
        L();
      });

      if (singleOnly > 0) {
        L(
          `_${singleOnly} additional ${singleOnly === 1 ? "query was" : "queries were"} present in only one experiment and omitted from the comparison._`,
        );
        L();
      }
    }
  }

  L(`---`);
  L(`_Report generated by the Query Agent Benchmarking console._`);
  L();

  return lines.join("\n");
}

/** Trigger a client-side download of a markdown file. */
function downloadMarkdown(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Effort sweep bar charts — hand-rolled SVG, no chart dependency.
   ═══════════════════════════════════════════════════════════════════════════ */

interface ChartBar {
  label: string;
  value: number | null;
  std?: number | null;
  color: string;
}

/* Effort levels map onto the agent spectrum: low (cyan) → high (green).
   Baselines (e.g. hybrid) sit outside the sweep, so they take the baseline
   color: white on dark, blue on light. */
function effortColor(effort: string | null): string {
  if (effort === "high") return "var(--color-green)";
  if (effort === "medium") return "var(--color-blue-green)";
  if (effort === "low") return "var(--color-cyan)";
  return "var(--color-baseline)";
}

function chartSlug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

/** Rasterize a chart card's SVG to a PNG download. Computed styles are inlined
 *  onto a clone first — the SVG relies on CSS variables that don't resolve
 *  outside the page, so a naive serialization would export black-on-black. */
function exportChartPng(card: HTMLDivElement, svg: SVGSVGElement, title: string) {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

  const srcEls = [svg as SVGElement, ...Array.from(svg.querySelectorAll<SVGElement>("*"))];
  const dstEls = [clone as SVGElement, ...Array.from(clone.querySelectorAll<SVGElement>("*"))];
  srcEls.forEach((el, i) => {
    const cs = window.getComputedStyle(el);
    const d = dstEls[i];
    if (el.hasAttribute("fill")) d.setAttribute("fill", cs.fill);
    if (el.hasAttribute("stroke")) d.setAttribute("stroke", cs.stroke);
    if (el.tagName === "text") {
      d.setAttribute("font-family", cs.fontFamily);
      d.setAttribute("font-size", cs.fontSize);
      d.setAttribute("font-weight", cs.fontWeight);
    }
  });

  const vb = svg.viewBox.baseVal;
  const scale = 4;
  const pad = 12;
  const titleH = 24;
  const width = (vb.width + pad * 2) * scale;
  const height = (vb.height + pad * 2 + titleH) * scale;

  const bg = window.getComputedStyle(card).backgroundColor;
  const eyebrow = card.querySelector(".eyebrow");
  const titleColor = eyebrow ? window.getComputedStyle(eyebrow).color : "#8896ab";
  // Cards with a gradient title (marked data-chart-title) get a matching
  // canvas gradient in the export; CSS background-clip text can't rasterize.
  const gradientTitleEl = card.querySelector<HTMLElement>("[data-chart-title]");
  const gradientTitleFont = gradientTitleEl
    ? window.getComputedStyle(gradientTitleEl).fontFamily
    : null;
  const cardStyles = window.getComputedStyle(card);
  const gradFrom = cardStyles.getPropertyValue("--color-cyan").trim() || "#01C6C9";
  const gradTo = cardStyles.getPropertyValue("--color-green").trim() || "#01F57A";

  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    const label = title.toUpperCase();
    if (gradientTitleFont) {
      // Mirror the on-screen gradient title: centered, display font, with a
      // left-to-right cyan → green gradient spanning the text.
      ctx.font = `700 ${13 * scale}px ${gradientTitleFont}`;
      const cx = width / 2;
      const textW = ctx.measureText(label).width;
      const grad = ctx.createLinearGradient(cx - textW / 2, 0, cx + textW / 2, 0);
      grad.addColorStop(0, gradFrom);
      grad.addColorStop(1, gradTo);
      ctx.fillStyle = grad;
      ctx.textAlign = "center";
      ctx.fillText(label, cx, (pad + 11) * scale);
      ctx.textAlign = "left";
    } else {
      ctx.fillStyle = titleColor;
      ctx.font = `600 ${10 * scale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.fillText(label, pad * scale, (pad + 10) * scale);
    }
    ctx.drawImage(img, pad * scale, (pad + titleH) * scale, vb.width * scale, vb.height * scale);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${chartSlug(title)}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, "image/png");
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(clone));
}

/** One bar chart with optional ± std error whiskers. `yMax` is passed in so
 *  sibling charts share a y-scale and stay visually comparable. */
function BarChart({
  title,
  bars,
  isTime,
  yMax,
}: {
  title: string;
  bars: ChartBar[];
  isTime: boolean;
  yMax: number;
}) {
  const W = 280;
  const H = 210;
  const M = { top: 30, right: 10, bottom: 26, left: 46 };
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;

  const y = (v: number) => M.top + innerH - Math.min(Math.max(v, 0) / yMax, 1) * innerH;
  const slot = innerW / Math.max(bars.length, 1);
  const barW = Math.min(slot * 0.55, 44);

  const fmtTick = (v: number) => (isTime ? `${v.toFixed(1)}s` : `${(v * 100).toFixed(0)}%`);
  const fmtVal = (v: number) => (isTime ? `${v.toFixed(2)}s` : `${(v * 100).toFixed(1)}%`);

  const cardRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  return (
    <div ref={cardRef} className="brand-card p-4">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="eyebrow">{title}</div>
        <button
          onClick={() =>
            cardRef.current && svgRef.current && exportChartPng(cardRef.current, svgRef.current, title)
          }
          className="cursor-pointer shrink-0 transition-colors"
          title="Download chart as PNG"
          aria-label={`Download ${title} chart as PNG`}
          style={{ color: "var(--text-muted)", lineHeight: 0, padding: "2px" }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--color-green)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v12m0 0l-4-4m4 4l4-4" />
            <path d="M4 21h16" />
          </svg>
        </button>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
        {[0.25, 0.5, 0.75, 1].map((f) => {
          const v = f * yMax;
          return (
            <g key={f}>
              <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--border-subtle)" strokeWidth={1} />
              <text
                x={M.left - 6}
                y={y(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
              >
                {fmtTick(v)}
              </text>
            </g>
          );
        })}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--border-default)" strokeWidth={1} />

        {bars.map((b, i) => {
          const cx = M.left + slot * i + slot / 2;
          if (b.value == null) {
            return (
              <text key={i} x={cx} y={y(0) - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
                --
              </text>
            );
          }
          const std = b.std ?? 0;
          const top = y(b.value);
          const whiskerTop = y(b.value + std);
          const whiskerBot = y(Math.max(b.value - std, 0));
          return (
            <g key={i}>
              <rect x={cx - barW / 2} y={top} width={barW} height={y(0) - top} rx={3} fill={b.color} opacity={0.85} />
              {std > 0 && (
                <g stroke="var(--text-secondary)" strokeWidth={1.2}>
                  <line x1={cx} x2={cx} y1={whiskerTop} y2={whiskerBot} />
                  <line x1={cx - 5} x2={cx + 5} y1={whiskerTop} y2={whiskerTop} />
                  <line x1={cx - 5} x2={cx + 5} y1={whiskerBot} y2={whiskerBot} />
                </g>
              )}
              <text
                x={cx}
                y={Math.min(top, whiskerTop) - 6}
                textAnchor="middle"
                fontSize={9.5}
                fontWeight={600}
                fill="var(--text-primary)"
                fontFamily="var(--font-mono)"
              >
                {fmtVal(b.value)}
              </text>
              <text x={cx} y={H - 8} textAnchor="middle" fontSize={10} fill="var(--text-secondary)">
                {b.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Move one element of `arr` from index `from` to index `to`. */
function moveItem<T>(arr: T[], from: number, to: number): T[] {
  const next = [...arr];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

/** All quality metrics in one plot: metric groups on the x-axis, one bar per
 *  experiment (colored by effort tier) within each group. Time metrics are
 *  excluded — they'd need a second axis. Chips in the header toggle which
 *  metrics are plotted; drag a chip to reorder the groups left-to-right. */
function GroupedBarChart({
  title,
  metricKeys,
  selected,
  onToggle,
  onReorder,
  experiments,
}: {
  title: string;
  metricKeys: string[];
  selected: Set<string>;
  onToggle: (key: string) => void;
  onReorder: (keys: string[]) => void;
  experiments: CompareExperiment[];
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  // Index of the chip being dragged; live-reordered on drag-enter so the
  // chart previews the new order mid-drag.
  const dragIdx = useRef<number | null>(null);
  const [draggingKey, setDraggingKey] = useState<string | null>(null);

  const activeKeys = metricKeys.filter((k) => selected.has(k));

  const barW = 22;
  const barGap = 5;
  const groupPad = 22;
  const groupW = experiments.length * (barW + barGap) - barGap + groupPad;
  const M = { top: 40, right: 12, bottom: 26, left: 46 };
  const W = M.left + M.right + activeKeys.length * groupW;
  const H = 250;
  const innerH = H - M.top - M.bottom;

  const allValues = activeKeys.flatMap((k) =>
    experiments.flatMap((e) => {
      const v = e.metrics[k];
      return v !== null && v !== undefined ? [v + (e.metricsStd?.[k] ?? 0)] : [];
    })
  );
  const hasData = allValues.length > 0 && Math.max(...allValues) > 0;
  // Fixed 0–100% axis so effort sweeps are comparable across charts and runs.
  const yMax = 1;

  const y = (v: number) => M.top + innerH - Math.min(Math.max(v, 0) / yMax, 1) * innerH;

  // Legend inside the SVG so PNG export includes it.
  const legendItems = experiments.map((e) => ({
    label: effortDisplay(e.effort) || e.agent_name,
    color: effortColor(e.effort),
  }));
  let legendX = M.left;

  return (
    <div ref={cardRef} className="brand-card p-4" style={{ gridColumn: "1 / -1" }}>
      {/* Centered gradient title. Built from the theme color vars (not the raw
          gradient token) so light mode gets its deepened, legible hues. */}
      <div
        data-chart-title
        className="text-center text-lg font-bold uppercase mb-2"
        style={{
          fontFamily: "var(--font-display)",
          letterSpacing: "0.08em",
          background: "linear-gradient(45deg, var(--color-cyan), var(--color-green))",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          color: "transparent",
        }}
      >
        {title}
      </div>
      <div className="flex items-center justify-center gap-2 mb-2 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex gap-1 flex-wrap">
            {metricKeys.map((k, i) => {
              const active = selected.has(k);
              return (
                <button
                  key={k}
                  onClick={() => onToggle(k)}
                  className="brand-btn-secondary"
                  title={`${active ? `Remove ${formatMetricName(k)} from chart` : `Add ${formatMetricName(k)} to chart`} · drag to reorder`}
                  aria-pressed={active}
                  draggable
                  onDragStart={(e) => {
                    dragIdx.current = i;
                    setDraggingKey(k);
                    e.dataTransfer.effectAllowed = "move";
                  }}
                  onDragOver={(e) => e.preventDefault()}
                  onDragEnter={() => {
                    const from = dragIdx.current;
                    if (from !== null && from !== i) {
                      onReorder(moveItem(metricKeys, from, i));
                      dragIdx.current = i;
                    }
                  }}
                  onDragEnd={() => {
                    dragIdx.current = null;
                    setDraggingKey(null);
                  }}
                  style={{
                    cursor: draggingKey === k ? "grabbing" : "grab",
                    ...(active
                      ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" }
                      : { opacity: 0.6 }),
                    ...(draggingKey === k ? { opacity: 0.4 } : {}),
                  }}
                >
                  {formatMetricName(k)}
                </button>
              );
            })}
          </div>
          <button
            onClick={() =>
              cardRef.current && svgRef.current && exportChartPng(cardRef.current, svgRef.current, title)
            }
            className="cursor-pointer shrink-0 transition-colors"
            title="Download chart as PNG"
            aria-label={`Download ${title} chart as PNG`}
            style={{ color: "var(--text-muted)", lineHeight: 0, padding: "2px" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--color-green)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3v12m0 0l-4-4m4 4l4-4" />
              <path d="M4 21h16" />
            </svg>
          </button>
        </div>
      </div>
      {!hasData && (
        <div className="py-8 text-center text-sm" style={{ color: "var(--text-muted)" }}>
          Select at least one metric to plot.
        </div>
      )}
      {hasData && (
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", maxWidth: W * 2 }}>
        {legendItems.map((item, i) => {
          const x = legendX;
          legendX += item.label.length * 5.4 + 26;
          return (
            <g key={i}>
              <rect x={x} y={6} width={9} height={9} rx={2} fill={item.color} opacity={0.85} />
              <text x={x + 13} y={14} fontSize={9} fill="var(--text-secondary)" fontFamily="var(--font-mono)">
                {item.label}
              </text>
            </g>
          );
        })}

        {[0.25, 0.5, 0.75, 1].map((f) => {
          const v = f * yMax;
          return (
            <g key={f}>
              <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--border-subtle)" strokeWidth={1} />
              <text
                x={M.left - 6}
                y={y(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
              >
                {`${(v * 100).toFixed(0)}%`}
              </text>
            </g>
          );
        })}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--border-default)" strokeWidth={1} />

        {activeKeys.map((k, gi) => {
          const groupX = M.left + groupW * gi + groupPad / 2;
          return (
            <g key={k}>
              <text
                x={groupX + (groupW - groupPad) / 2}
                y={H - 8}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-secondary)"
              >
                {formatMetricName(k)}
              </text>
              {experiments.map((e, bi) => {
                const v = e.metrics[k];
                const cx = groupX + bi * (barW + barGap) + barW / 2;
                if (v === null || v === undefined) {
                  return (
                    <text key={e.id} x={cx} y={y(0) - 6} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
                      --
                    </text>
                  );
                }
                const std = e.metricsStd?.[k] ?? 0;
                const top = y(v);
                const whiskerTop = y(v + std);
                const whiskerBot = y(Math.max(v - std, 0));
                return (
                  <g key={e.id}>
                    <rect x={cx - barW / 2} y={top} width={barW} height={y(0) - top} rx={3} fill={effortColor(e.effort)} opacity={0.85}>
                      <title>{`${effortDisplay(e.effort) || e.agent_name} — ${formatMetricName(k)}: ${(v * 100).toFixed(1)}%${std > 0 ? ` ±${(std * 100).toFixed(1)}%` : ""}`}</title>
                    </rect>
                    {std > 0 && (
                      <g stroke="var(--text-secondary)" strokeWidth={1.2}>
                        <line x1={cx} x2={cx} y1={whiskerTop} y2={whiskerBot} />
                        <line x1={cx - 4} x2={cx + 4} y1={whiskerTop} y2={whiskerTop} />
                        <line x1={cx - 4} x2={cx + 4} y1={whiskerBot} y2={whiskerBot} />
                      </g>
                    )}
                    <text
                      x={cx}
                      y={Math.min(top, whiskerTop) - 5}
                      textAnchor="middle"
                      fontSize={8}
                      fontWeight={600}
                      fill="var(--text-primary)"
                      fontFamily="var(--font-mono)"
                    >
                      {(v * 100).toFixed(1)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      )}
    </div>
  );
}

function EffortChartsSection({ data }: { data: CompareData }) {
  const { metricKeys, experiments } = data;
  const qualityMetricKeys = metricKeys.filter((k) => !isTimeMetric(k));
  const [metricKey, setMetricKey] = useState<string>(
    () => metricKeys.find((k) => !isTimeMetric(k)) ?? metricKeys[0] ?? ""
  );
  // Metrics included in the grouped "all metrics" chart (all quality metrics by default).
  const [groupedMetrics, setGroupedMetrics] = useState<Set<string>>(
    () => new Set(qualityMetricKeys)
  );
  // Left-to-right order of metric groups in the grouped chart (drag chips to change).
  const [groupedOrder, setGroupedOrder] = useState<string[]>(() => [...qualityMetricKeys]);
  const toggleGroupedMetric = (k: string) =>
    setGroupedMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  if (!metricKey) return null;
  const isTime = isTimeMetric(metricKey);

  const meanBars: ChartBar[] = experiments.map((e) => ({
    label: e.effort ?? e.agent_name,
    value: e.metrics[metricKey],
    std: e.metricsStd?.[metricKey],
    color: effortColor(e.effort),
  }));

  const trialCharts = experiments.map((e) => ({
    exp: e,
    bars: (e.metricsTrials?.[metricKey] ?? [])
      .filter((t) => t.value !== null)
      .map((t) => ({
        label: t.label ? shortSubsetLabel(t.label) : `T${t.trial}`,
        value: t.value,
        color: effortColor(e.effort),
      })) as ChartBar[],
  }));

  // Percent metrics use a fixed 0–100% axis so charts are comparable across
  // metrics and runs; time metrics have no natural ceiling, so they scale.
  const allValues = [
    ...meanBars.flatMap((b) => (b.value !== null ? [b.value + (b.std ?? 0)] : [])),
    ...trialCharts.flatMap((c) => c.bars.map((b) => b.value ?? 0)),
  ];
  if (allValues.length === 0 || Math.max(...allValues) <= 0) return null;
  const yMax = isTime ? Math.max(...allValues) * 1.15 : 1;

  const hasTrialCharts = trialCharts.some((c) => c.bars.length > 1);

  return (
    <section className="mb-10">
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <h2 className="text-lg font-bold" style={{ fontFamily: "var(--font-display)" }}>
          Charts
        </h2>
        <div className="flex gap-1 flex-wrap">
          {metricKeys.map((k) => (
            <button
              key={k}
              onClick={() => setMetricKey(k)}
              className="brand-btn-secondary"
              style={
                metricKey === k
                  ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" }
                  : {}
              }
            >
              {formatMetricName(k)}
            </button>
          ))}
        </div>
      </div>
      {qualityMetricKeys.length > 1 && (
        <div className="mb-4">
          <GroupedBarChart
            title={data.isAggregate ? aggregateTitle(experiments) : datasetTitle(experiments[0].dataset)}
            metricKeys={groupedOrder}
            selected={groupedMetrics}
            onToggle={toggleGroupedMetric}
            onReorder={setGroupedOrder}
            experiments={experiments}
          />
        </div>
      )}
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        <BarChart
          title={`${formatMetricName(metricKey)} — mean ± std`}
          bars={meanBars}
          isTime={isTime}
          yMax={yMax}
        />
        {hasTrialCharts &&
          trialCharts.map((c) => (
            <BarChart
              key={c.exp.id}
              title={`${effortDisplay(c.exp.effort) || "?"} — ${data.isAggregate ? "per dataset" : "per trial"}`}
              bars={c.bars}
              isTime={isTime}
              yMax={yMax}
            />
          ))}
      </div>
    </section>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>Loading comparison...</div>
      }
    >
      <ComparePageInner />
    </Suspense>
  );
}

function ComparePageInner() {
  const searchParams = useSearchParams();
  const ids = searchParams.get("ids") ?? "";
  const aggregate = searchParams.get("aggregate") === "1";
  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  // Metric rows expanded to show their per-trial breakdown.
  const [expandedMetrics, setExpandedMetrics] = useState<Set<string>>(new Set());

  const toggleMetric = (key: string) =>
    setExpandedMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  useEffect(() => {
    if (!ids) {
      setError("No experiment IDs provided.");
      setLoading(false);
      return;
    }
    fetch(`/api/compare?ids=${ids}${aggregate ? "&aggregate=1" : ""}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load comparison data");
        return r.json();
      })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [ids, aggregate]);

  const analysis = useMemo(() => {
    if (!data) return null;
    const { metricKeys, experiments } = data;
    const metricAnalysis: Record<
      string,
      { bestIdx: number | null; worstIdx: number | null; spread: number | null }
    > = {};

    for (const key of metricKeys) {
      const values = experiments.map((e) => e.metrics[key]);
      const numeric = values
        .map((v, i) => (v !== null ? { v, i } : null))
        .filter(Boolean) as { v: number; i: number }[];

      if (numeric.length < 2) {
        metricAnalysis[key] = { bestIdx: null, worstIdx: null, spread: null };
        continue;
      }

      const isTime = isTimeMetric(key);
      numeric.sort((a, b) => (isTime ? a.v - b.v : b.v - a.v));
      metricAnalysis[key] = {
        bestIdx: numeric[0].i,
        worstIdx: numeric[numeric.length - 1].i,
        spread: Math.abs(numeric[0].v - numeric[numeric.length - 1].v),
      };
    }

    const wins = experiments.map(() => 0);
    for (const key of metricKeys) {
      if (isTimeMetric(key)) continue;
      const best = metricAnalysis[key].bestIdx;
      if (best !== null) wins[best]++;
    }

    return { metricAnalysis, wins };
  }, [data]);

  const handleExport = async () => {
    if (!data || exporting) return;
    setExporting(true);
    try {
      // Fetch per-query data so the report is complete regardless of whether
      // the user expanded the per-query section. It's optional — if it fails
      // we still export the metrics + analysis.
      let queries: QueryComparison | null = null;
      if (!data.isAggregate) {
        try {
          const r = await fetch(`/api/compare/queries?ids=${ids}`);
          if (r.ok) queries = (await r.json()) as QueryComparison;
        } catch {
          // per-query analysis is best-effort
        }
      }

      const now = new Date();
      const md = buildMarkdownReport(data, analysis, queries, now.toLocaleString());
      // Unique filename: report kind + dataset (when shared) + date-and-time
      // stamp so multiple exports on the same day don't collide.
      const stamp = now.toISOString().slice(0, 19).replace(/[T:]/g, "-");
      const kind = data.isEffortSweep ? "effort-sweep" : "experiment-comparison";
      const datasets = new Set(data.experiments.map((e) => e.dataset));
      const scope = data.isAggregate
        ? "averaged"
        : datasets.size === 1
          ? chartSlug(data.experiments[0].dataset)
          : "";
      downloadMarkdown(md, [kind, scope, stamp].filter(Boolean).join("-") + ".md");
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="py-20 text-center" style={{ color: "var(--text-muted)" }}>
        <div className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3" style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }} />
        <p>Loading comparison...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-20 text-center">
        <p className="mb-4" style={{ color: "var(--color-coral)" }}>{error ?? "Unknown error"}</p>
        <a href="/results" className="text-sm font-semibold" style={{ color: "var(--color-green)" }}>
          &larr; Back to dashboard
        </a>
      </div>
    );
  }

  const { metricKeys, experiments } = data;

  // Delta anchors: for a sweep, compare the actual low/high entries (baselines
  // like hybrid sit before "low" in column order, so first ≠ low).
  const iLow = experiments.findIndex((e) => e.effort === "low");
  const iHigh = experiments.findIndex((e) => e.effort === "high");
  const sweepDelta = data.isEffortSweep && iLow !== -1 && iHigh !== -1;
  const showDelta = experiments.length === 2 || sweepDelta;
  const iDeltaA = sweepDelta ? iLow : 0;
  const iDeltaB = sweepDelta ? iHigh : 1;

  return (
    <div>
      {/* ── Breadcrumb ───────────────────────────────────────────────────── */}
      <nav className="flex items-center gap-2 mb-8 text-sm" style={{ color: "var(--text-muted)" }}>
        <a href="/results" className="transition-colors hover:underline" style={{ color: "var(--color-green)" }}>
          Experiments
        </a>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Compare</span>
      </nav>

      <div className="flex items-start justify-between gap-4 mb-6 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
            {data.isAggregate
              ? data.isEffortSweep
                ? "Effort Sweep — Averaged"
                : "Averaged Comparison"
              : data.isEffortSweep
                ? "Effort Sweep"
                : "Compare Experiments"}
          </h1>
          {data.isAggregate && (
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
              Metrics macro-averaged across{" "}
              {new Set(experiments.flatMap((e) => (e.constituents ?? []).map((c) => c.dataset))).size}{" "}
              datasets · error bars show std across trials
            </p>
          )}
          {!data.isAggregate && data.isEffortSweep && (
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
              Same benchmark at increasing search-mode compute effort ({experiments.map((e) => e.effort).join(" → ")})
            </p>
          )}
        </div>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="brand-btn-primary"
          style={{ padding: "8px 18px", opacity: exporting ? 0.6 : 1 }}
          title="Download a detailed markdown comparison report"
        >
          {exporting ? "Exporting…" : "↓ Export Report"}
        </button>
      </div>

      {/* ── Unbalanced-aggregate warning ──────────────────────────────────── */}
      {data.isAggregate && data.warning && (
        <div
          className="brand-card p-4 mb-6 text-sm"
          style={{ color: "var(--color-warn)", background: "rgba(255,199,44,0.08)" }}
        >
          {data.warning}
        </div>
      )}

      {/* ── Experiment legend ─────────────────────────────────────────────── */}
      <div
        className="grid gap-4 mb-8"
        style={{ gridTemplateColumns: `repeat(${Math.min(experiments.length, 3)}, 1fr)` }}
      >
        {experiments.map((exp, i) => {
          const c = EXP_COLORS[i % EXP_COLORS.length];
          return (
            <div
              key={exp.id}
              className="brand-card p-5"
              style={{ borderLeft: `3px solid ${c.fg}` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="brand-badge"
                  style={{ background: c.bg, color: c.fg, fontWeight: 700 }}
                >
                  {i + 1}
                </span>
                <span className="font-semibold text-sm" style={{ fontFamily: "var(--font-display)" }}>
                  {displayName(exp)}
                </span>
                <EffortBadge effort={exp.effort} />
              </div>
              {exp.constituents ? (
                <>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                    Average of {exp.constituents.length} experiments &middot;{" "}
                    <span style={{ fontFamily: "var(--font-mono)" }}>{exp.agent_name}</span>
                  </p>
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    {exp.constituents.map((c) => datasetTitle(c.dataset)).join(" · ")}
                  </p>
                </>
              ) : (
                <>
                  {exp.label && (
                    <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {exp.dataset} &middot;{" "}
                      <span style={{ fontFamily: "var(--font-mono)" }}>{exp.agent_name}</span>
                    </p>
                  )}
                  {!exp.label && (
                    <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                      Agent: <span style={{ fontFamily: "var(--font-mono)" }}>{exp.agent_name}</span>
                    </p>
                  )}
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    {exp.num_trials} trials &middot;{" "}
                    {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
                  </p>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Metrics comparison table ─────────────────────────────────────── */}
      <section className="mb-10">
        <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
          Metrics
        </h2>
        <div className="brand-card overflow-x-auto">
          <table className="w-full text-sm text-left brand-table">
            <thead>
              <tr>
                <th>Metric</th>
                {experiments.map((exp, i) => {
                  const c = EXP_COLORS[i % EXP_COLORS.length];
                  return (
                    <th key={exp.id} className="text-right">
                      <span className="brand-badge mr-1" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                        {i + 1}
                      </span>
                      {data.isEffortSweep ? (
                        <EffortBadge effort={exp.effort} />
                      ) : (
                        exp.label || exp.agent_name
                      )}
                    </th>
                  );
                })}
                {showDelta && (
                  <th className="text-right">{sweepDelta ? "Δ high−low" : "Delta"}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {metricKeys.map((key) => {
                const isTime = isTimeMetric(key);
                const best = analysis?.metricAnalysis[key].bestIdx ?? null;
                const hasTrials = experiments.some(
                  (e) => (e.metricsTrials?.[key] ?? []).filter((t) => t.value !== null).length > 1
                );
                const open = expandedMetrics.has(key);

                return (
                  <Fragment key={key}>
                  <tr
                    onClick={hasTrials ? () => toggleMetric(key) : undefined}
                    style={hasTrials ? { cursor: "pointer" } : undefined}
                    title={hasTrials ? "Click to show per-trial results" : undefined}
                  >
                    <td className="font-semibold">
                      {hasTrials && (
                        <span
                          className="mr-1.5 inline-block"
                          style={{
                            transition: "transform 0.15s",
                            transform: open ? "rotate(90deg)" : "none",
                            color: "var(--text-muted)",
                            fontSize: "0.6rem",
                          }}
                        >
                          ▶
                        </span>
                      )}
                      {formatMetricName(key)}
                    </td>
                    {experiments.map((exp, i) => {
                      const val = exp.metrics[key];
                      if (val === null) {
                        return (
                          <td key={i} className="text-right" style={{ color: "var(--text-muted)" }}>
                            --
                          </td>
                        );
                      }
                      const isBest = i === best && experiments.length > 1;
                      const formatted = isTime ? val.toFixed(2) : (val * 100).toFixed(2) + "%";
                      const std = exp.metricsStd?.[key];
                      return (
                        <td
                          key={i}
                          className="text-right"
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontWeight: isBest ? 700 : 400,
                            color: isBest ? "var(--color-green)" : "var(--text-primary)",
                          }}
                        >
                          {formatted}
                          {std != null && std > 0 && (
                            <span style={{ color: "var(--text-muted)", fontWeight: 400, fontSize: "0.7rem" }}>
                              {" "}±{isTime ? std.toFixed(2) : (std * 100).toFixed(1) + "%"}
                            </span>
                          )}
                        </td>
                      );
                    })}
                    {showDelta &&
                      (() => {
                        const v0 = experiments[iDeltaA].metrics[key];
                        const v1 = experiments[iDeltaB].metrics[key];
                        if (v0 === null || v1 === null) {
                          return (
                            <td className="text-right" style={{ color: "var(--text-muted)" }}>
                              --
                            </td>
                          );
                        }
                        const delta = v1 - v0;
                        if (Math.abs(delta) < 1e-9) {
                          return (
                            <td
                              className="text-right"
                              style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}
                            >
                              0
                            </td>
                          );
                        }
                        const isImprovement = isTime ? delta < 0 : delta > 0;
                        const sign = delta > 0 ? "+" : "";
                        const formatted = isTime
                          ? `${sign}${delta.toFixed(2)}s`
                          : `${sign}${(delta * 100).toFixed(2)}%`;
                        return (
                          <td
                            className="text-right"
                            style={{
                              fontFamily: "var(--font-mono)",
                              fontWeight: 600,
                              color: isImprovement ? "var(--color-green)" : "var(--color-coral)",
                            }}
                          >
                            {formatted}
                          </td>
                        );
                      })()}
                  </tr>
                  {open && (
                    <tr style={{ background: "var(--bg-surface)" }}>
                      <td className="text-xs align-top" style={{ color: "var(--text-muted)" }}>
                        {data.isAggregate ? "per-dataset" : "per-trial"}
                      </td>
                      {experiments.map((exp, i) => {
                        const trials = (exp.metricsTrials?.[key] ?? []).filter(
                          (t) => t.value !== null
                        );
                        return (
                          <td key={i} className="text-right align-top">
                            {trials.length === 0 ? (
                              <span style={{ color: "var(--text-muted)" }}>--</span>
                            ) : (
                              <div className="space-y-0.5 py-1">
                                {trials.map((t) => (
                                  <div
                                    key={t.trial}
                                    className="text-xs"
                                    style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}
                                  >
                                    <span style={{ color: "var(--text-muted)" }}>{t.label ?? `T${t.trial}`}</span>{" "}
                                    {isTime ? `${t.value!.toFixed(2)}s` : `${(t.value! * 100).toFixed(2)}%`}
                                  </div>
                                ))}
                              </div>
                            )}
                          </td>
                        );
                      })}
                      {showDelta && <td />}
                    </tr>
                  )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Effort sweep charts. Keyed by the experiment set so metric-selection
          state resets when navigation swaps in different data. ─────────────── */}
      {data.isEffortSweep && (
        <EffortChartsSection key={experiments.map((e) => e.id).join("|")} data={data} />
      )}

      {/* ── Analysis section ──────────────────────────────────────────────── */}
      {analysis && (
        <section>
          <h2 className="text-lg font-bold mb-4" style={{ fontFamily: "var(--font-display)" }}>
            Analysis
          </h2>

          {/* Win counts */}
          <div className="brand-card p-6 mb-6">
            <h3 className="eyebrow mb-4">Metric Wins (excluding time)</h3>
            <div className="flex gap-6">
              {experiments.map((exp, i) => {
                const c = EXP_COLORS[i % EXP_COLORS.length];
                const total = metricKeys.filter((k) => !isTimeMetric(k)).length;
                const winPct = total > 0 ? (analysis.wins[i] / total) * 100 : 0;
                return (
                  <div key={exp.id} className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className="brand-badge"
                        style={{ background: c.bg, color: c.fg, fontWeight: 700 }}
                      >
                        {i + 1}
                      </span>
                      <span className="text-sm font-semibold">{displayName(exp)}</span>
                    </div>
                    <div className="text-3xl font-bold" style={{ fontFamily: "var(--font-display)", color: c.fg }}>
                      {analysis.wins[i]}
                    </div>
                    <div className="w-full h-2 rounded-full mt-2" style={{ background: "var(--border-subtle)" }}>
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{ width: `${winPct}%`, background: c.fg }}
                      />
                    </div>
                    <div className="text-xs mt-1" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {winPct.toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Speed comparison (2-way, or low-vs-high effort in a sweep) */}
          {showDelta &&
            (() => {
              const timeKey = metricKeys.find((k) => isTimeMetric(k));
              if (!timeKey) return null;
              const t0 = experiments[iDeltaA].metrics[timeKey];
              const t1 = experiments[iDeltaB].metrics[timeKey];
              if (t0 === null || t1 === null) return null;

              const faster = t0 < t1 ? iDeltaA : iDeltaB;
              const slower = faster === iDeltaA ? iDeltaB : iDeltaA;
              const speedup =
                ((Math.max(t0, t1) - Math.min(t0, t1)) / Math.max(t0, t1)) * 100;

              if (speedup < 0.1) return null;

              const cFaster = EXP_COLORS[faster % EXP_COLORS.length];
              const cSlower = EXP_COLORS[slower % EXP_COLORS.length];

              return (
                <div className="brand-card p-6 mb-6">
                  <h3 className="eyebrow mb-3">Speed</h3>
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    <span className="font-bold" style={{ color: cFaster.fg }}>
                      [{faster + 1}] {displayName(experiments[faster])}
                    </span>{" "}
                    is{" "}
                    <span className="font-bold" style={{ color: "var(--color-green)" }}>
                      {speedup.toFixed(1)}% faster
                    </span>{" "}
                    than{" "}
                    <span className="font-bold" style={{ color: cSlower.fg }}>
                      [{slower + 1}] {displayName(experiments[slower])}
                    </span>{" "}
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      ({Math.min(t0, t1).toFixed(2)}s vs {Math.max(t0, t1).toFixed(2)}s)
                    </span>
                  </p>
                </div>
              );
            })()}

          {/* Biggest differences */}
          {(() => {
            const diffs = metricKeys
              .filter((k) => !isTimeMetric(k))
              .map((k) => ({
                key: k,
                spread: analysis.metricAnalysis[k].spread,
              }))
              .filter((d) => d.spread !== null && d.spread > 0)
              .sort((a, b) => (b.spread ?? 0) - (a.spread ?? 0));

            if (diffs.length === 0) return null;

            return (
              <div className="brand-card p-6">
                <h3 className="eyebrow mb-4">Largest Differences</h3>
                <div className="space-y-3">
                  {diffs.slice(0, 5).map((d) => (
                    <div key={d.key} className="flex items-center gap-3">
                      <span className="text-sm font-semibold w-40">{formatMetricName(d.key)}</span>
                      <div className="flex-1 h-2 rounded-full" style={{ background: "var(--border-subtle)" }}>
                        <div
                          className="h-2 rounded-full"
                          style={{
                            width: `${Math.min((d.spread! / 0.5) * 100, 100)}%`,
                            background: "var(--gradient-agent)",
                          }}
                        />
                      </div>
                      <span
                        className="text-xs w-20 text-right"
                        style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}
                      >
                        {(d.spread! * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </section>
      )}

      {/* ── Per-query analysis (skipped for aggregates: queries differ per dataset) ── */}
      {!data.isAggregate && <PerQuerySection ids={ids} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Per-query analysis: what each experiment predicted for individual queries.
   Loaded on demand (trial files are large) and aligned by question text.
   ═══════════════════════════════════════════════════════════════════════════ */

type QueryFilter = "all" | "mixed" | "all_correct" | "all_wrong";

const OUTCOME_META: Record<
  ComparisonOutcome,
  { label: string; fg: string; bg: string }
> = {
  mixed: { label: "Disagreement", fg: "var(--color-coral)", bg: "rgba(255,79,94,0.12)" },
  all_correct: { label: "All correct", fg: "var(--color-green)", bg: "rgba(1,245,122,0.15)" },
  all_wrong: { label: "All wrong", fg: "var(--color-warn)", bg: "rgba(255,199,44,0.15)" },
};

function PerQuerySection({ ids }: { ids: string }) {
  const [data, setData] = useState<QueryComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const [filter, setFilter] = useState<QueryFilter>("all");
  const [search, setSearch] = useState("");

  const load = () => {
    setLoading(true);
    setError(null);
    fetch(`/api/compare/queries?ids=${ids}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load per-query comparison");
        return r.json();
      })
      .then((d: QueryComparison) => {
        setData(d);
        setLoaded(true);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  };

  const filteredRows = useMemo(() => {
    if (!data) return [];
    const term = search.trim().toLowerCase();
    return data.rows.filter((row) => {
      if (row.presentCount < 2) return false; // only comparable rows
      if (filter !== "all" && row.outcome !== filter) return false;
      if (term && !row.question.toLowerCase().includes(term)) return false;
      return true;
    });
  }, [data, filter, search]);

  return (
    <section className="mt-10">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h2 className="text-lg font-bold" style={{ fontFamily: "var(--font-display)" }}>
          Per-Query Analysis
        </h2>
        {!loaded && !loading && (
          <button onClick={load} className="brand-btn-primary" style={{ padding: "6px 16px" }}>
            Load per-query analysis
          </button>
        )}
      </div>

      {!loaded && !loading && (
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Compare what each experiment predicted for individual queries, aligned by question text.
          Surfaces disagreements where one experiment succeeds and another fails.
        </p>
      )}

      {loading && (
        <div className="py-10 text-center" style={{ color: "var(--text-muted)" }}>
          <div
            className="inline-block w-6 h-6 border-2 border-t-transparent rounded-full animate-spin mb-3"
            style={{ borderColor: "var(--color-green)", borderTopColor: "transparent" }}
          />
          <p>Loading per-query data...</p>
        </div>
      )}

      {error && (
        <p className="text-sm" style={{ color: "var(--color-coral)" }}>
          {error}
        </p>
      )}

      {data && data.warning && (
        <div
          className="brand-card p-4 mb-4 text-sm"
          style={{ color: "var(--color-warn)", background: "rgba(255,199,44,0.08)" }}
        >
          {data.warning}
        </div>
      )}

      {data && !data.warning && data.rows.length === 0 && (
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          No overlapping queries found across these experiments (they may use different datasets).
        </p>
      )}

      {data && !data.warning && data.rows.length > 0 && (
        <PerQueryBody
          data={data}
          filteredRows={filteredRows}
          filter={filter}
          setFilter={setFilter}
          search={search}
          setSearch={setSearch}
        />
      )}
    </section>
  );
}

function PerQueryBody({
  data,
  filteredRows,
  filter,
  setFilter,
  search,
  setSearch,
}: {
  data: QueryComparison;
  filteredRows: QueryComparisonRow[];
  filter: QueryFilter;
  setFilter: (f: QueryFilter) => void;
  search: string;
  setSearch: (s: string) => void;
}) {
  const { experiments, counts, sharedByAll, totalQuestions, mode } = data;

  const summary: { key: QueryFilter; label: string; value: number; fg: string }[] = [
    { key: "mixed", label: "Disagreements", value: counts.mixed, fg: "var(--color-coral)" },
    { key: "all_correct", label: "All correct", value: counts.allCorrect, fg: "var(--color-green)" },
    { key: "all_wrong", label: "All wrong", value: counts.allWrong, fg: "var(--color-warn)" },
  ];

  return (
    <div>
      {/* Summary strip */}
      <div className="grid gap-4 mb-5" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="brand-card p-4">
          <div className="eyebrow mb-1">Comparable</div>
          <div className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)" }}>
            {counts.comparable}
          </div>
          <div className="text-xs mt-1" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {sharedByAll} in all · {totalQuestions} total
          </div>
        </div>
        {summary.map((s) => (
          <button
            key={s.key}
            onClick={() => setFilter(filter === s.key ? "all" : s.key)}
            className="brand-card p-4 text-left transition-all"
            style={{
              cursor: "pointer",
              outline: filter === s.key ? `2px solid ${s.fg}` : "none",
            }}
          >
            <div className="eyebrow mb-1">{s.label}</div>
            <div className="text-2xl font-bold" style={{ fontFamily: "var(--font-display)", color: s.fg }}>
              {s.value}
            </div>
            <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
              {filter === s.key ? "filtering" : "click to filter"}
            </div>
          </button>
        ))}
      </div>

      {/* Controls */}
      <div
        className="flex gap-1.5 items-center flex-wrap rounded-lg px-4 py-3 mb-4"
        style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)" }}
      >
        <span className="eyebrow mr-2">Filter</span>
        {(["all", "mixed", "all_correct", "all_wrong"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className="brand-btn-secondary"
            style={
              filter === f
                ? { background: "var(--color-green)", color: "var(--color-navy)", borderColor: "var(--color-green)" }
                : {}
            }
          >
            {f === "all" ? "all" : f === "mixed" ? "disagreements" : f === "all_correct" ? "all correct" : "all wrong"}
          </button>
        ))}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search question..."
          className="rounded-md px-2.5 py-1 text-xs ml-2"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-default)",
            color: "var(--text-primary)",
            width: "220px",
          }}
        />
        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {filteredRows.length} of {counts.comparable}
        </span>
      </div>

      {/* Comparison table */}
      <div className="brand-card overflow-x-auto">
        <table className="w-full text-sm text-left brand-table">
          <thead>
            <tr>
              <th style={{ width: "1%" }}></th>
              <th>Question</th>
              {experiments.map((exp, i) => {
                const c = EXP_COLORS[i % EXP_COLORS.length];
                return (
                  <th key={exp.id} className="text-center whitespace-nowrap">
                    <span className="brand-badge mr-1" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                      {i + 1}
                    </span>
                    {exp.label || (exp.effort ? effortDisplay(exp.effort) : exp.agent_name)}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <ComparisonRowView key={row.question} row={row} mode={mode} experiments={experiments} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonRowView({
  row,
  mode,
  experiments,
}: {
  row: QueryComparisonRow;
  mode: QueryComparison["mode"];
  experiments: QueryComparison["experiments"];
}) {
  const [open, setOpen] = useState(false);
  const meta = OUTCOME_META[row.outcome];
  const colSpan = 2 + experiments.length;

  return (
    <Fragment>
      <tr
        onClick={() => setOpen((v) => !v)}
        style={{
          cursor: "pointer",
          background: row.outcome === "mixed" ? "rgba(255,79,94,0.05)" : undefined,
        }}
      >
        <td style={{ textAlign: "center", color: "var(--text-muted)" }}>
          <span
            style={{
              display: "inline-block",
              transition: "transform 0.15s",
              transform: open ? "rotate(90deg)" : "rotate(0deg)",
              fontFamily: "var(--font-mono)",
            }}
          >
            ▶
          </span>
        </td>
        <td className="max-w-md">
          <div className="flex items-center gap-2">
            <span
              className="brand-badge shrink-0"
              style={{ background: meta.bg, color: meta.fg, fontSize: "0.65rem" }}
              title={meta.label}
            >
              {row.outcome === "mixed" ? "≠" : row.outcome === "all_correct" ? "✓" : "✗"}
            </span>
            <span className="truncate">{row.question}</span>
          </div>
        </td>
        {row.cells.map((cell, i) => (
          <td key={i} className="text-center">
            <CellBadge cell={cell} mode={mode} />
          </td>
        ))}
      </tr>
      {open && (
        <tr>
          <td colSpan={colSpan} style={{ padding: 0 }}>
            <div
              className="px-4 py-4"
              style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-subtle)" }}
            >
              <RowDetail row={row} mode={mode} experiments={experiments} />
            </div>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

/** Compact per-experiment cell shown in the main comparison row. */
function CellBadge({ cell, mode }: { cell: ComparisonCell | null; mode: QueryComparison["mode"] }) {
  if (!cell) {
    return <span style={{ color: "var(--text-muted)" }}>—</span>;
  }
  const ok = cell.correct === true;
  const unknown = cell.correct === null;
  const fg = unknown ? "var(--text-muted)" : ok ? "var(--color-green)" : "var(--color-coral)";
  const bg = unknown ? "var(--border-subtle)" : ok ? "rgba(1,245,122,0.15)" : "rgba(255,79,94,0.12)";

  let label: string;
  if (mode === "search") {
    const gt = cell.overlap ?? 0;
    label = unknown ? "?" : `${gt} hit`;
  } else {
    label = cell.is_error ? "err" : unknown ? "?" : ok ? "✓" : "✗";
  }

  return (
    <span className="brand-badge" style={{ background: bg, color: fg, fontWeight: 600 }}>
      {label}
    </span>
  );
}

/** Expanded full-width detail comparing each experiment's prediction. */
function RowDetail({
  row,
  mode,
  experiments,
}: {
  row: QueryComparison["rows"][number];
  mode: QueryComparison["mode"];
  experiments: QueryComparison["experiments"];
}) {
  return (
    <div className="space-y-4">
      {/* Ground truth */}
      <div>
        <div className="eyebrow mb-1">Question</div>
        <p className="text-sm mb-3" style={{ color: "var(--text-primary)" }}>{row.question}</p>
        {mode === "ask" && row.ground_truth_answer && (
          <>
            <div className="eyebrow mb-1">Ground Truth Answer</div>
            <p className="text-sm rounded-md p-3" style={{ background: "var(--bg-surface)", color: "var(--text-primary)" }}>
              {row.ground_truth_answer}
            </p>
          </>
        )}
        {mode === "search" && row.ground_truth_ids && (
          <div className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            Ground truth ({row.ground_truth_ids.length}): {row.ground_truth_ids.join(", ")}
          </div>
        )}
      </div>

      {/* Per-experiment predictions */}
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(row.cells.length, 2)}, 1fr)` }}>
        {row.cells.map((cell, i) => {
          const c = EXP_COLORS[i % EXP_COLORS.length];
          return (
            <div
              key={i}
              className="rounded-lg p-4"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)" }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="brand-badge" style={{ background: c.bg, color: c.fg, fontWeight: 700 }}>
                  {i + 1}
                </span>
                <span className="text-xs font-semibold" style={{ fontFamily: "var(--font-display)" }}>
                  {experiments[i]?.label ||
                    (experiments[i]?.effort ? effortDisplay(experiments[i].effort) : experiments[i]?.agent_name)}
                </span>
                {cell ? (
                  <CellBadge cell={cell} mode={mode} />
                ) : (
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>not present</span>
                )}
                {cell?.time_taken != null && (
                  <span className="text-xs ml-auto" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {cell.time_taken.toFixed(2)}s
                  </span>
                )}
              </div>
              {cell && <CellDetail cell={cell} mode={mode} groundTruthIds={row.ground_truth_ids} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CellDetail({
  cell,
  mode,
  groundTruthIds,
}: {
  cell: ComparisonCell;
  mode: QueryComparison["mode"];
  groundTruthIds?: string[];
}) {
  if (mode === "ask") {
    return (
      <div>
        <div className="eyebrow mb-1">Answer</div>
        <p className="text-sm rounded-md p-2 mb-2" style={{ background: "var(--bg-card)", color: "var(--text-primary)" }}>
          {cell.system_answer || <em style={{ color: "var(--text-muted)" }}>(empty)</em>}
        </p>
        {cell.judge_reasoning && (
          <details>
            <summary className="text-xs cursor-pointer" style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              Judge reasoning
            </summary>
            <p className="mt-1 text-xs rounded-md p-2" style={{ background: "var(--bg-card)", color: "var(--text-secondary)" }}>
              {cell.judge_reasoning}
            </p>
          </details>
        )}
      </div>
    );
  }

  // search mode
  const gt = new Set(groundTruthIds ?? []);
  const retrieved = cell.retrieved_ids ?? [];
  return (
    <div className="space-y-3">
      <div>
        <div className="eyebrow mb-1">
          Retrieved ({cell.num_retrieved ?? retrieved.length}) · {cell.overlap ?? 0} relevant
        </div>
        <div className="flex flex-wrap gap-1">
          {retrieved.length === 0 && (
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>none</span>
          )}
          {retrieved.slice(0, 20).map((id, i) => {
            const hit = gt.has(id);
            return (
              <span
                key={i}
                className="inline-flex items-center rounded px-1.5 py-0.5"
                style={{
                  background: hit ? "rgba(1,245,122,0.18)" : "var(--bg-card)",
                  color: hit ? "var(--color-green)" : "var(--text-muted)",
                  border: `1px solid ${hit ? "rgba(1,245,122,0.4)" : "var(--border-subtle)"}`,
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.65rem",
                }}
                title={hit ? "relevant" : undefined}
              >
                {id}
              </span>
            );
          })}
          {retrieved.length > 20 && (
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>+{retrieved.length - 20} more</span>
          )}
        </div>
      </div>
      {cell.searches != null && cell.searches.length > 0 && (
        <div>
          <div className="eyebrow mb-1">Search Plan</div>
          <SearchPlan searches={cell.searches} />
        </div>
      )}
    </div>
  );
}
