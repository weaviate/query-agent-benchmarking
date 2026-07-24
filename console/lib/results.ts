import fs from "fs";
import path from "path";

// ============================================================================
// Types
// ============================================================================

export interface TrialMetadata {
  dataset: string;
  agent_name: string;
  trial_number: number;
  total_queries: number;
  total_errors?: number;
  total_misaligned?: number;
  timestamp: string;
  mode: "search" | "ask";
  effort?: string;
  filtering?: string;
  sweep_id?: string;
}

/**
 * A leaf property filter, mirroring the Query Agent's normalized filter shape.
 * e.g. { filter_type: "integer", property_name: "price", operator: "<", value: 30000 }
 * Date filters carry a structured `value` object instead of a scalar.
 */
export interface PropertyFilterLeaf {
  filter_type?: string | null;
  property_name?: string;
  operator?: string;
  value?: unknown;
  is_null?: boolean;
  // geo filters
  latitude?: number;
  longitude?: number;
  max_distance_meters?: number;
}

/** A boolean group combining nested filters with AND / OR. */
export interface FilterGroup {
  combine: "AND" | "OR";
  filters: FilterNode[];
}

export type FilterNode = PropertyFilterLeaf | FilterGroup;

export interface QuerySort {
  property_name: string;
  order: "ascending" | "descending";
  tie_break?: QuerySort | null;
}

/**
 * One structured sub-search the agent issued for a single benchmark query —
 * its search plan. Mirrors the backend `AgentSearch` model.
 */
export interface AgentSearch {
  collection: string;
  query?: string | null;
  filters?: FilterNode | null;
  sort_property?: QuerySort | null;
  uuid_value?: string | null;
}

export interface SearchQuery {
  query_id: string;
  question: string;
  ground_truth_ids: string[];
  retrieved_ids: string[];
  num_retrieved: number;
  num_ground_truth: number;
  time_taken: number;
  /** The agent's search plan. Absent/empty for direct or BYOS retrievers. */
  num_searches?: number;
  searches?: AgentSearch[];
}

export interface AskQuery {
  query_id: string;
  question: string;
  ground_truth_answer: string;
  system_answer: string;
  time_taken: number;
  score?: number;
  is_error?: boolean;
  oracle_context_id?: string;
  tenant_id?: string;
  judge_reasoning?: string;
  question_type?: string;
  retrieved_context?: Record<string, unknown>;
}

export interface TrialResultFile {
  metadata: TrialMetadata;
  queries: SearchQuery[] | AskQuery[];
  failed_query_ids?: string[];
  misaligned_query_ids?: string[];
}

export interface AggregatedResultFile {
  timestamp: string;
  config: {
    dataset: string;
    agent_name: string;
    num_trials: number;
    use_async: boolean;
    batch_size: number | null;
    max_concurrent: number | null;
    effort?: string;
    filtering?: string;
    sweep_id?: string;
  };
  [key: string]: unknown;
}

export interface TrialMetricsFile {
  avg_query_time: number;
  [key: string]: unknown;
}

// ============================================================================
// Parsed / grouped types
// ============================================================================

export type FileType = "trial_results" | "trial_metrics" | "aggregated";

export interface ParsedFile {
  filename: string;
  type: FileType;
  data: TrialResultFile | TrialMetricsFile | AggregatedResultFile;
}

export interface Experiment {
  id: string;
  dataset: string;
  agent_name: string;
  mode: "search" | "ask" | "unknown";
  num_trials: number;
  timestamp: string;
  /** Query Agent search-mode compute effort ("low" | "medium" | "high"), when known. */
  effort: string | null;
  /** Query Agent search filtering strategy ("recall" | "precision"), when known. */
  filtering: string | null;
  /** Shared id tying together the runs of one effort sweep. */
  sweep_id: string | null;
  aggregated: AggregatedResultFile | null;
  trials: {
    trialNumber: number;
    results: TrialResultFile | null;
    metrics: TrialMetricsFile | null;
  }[];
}

/** Canonical sweep column ordering: the hybrid-search baseline leads as the
 *  reference point, then the effort levels read low → high. */
export const EFFORT_RANK: Record<string, number> = {
  hybrid: 0,
  low: 1,
  medium: 2,
  high: 3,
};

// ============================================================================
// File loading
// ============================================================================

const RESULTS_DIR = path.join(process.cwd(), "results");

function getResultFiles(): string[] {
  if (!fs.existsSync(RESULTS_DIR)) {
    return [];
  }
  return fs
    .readdirSync(RESULTS_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
}

function classifyFile(
  filename: string
): { type: FileType; baseName: string; trialNumber?: number } | null {
  // Trial metrics: *-trial-N-metrics.json
  const metricsMatch = filename.match(/^(.+)-trial-(\d+)-metrics\.json$/);
  if (metricsMatch) {
    return {
      type: "trial_metrics",
      baseName: metricsMatch[1],
      trialNumber: parseInt(metricsMatch[2]),
    };
  }

  // Trial results: *-trial-N.json
  const trialMatch = filename.match(/^(.+)-trial-(\d+)\.json$/);
  if (trialMatch) {
    return {
      type: "trial_results",
      baseName: trialMatch[1],
      trialNumber: parseInt(trialMatch[2]),
    };
  }

  // Aggregated: *-results.json (but not trial files)
  if (filename.endsWith("-results.json")) {
    const baseName = filename.replace(/\.json$/, "");
    return { type: "aggregated", baseName };
  }

  return null;
}

export function loadAllExperiments(): Experiment[] {
  const files = getResultFiles();
  // Group files by baseName
  const groups: Record<
    string,
    {
      aggregated?: AggregatedResultFile;
      trialResults: Record<number, TrialResultFile>;
      trialMetrics: Record<number, TrialMetricsFile>;
    }
  > = {};

  for (const filename of files) {
    const classification = classifyFile(filename);
    if (!classification) continue;

    const { type, baseName, trialNumber } = classification;
    if (!groups[baseName]) {
      groups[baseName] = { trialResults: {}, trialMetrics: {} };
    }

    const filePath = path.join(RESULTS_DIR, filename);
    let data: unknown;
    try {
      data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    } catch {
      continue;
    }

    if (type === "aggregated") {
      groups[baseName].aggregated = data as AggregatedResultFile;
    } else if (type === "trial_results" && trialNumber !== undefined) {
      groups[baseName].trialResults[trialNumber] = data as TrialResultFile;
    } else if (type === "trial_metrics" && trialNumber !== undefined) {
      groups[baseName].trialMetrics[trialNumber] = data as TrialMetricsFile;
    }
  }

  // Build experiment objects
  const experiments: Experiment[] = [];

  for (const [baseName, group] of Object.entries(groups)) {
    const id = encodeURIComponent(baseName);

    // Determine mode, dataset, agent from trial results or aggregated
    let dataset = "unknown";
    let agent_name = "unknown";
    let mode: "search" | "ask" | "unknown" = "unknown";
    let timestamp = "";

    const trialNums = [
      ...new Set([
        ...Object.keys(group.trialResults).map(Number),
        ...Object.keys(group.trialMetrics).map(Number),
      ]),
    ].sort((a, b) => a - b);

    // Try to get metadata from first trial result
    const firstTrial = group.trialResults[trialNums[0]];
    if (firstTrial?.metadata) {
      dataset = firstTrial.metadata.dataset;
      agent_name = firstTrial.metadata.agent_name;
      mode = firstTrial.metadata.mode || "unknown";
      timestamp = firstTrial.metadata.timestamp;
    } else if (group.aggregated?.config) {
      dataset = group.aggregated.config.dataset;
      agent_name = group.aggregated.config.agent_name;
      timestamp = group.aggregated.timestamp;
    }

    // If no timestamp from trials, use aggregated
    if (!timestamp && group.aggregated) {
      timestamp = group.aggregated.timestamp;
    }

    // Effort-sweep identity: prefer the persisted config/metadata, then fall
    // back to parsing the run id baked into the filename (older result files
    // from before effort/sweep_id were persisted).
    const aggConfig = group.aggregated?.config;
    let effort = aggConfig?.effort ?? firstTrial?.metadata?.effort ?? null;
    let sweep_id = aggConfig?.sweep_id ?? firstTrial?.metadata?.sweep_id ?? null;
    const filtering = aggConfig?.filtering ?? firstTrial?.metadata?.filtering ?? null;
    if (!effort || !sweep_id) {
      const m = baseName.match(/-(\d{8}-\d{6})-effort_(low|medium|high|hybrid)-results$/);
      if (m) {
        sweep_id = sweep_id ?? m[1];
        effort = effort ?? m[2];
      }
    }

    const trials = trialNums.map((num) => ({
      trialNumber: num,
      results: group.trialResults[num] || null,
      metrics: group.trialMetrics[num] || null,
    }));

    experiments.push({
      id,
      dataset,
      agent_name,
      mode,
      num_trials: Math.max(trialNums.length, group.aggregated?.config?.num_trials || 0),
      timestamp,
      effort,
      filtering,
      sweep_id,
      aggregated: group.aggregated || null,
      trials,
    });
  }

  return experiments.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
}

export function loadExperiment(id: string): Experiment | null {
  const experiments = loadAllExperiments();
  return experiments.find((e) => e.id === id) || null;
}

// ============================================================================
// Per-query comparison across K experiments
// ============================================================================

export type ComparisonOutcome = "all_correct" | "all_wrong" | "mixed";

/** One experiment's prediction for a single aligned query. */
export interface ComparisonCell {
  present: boolean;
  /** Unified success signal — search: retrieved a relevant doc; ask: judged correct. */
  correct: boolean | null;
  time_taken?: number;
  // search-mode fields
  retrieved_ids?: string[];
  num_retrieved?: number;
  overlap?: number;
  searches?: AgentSearch[] | null;
  // ask-mode fields
  system_answer?: string;
  score?: number;
  is_error?: boolean;
  judge_reasoning?: string;
}

/** One query aligned across all compared experiments (joined by question text). */
export interface QueryComparisonRow {
  question: string;
  ground_truth_ids?: string[];
  ground_truth_answer?: string;
  question_type?: string;
  tenant_id?: string;
  /** Indexed by experiment order; `null` when the query is absent from that experiment. */
  cells: (ComparisonCell | null)[];
  presentCount: number;
  correctCount: number;
  /** Classification over present cells (only meaningful when presentCount >= 2). */
  outcome: ComparisonOutcome;
}

export interface QueryComparisonExperiment {
  id: string;
  label: string;
  agent_name: string;
  dataset: string;
  mode: "search" | "ask" | "unknown";
  effort: string | null;
  trialNumber: number | null;
}

export interface QueryComparison {
  mode: "search" | "ask" | "mixed" | "unknown";
  experiments: QueryComparisonExperiment[];
  rows: QueryComparisonRow[];
  /** Total distinct questions across the union of experiments. */
  totalQuestions: number;
  /** Questions present in every compared experiment. */
  sharedByAll: number;
  /** Summary counts over comparable rows (present in >= 2 experiments). */
  counts: { comparable: number; allCorrect: number; allWrong: number; mixed: number };
  warning?: string;
}

function firstTrialWithResults(
  exp: Experiment
): { trialNumber: number; results: TrialResultFile } | null {
  const trial = exp.trials.find((t) => t.results != null);
  return trial && trial.results ? { trialNumber: trial.trialNumber, results: trial.results } : null;
}

function buildSearchCell(q: SearchQuery): ComparisonCell {
  const gt = new Set(q.ground_truth_ids ?? []);
  const overlap = (q.retrieved_ids ?? []).filter((id) => gt.has(id)).length;
  return {
    present: true,
    correct: overlap > 0,
    time_taken: q.time_taken,
    retrieved_ids: q.retrieved_ids,
    num_retrieved: q.num_retrieved,
    overlap,
    searches: q.searches ?? null,
  };
}

function buildAskCell(q: AskQuery): ComparisonCell {
  const correct = q.is_error ? false : q.score === undefined ? null : q.score === 1;
  return {
    present: true,
    correct,
    time_taken: q.time_taken,
    system_answer: q.system_answer,
    score: q.score,
    is_error: q.is_error,
    judge_reasoning: q.judge_reasoning,
  };
}

/**
 * Build a per-query comparison across K experiments.
 *
 * Queries are aligned by **question text** (robust to per-run query-id
 * reordering from subsetting/shuffling). Uses the first trial with results for
 * each experiment. Requires all experiments to share a single mode (search or
 * ask); otherwise returns a warning and no rows.
 */
export function buildQueryComparison(ids: string[]): QueryComparison | null {
  const all = loadAllExperiments();
  const matched = ids
    .map((id) => all.find((e) => e.id === id))
    .filter((e): e is Experiment => Boolean(e));

  if (matched.length < 2) return null;

  // Match the compare API's column ordering: baseline first, then low → high.
  if (
    matched.every((e) => e.effort != null && e.effort in EFFORT_RANK) &&
    new Set(matched.map((e) => e.effort)).size === matched.length
  ) {
    matched.sort((a, b) => EFFORT_RANK[a.effort!] - EFFORT_RANK[b.effort!]);
  }

  const labels = loadLabels();
  const trials = matched.map(firstTrialWithResults);

  const experiments: QueryComparisonExperiment[] = matched.map((exp, i) => ({
    id: exp.id,
    label: labels[exp.id] ?? "",
    agent_name: exp.agent_name,
    dataset: exp.dataset,
    mode: exp.mode,
    effort: exp.effort,
    trialNumber: trials[i]?.trialNumber ?? null,
  }));

  // Determine a single shared mode for the comparison.
  const modes = new Set(matched.map((e) => e.mode));
  let mode: QueryComparison["mode"];
  let warning: string | undefined;
  if (modes.size > 1) {
    mode = "mixed";
    warning = "Selected experiments use different modes; per-query comparison requires a single mode.";
  } else {
    mode = [...modes][0] as QueryComparison["mode"];
  }

  const K = matched.length;
  const emptyResult = (w?: string): QueryComparison => ({
    mode,
    experiments,
    rows: [],
    totalQuestions: 0,
    sharedByAll: 0,
    counts: { comparable: 0, allCorrect: 0, allWrong: 0, mixed: 0 },
    warning: w ?? warning,
  });

  if (mode === "mixed" || mode === "unknown") return emptyResult();

  // Align by question text, preserving first-seen order.
  const rowMap = new Map<string, QueryComparisonRow>();

  const ensureRow = (question: string): QueryComparisonRow => {
    let row = rowMap.get(question);
    if (!row) {
      row = {
        question,
        cells: new Array(K).fill(null),
        presentCount: 0,
        correctCount: 0,
        outcome: "all_wrong",
      };
      rowMap.set(question, row);
    }
    return row;
  };

  trials.forEach((trial, expIdx) => {
    if (!trial) return;
    const queries = trial.results.queries ?? [];
    for (const q of queries) {
      const question = q.question?.trim();
      if (!question) continue;
      const row = ensureRow(question);
      if (mode === "search") {
        const sq = q as SearchQuery;
        row.cells[expIdx] = buildSearchCell(sq);
        if (!row.ground_truth_ids) row.ground_truth_ids = sq.ground_truth_ids;
      } else {
        const aq = q as AskQuery;
        row.cells[expIdx] = buildAskCell(aq);
        if (!row.ground_truth_answer) row.ground_truth_answer = aq.ground_truth_answer;
        if (!row.question_type && aq.question_type) row.question_type = aq.question_type;
        if (!row.tenant_id && aq.tenant_id) row.tenant_id = aq.tenant_id;
      }
    }
  });

  // Finalize per-row aggregates.
  const rows = [...rowMap.values()];
  let sharedByAll = 0;
  const counts = { comparable: 0, allCorrect: 0, allWrong: 0, mixed: 0 };

  for (const row of rows) {
    const present = row.cells.filter((c): c is ComparisonCell => c != null);
    row.presentCount = present.length;
    // Treat unknown (null) correctness as not-correct for counting.
    row.correctCount = present.filter((c) => c.correct === true).length;
    row.outcome =
      row.correctCount === row.presentCount
        ? "all_correct"
        : row.correctCount === 0
          ? "all_wrong"
          : "mixed";

    if (row.presentCount === K) sharedByAll++;
    if (row.presentCount >= 2) {
      counts.comparable++;
      if (row.outcome === "all_correct") counts.allCorrect++;
      else if (row.outcome === "all_wrong") counts.allWrong++;
      else counts.mixed++;
    }
  }

  // Surface disagreements first, then all-wrong, then all-correct; stable within.
  const outcomeRank: Record<ComparisonOutcome, number> = { mixed: 0, all_wrong: 1, all_correct: 2 };
  rows.sort((a, b) => {
    // Comparable rows (>=2 present) before partial ones.
    const aCmp = a.presentCount >= 2 ? 0 : 1;
    const bCmp = b.presentCount >= 2 ? 0 : 1;
    if (aCmp !== bCmp) return aCmp - bCmp;
    return outcomeRank[a.outcome] - outcomeRank[b.outcome];
  });

  return {
    mode,
    experiments,
    rows,
    totalQuestions: rows.length,
    sharedByAll,
    counts,
    warning,
  };
}

/** Return all JSON filenames on disk that belong to the given experiment id. */
export function getExperimentFiles(id: string): string[] {
  const baseName = decodeURIComponent(id);
  const allFiles = getResultFiles();
  return allFiles.filter((filename) => {
    const classification = classifyFile(filename);
    return classification !== null && classification.baseName === baseName;
  });
}

/** Delete all JSON files that belong to the given experiment id. Returns count deleted. */
export function deleteExperimentFiles(id: string): number {
  const files = getExperimentFiles(id);
  for (const filename of files) {
    fs.unlinkSync(path.join(RESULTS_DIR, filename));
  }
  return files.length;
}

// ============================================================================
// Experiment labels (user-assigned names persisted in _labels.json)
// ============================================================================

const LABELS_FILE = path.join(RESULTS_DIR, "_labels.json");

export function loadLabels(): Record<string, string> {
  try {
    if (fs.existsSync(LABELS_FILE)) {
      return JSON.parse(fs.readFileSync(LABELS_FILE, "utf-8"));
    }
  } catch {
    // ignore corrupt file
  }
  return {};
}

export function saveLabel(id: string, label: string): void {
  const labels = loadLabels();
  if (label.trim()) {
    labels[id] = label.trim();
  } else {
    delete labels[id];
  }
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  fs.writeFileSync(LABELS_FILE, JSON.stringify(labels, null, 2));
}

/**
 * Format a raw metric key into a human-readable label.
 * e.g. "avg_nDCG_at_10_mean" -> "nDCG@10"
 */
function formatMetricKey(key: string): string {
  let name = key.replace(/^avg_/, "").replace(/_mean$/, "");
  name = name.replace(/recall_at_(\d+)/g, "Recall@$1");
  name = name.replace(/nDCG_at_(\d+)/g, "nDCG@$1");
  name = name.replace(/success_at_(\d+)/g, "Success@$1");
  name = name.replace(/coverage_at_(\d+)/g, "Coverage@$1");
  name = name.replace(/alpha_ndcg_at_(\d+)/g, "alpha-nDCG@$1");
  name = name.replace("alignment_score", "Alignment");
  name = name.replace("exact_match_accuracy", "Exact Match");
  name = name.replace("officeqa_accuracy", "OfficeQA");
  return name;
}

export function getKeyMetric(experiment: Experiment): { name: string; value: number | null } {
  const agg = experiment.aggregated;
  if (!agg) return { name: "N/A", value: null };

  // Prefer the explicit key_metric written by the Python serializer.
  const keyMetricField = agg["key_metric"] as string | undefined;
  if (keyMetricField && typeof agg[keyMetricField] === "number") {
    return {
      name: formatMetricKey(keyMetricField),
      value: agg[keyMetricField] as number,
    };
  }

  // Fallback: scan for known _mean metric keys (supports older result files
  // that don't have a key_metric field yet).
  const fallbackKeys = [
    "avg_nDCG_at_10_mean",
    "avg_alignment_score_mean",
    "avg_exact_match_accuracy_mean",
    "avg_officeqa_accuracy_mean",
    "avg_success_at_5_mean",
    "avg_alpha_ndcg_at_10_mean",
    "avg_recall_at_5_mean",
    "avg_recall_at_20_mean",
    "avg_recall_at_1_mean",
  ];

  for (const key of fallbackKeys) {
    if (typeof agg[key] === "number") {
      return { name: formatMetricKey(key), value: agg[key] as number };
    }
  }

  return { name: "N/A", value: null };
}
