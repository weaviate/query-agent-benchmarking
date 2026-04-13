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
}

export interface SearchQuery {
  query_id: string;
  question: string;
  ground_truth_ids: string[];
  retrieved_ids: string[];
  num_retrieved: number;
  num_ground_truth: number;
  time_taken: number;
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
  aggregated: AggregatedResultFile | null;
  trials: {
    trialNumber: number;
    results: TrialResultFile | null;
    metrics: TrialMetricsFile | null;
  }[];
}

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
