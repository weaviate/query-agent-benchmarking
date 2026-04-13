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

export function getKeyMetric(experiment: Experiment): { name: string; value: number | null } {
  const agg = experiment.aggregated;
  if (!agg) return { name: "N/A", value: null };

  // Check for common metric keys
  const metricKeys = [
    "avg_alignment_score",
    "avg_exact_match_accuracy",
    "avg_recall@5",
    "avg_nDCG@10",
    "avg_recall@20",
    "avg_recall@1",
  ];

  for (const key of metricKeys) {
    // Check both top-level and nested in mean
    if (typeof agg[key] === "number") {
      return { name: key.replace("avg_", ""), value: agg[key] as number };
    }
    const mean = agg["mean"] as Record<string, number> | undefined;
    if (mean && typeof mean[key] === "number") {
      return { name: key.replace("avg_", ""), value: mean[key] };
    }
  }

  return { name: "N/A", value: null };
}
