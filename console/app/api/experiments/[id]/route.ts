import { NextRequest, NextResponse } from "next/server";
import { loadExperiment, deleteExperimentFiles, saveLabel } from "@/lib/results";

export const dynamic = "force-dynamic";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const experiment = loadExperiment(id);

  if (!experiment) {
    return NextResponse.json({ error: "Experiment not found" }, { status: 404 });
  }

  const agg = experiment.aggregated;
  const skipKeys = new Set([
    "timestamp", "config", "query_times", "misaligned_indices",
    "metric", "total_input_tokens", "total_output_tokens",
    "num_trials", "trials", "type_accuracy",
  ]);

  const numTrials = experiment.num_trials;
  const metricEntries: { key: string; value: string }[] = [];
  if (agg) {
    const mean = agg["mean"] as Record<string, unknown> | undefined;
    const std = agg["std"] as Record<string, unknown> | undefined;

    if (mean && typeof mean === "object") {
      for (const [key, val] of Object.entries(mean)) {
        if (typeof val === "number") {
          const stdVal = std?.[key];
          metricEntries.push({
            key,
            value:
              numTrials > 1 && typeof stdVal === "number"
                ? `${(val * 100).toFixed(2)}% (+/- ${(stdVal * 100).toFixed(2)}%)`
                : `${(val * 100).toFixed(2)}%`,
          });
        }
      }
    } else {
      // Flat format: keys like avg_alignment_score_mean, avg_alignment_score_std, etc.
      const meanEntries: Record<string, number> = {};
      const stdEntries: Record<string, number> = {};

      for (const [key, val] of Object.entries(agg)) {
        if (skipKeys.has(key)) continue;
        if (typeof val !== "number") continue;
        if (key.endsWith("_std") || key.endsWith("_min") || key.endsWith("_max")) {
          if (key.endsWith("_std")) stdEntries[key.replace(/_std$/, "")] = val;
          continue;
        }
        if (key.endsWith("_mean")) {
          meanEntries[key.replace(/_mean$/, "")] = val;
        }
      }

      for (const [baseKey, val] of Object.entries(meanEntries)) {
        const isTime = baseKey.includes("time");
        const stdVal = stdEntries[baseKey];
        let value: string;
        if (isTime) {
          value = numTrials > 1 && stdVal !== undefined
            ? `${val.toFixed(2)}s (+/- ${stdVal.toFixed(2)}s)`
            : `${val.toFixed(2)}s`;
        } else {
          value = numTrials > 1 && stdVal !== undefined
            ? `${(val * 100).toFixed(2)}% (+/- ${(stdVal * 100).toFixed(2)}%)`
            : `${(val * 100).toFixed(2)}%`;
        }
        metricEntries.push({ key: baseKey, value });
      }
    }
  }

  return NextResponse.json({
    id: experiment.id,
    dataset: experiment.dataset,
    agent_name: experiment.agent_name,
    mode: experiment.mode,
    num_trials: experiment.num_trials,
    timestamp: experiment.timestamp,
    metricEntries,
    trials: experiment.trials.map((t) => ({
      trialNumber: t.trialNumber,
      hasResults: t.results !== null,
      totalQueries: t.results?.metadata?.total_queries ?? null,
      avgQueryTime: t.metrics?.avg_query_time ?? null,
      metrics: t.metrics,
    })),
  });
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();
  const label = typeof body.label === "string" ? body.label : "";
  saveLabel(id, label);
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const experiment = loadExperiment(id);

  if (!experiment) {
    return NextResponse.json({ error: "Experiment not found" }, { status: 404 });
  }

  const deletedCount = deleteExperimentFiles(id);
  return NextResponse.json({ deleted: deletedCount });
}
