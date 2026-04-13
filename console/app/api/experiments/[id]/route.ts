import { NextRequest, NextResponse } from "next/server";
import { loadExperiment, deleteExperimentFiles } from "@/lib/results";

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
  ]);

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
              typeof stdVal === "number"
                ? `${(val * 100).toFixed(2)}% (+/- ${(stdVal * 100).toFixed(2)}%)`
                : `${(val * 100).toFixed(2)}%`,
          });
        }
      }
    } else {
      for (const [key, val] of Object.entries(agg)) {
        if (skipKeys.has(key)) continue;
        if (["mean", "std", "min", "max"].includes(key)) continue;
        if (typeof val === "number") {
          const isTime = key.includes("time");
          metricEntries.push({
            key,
            value: isTime ? `${val.toFixed(2)}s` : `${(val * 100).toFixed(2)}%`,
          });
        }
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
