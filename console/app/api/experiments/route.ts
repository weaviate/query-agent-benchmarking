import { NextResponse } from "next/server";
import { loadAllExperiments, getKeyMetric } from "@/lib/results";

export const dynamic = "force-dynamic";

export function GET() {
  const experiments = loadAllExperiments();
  const payload = experiments.map((exp) => {
    const metric = getKeyMetric(exp);
    return {
      id: exp.id,
      dataset: exp.dataset,
      agent_name: exp.agent_name,
      mode: exp.mode,
      num_trials: exp.num_trials,
      timestamp: exp.timestamp,
      keyMetric: metric,
      aggregated: exp.aggregated,
      trials: exp.trials.map((t) => ({
        trialNumber: t.trialNumber,
        hasResults: t.results !== null,
        totalQueries: t.results?.metadata?.total_queries ?? null,
        avgQueryTime: t.metrics?.avg_query_time ?? null,
        metrics: t.metrics,
      })),
    };
  });
  return NextResponse.json(payload);
}
