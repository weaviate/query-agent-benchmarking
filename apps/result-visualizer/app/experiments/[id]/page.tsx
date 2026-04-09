import { loadExperiment } from "@/lib/results";
import { notFound } from "next/navigation";

export default async function ExperimentDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const experiment = loadExperiment(id);

  if (!experiment) {
    notFound();
  }

  const agg = experiment.aggregated;

  // Extract metric entries from aggregated results (skip non-metric keys)
  const skipKeys = new Set(["timestamp", "config", "query_times", "misaligned_indices", "metric", "total_input_tokens", "total_output_tokens"]);
  const metricEntries: { key: string; value: unknown }[] = [];
  if (agg) {
    // Check if there's a "mean" sub-object (multi-trial aggregation)
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
        if (key === "mean" || key === "std" || key === "min" || key === "max") continue;
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

  return (
    <div>
      <div className="mb-6">
        <a href="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
          &larr; Back to dashboard
        </a>
      </div>

      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">{experiment.dataset}</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Agent: <span className="font-mono">{experiment.agent_name}</span> &middot;
          Mode: <span className="capitalize">{experiment.mode}</span> &middot;
          {experiment.num_trials} trial{experiment.num_trials !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Aggregated metrics */}
      {metricEntries.length > 0 && (
        <section className="mb-10">
          <h2 className="text-lg font-semibold mb-4">Aggregated Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {metricEntries.map(({ key, value }) => (
              <div
                key={key}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
              >
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="text-lg font-semibold">{String(value)}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-trial table */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Trials</h2>
        {experiment.trials.length === 0 ? (
          <p className="text-gray-500">No trial data available.</p>
        ) : (
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
              <tr>
                <th className="py-3 pr-4">Trial</th>
                <th className="py-3 pr-4">Queries</th>
                <th className="py-3 pr-4">Avg Query Time</th>
                <th className="py-3 pr-4">Key Score</th>
                <th className="py-3"></th>
              </tr>
            </thead>
            <tbody>
              {experiment.trials.map((trial) => {
                const totalQueries = trial.results?.metadata?.total_queries ?? "--";
                const avgTime =
                  trial.metrics?.avg_query_time != null
                    ? `${trial.metrics.avg_query_time.toFixed(2)}s`
                    : "--";

                // Find key score from metrics
                let keyScore = "--";
                if (trial.metrics) {
                  const m = trial.metrics as Record<string, unknown>;
                  for (const k of [
                    "avg_alignment_score",
                    "avg_exact_match_accuracy",
                    "avg_recall@5",
                    "avg_nDCG@10",
                  ]) {
                    if (typeof m[k] === "number") {
                      keyScore = `${((m[k] as number) * 100).toFixed(1)}%`;
                      break;
                    }
                  }
                }

                const hasResults = trial.results !== null;

                return (
                  <tr
                    key={trial.trialNumber}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-3 pr-4 font-medium">Trial {trial.trialNumber}</td>
                    <td className="py-3 pr-4">{totalQueries}</td>
                    <td className="py-3 pr-4">{avgTime}</td>
                    <td className="py-3 pr-4">{keyScore}</td>
                    <td className="py-3">
                      {hasResults ? (
                        <a
                          href={`/experiments/${id}/trial/${trial.trialNumber}`}
                          className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
                        >
                          View queries
                        </a>
                      ) : (
                        <span className="text-xs text-gray-400">No query data</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
