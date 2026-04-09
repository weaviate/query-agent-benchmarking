import { loadAllExperiments, getKeyMetric } from "@/lib/results";

function ScoreBadge({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-400">--</span>;
  const pct = value * 100;
  const color =
    pct >= 80
      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
      : pct >= 50
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-sm font-medium ${color}`}>
      {pct.toFixed(1)}%
    </span>
  );
}

function ModeBadge({ mode }: { mode: string }) {
  const color =
    mode === "search"
      ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
      : mode === "ask"
        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
        : "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {mode}
    </span>
  );
}

export default function Dashboard() {
  const experiments = loadAllExperiments();

  if (experiments.length === 0) {
    return (
      <div className="text-center py-20">
        <h2 className="text-xl font-semibold mb-2">No results found</h2>
        <p className="text-gray-500 dark:text-gray-400">
          Run a benchmark to generate results in{" "}
          <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm">
            result-visualizer/results/
          </code>
        </p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Experiments</h1>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
            <tr>
              <th className="py-3 pr-4">Dataset</th>
              <th className="py-3 pr-4">Agent</th>
              <th className="py-3 pr-4">Mode</th>
              <th className="py-3 pr-4">Key Metric</th>
              <th className="py-3 pr-4">Trials</th>
              <th className="py-3 pr-4">Timestamp</th>
              <th className="py-3"></th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => {
              const metric = getKeyMetric(exp);
              return (
                <tr
                  key={exp.id}
                  className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900"
                >
                  <td className="py-3 pr-4 font-medium">{exp.dataset}</td>
                  <td className="py-3 pr-4 font-mono text-xs">{exp.agent_name}</td>
                  <td className="py-3 pr-4">
                    <ModeBadge mode={exp.mode} />
                  </td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <ScoreBadge value={metric.value} />
                      <span className="text-xs text-gray-400">{metric.name}</span>
                    </div>
                  </td>
                  <td className="py-3 pr-4">{exp.num_trials}</td>
                  <td className="py-3 pr-4 text-xs text-gray-500">
                    {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : "--"}
                  </td>
                  <td className="py-3">
                    <a
                      href={`/experiments/${exp.id}`}
                      className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
                    >
                      View details
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
