"use client";

import { useEffect, useState, useCallback } from "react";
import type { TrialResultFile, SearchQuery, AskQuery } from "@/lib/results";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700 transition-colors"
      title="Copy formatted query to clipboard"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function formatAskQueryForCopy(q: AskQuery): string {
  return `Question: ${q.question}

Ground Truth: ${q.ground_truth_answer}

System Answer: ${q.system_answer}

Judge Reasoning: ${q.judge_reasoning ?? "N/A"}`;
}

// We load data via an API route to support client-side filtering
// But since we're using SSR, we'll use a hybrid approach

export default function TrialQueriesPage({
  params: paramsPromise,
}: {
  params: Promise<{ id: string; trialNum: string }>;
}) {
  const [params, setParams] = useState<{ id: string; trialNum: string } | null>(null);
  const [data, setData] = useState<TrialResultFile | null>(null);
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect" | "errors">("all");
  const [sortBy, setSortBy] = useState<"id" | "time">("id");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    paramsPromise.then(setParams);
  }, [paramsPromise]);

  useEffect(() => {
    if (!params) return;
    fetch(`/api/trial?id=${params.id}&trial=${params.trialNum}`)
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [params]);

  if (loading || !params) {
    return <div className="py-20 text-center text-gray-500">Loading...</div>;
  }

  if (!data) {
    return <div className="py-20 text-center text-gray-500">Trial data not found.</div>;
  }

  const mode = data.metadata.mode;
  const isAsk = mode === "ask";

  // Apply filter and sort
  let filteredQueries: SearchQuery[] | AskQuery[];
  if (isAsk) {
    let askQueries = data.queries as AskQuery[];
    if (filter !== "all") {
      askQueries = askQueries.filter((q) => {
        if (filter === "correct") return q.score === 1 && !q.is_error;
        if (filter === "incorrect") return q.score === 0 && !q.is_error;
        if (filter === "errors") return q.is_error;
        return true;
      });
    }
    if (sortBy === "time") {
      askQueries = [...askQueries].sort((a, b) => b.time_taken - a.time_taken);
    }
    filteredQueries = askQueries;
  } else {
    let searchQueries = data.queries as SearchQuery[];
    if (sortBy === "time") {
      searchQueries = [...searchQueries].sort((a, b) => b.time_taken - a.time_taken);
    }
    filteredQueries = searchQueries;
  }

  return (
    <div>
      <div className="mb-6">
        <a
          href={`/experiments/${params.id}`}
          className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
        >
          &larr; Back to experiment
        </a>
      </div>

      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-1">
          Trial {data.metadata.trial_number} &mdash; {data.metadata.dataset}
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {data.metadata.total_queries} queries &middot; Mode: {mode}
          {data.metadata.total_errors != null && data.metadata.total_errors > 0 && (
            <span className="text-red-500"> &middot; {data.metadata.total_errors} errors</span>
          )}
          {data.metadata.total_misaligned != null && data.metadata.total_misaligned > 0 && (
            <span className="text-yellow-500"> &middot; {data.metadata.total_misaligned} misaligned</span>
          )}
        </p>
        {data.failed_query_ids && data.failed_query_ids.length > 0 && (
          <p className="text-xs text-red-500 mt-1">
            Failed IDs: {data.failed_query_ids.join(", ")}
          </p>
        )}
        {data.misaligned_query_ids && data.misaligned_query_ids.length > 0 && (
          <p className="text-xs text-yellow-500 mt-1">
            Misaligned IDs: {data.misaligned_query_ids.join(", ")}
          </p>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-4 mb-6 items-center flex-wrap">
        {isAsk && (
          <div className="flex gap-1 items-center">
            <span className="text-xs text-gray-500 mr-1">Filter:</span>
            {(["all", "correct", "incorrect", "errors"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded text-xs font-medium ${
                  filter === f
                    ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
                    : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        )}
        <div className="flex gap-1 items-center">
          <span className="text-xs text-gray-500 mr-1">Sort:</span>
          {(["id", "time"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className={`px-3 py-1 rounded text-xs font-medium ${
                sortBy === s
                  ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
                  : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {s === "id" ? "Query ID" : "Time (desc)"}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-400 ml-auto">
          Showing {filteredQueries.length} of {data.queries.length}
        </span>
      </div>

      {/* Query table */}
      {isAsk ? (
        <AskQueriesTable queries={filteredQueries as AskQuery[]} />
      ) : (
        <SearchQueriesTable queries={filteredQueries as SearchQuery[]} />
      )}
    </div>
  );
}

function AskQueriesTable({ queries }: { queries: AskQuery[] }) {
  return (
    <div className="space-y-3">
      {queries.map((q) => {
        const isError = q.is_error === true;
        const isCorrect = q.score === 1 && !isError;
        const borderColor = isError
          ? "border-orange-400 dark:border-orange-700"
          : q.score === undefined
            ? "border-gray-200 dark:border-gray-700"
            : isCorrect
              ? "border-green-300 dark:border-green-800"
              : "border-red-300 dark:border-red-800";
        const bgColor = isError
          ? "bg-orange-50 dark:bg-orange-950/30"
          : q.score === undefined
            ? ""
            : isCorrect
              ? "bg-green-50 dark:bg-green-950/30"
              : "bg-red-50 dark:bg-red-950/30";

        const statusLabel = isError
          ? "Error"
          : isCorrect
            ? "Correct"
            : q.score === 0
              ? "Incorrect"
              : undefined;
        const statusColor = isError
          ? "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200"
          : isCorrect
            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";

        return (
          <div key={q.query_id} className={`border rounded-lg p-4 ${borderColor} ${bgColor}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-gray-500">{q.query_id}</span>
              <div className="flex items-center gap-3">
                <CopyButton text={formatAskQueryForCopy(q)} />
                <span className="text-xs text-gray-400">{q.time_taken.toFixed(2)}s</span>
                {q.tenant_id && (
                  <span className="text-xs text-gray-400">tenant: {q.tenant_id}</span>
                )}
                {statusLabel && (
                  <span className={`text-xs font-medium px-2 py-0.5 rounded ${statusColor}`}>
                    {statusLabel}
                  </span>
                )}
              </div>
            </div>
            <div className="mb-2">
              <div className="text-xs text-gray-500 mb-0.5">Question</div>
              <p className="text-sm">{q.question}</p>
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-gray-500 mb-0.5">Ground Truth</div>
                <p className="text-sm bg-gray-50 dark:bg-gray-900 rounded p-2">
                  {q.ground_truth_answer}
                </p>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-0.5">System Answer</div>
                <p className="text-sm bg-gray-50 dark:bg-gray-900 rounded p-2">
                  {q.system_answer}
                </p>
              </div>
            </div>
            {q.judge_reasoning && (
              <details className="mt-3">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300">
                  Judge Reasoning
                </summary>
                <p className="mt-1 text-sm bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded p-2 text-gray-700 dark:text-gray-300">
                  {q.judge_reasoning}
                </p>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

function SearchQueriesTable({ queries }: { queries: SearchQuery[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="text-xs uppercase text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
          <tr>
            <th className="py-3 pr-4">ID</th>
            <th className="py-3 pr-4">Question</th>
            <th className="py-3 pr-4">Ground Truth</th>
            <th className="py-3 pr-4">Retrieved</th>
            <th className="py-3 pr-4">Time</th>
          </tr>
        </thead>
        <tbody>
          {queries.map((q) => {
            const overlap = q.retrieved_ids.filter((id) =>
              q.ground_truth_ids.includes(id)
            ).length;
            const hasOverlap = overlap > 0;

            return (
              <tr
                key={q.query_id}
                className={`border-b border-gray-100 dark:border-gray-800 ${
                  hasOverlap ? "" : "bg-red-50 dark:bg-red-950/20"
                }`}
              >
                <td className="py-3 pr-4 font-mono text-xs">{q.query_id}</td>
                <td className="py-3 pr-4 max-w-xs truncate">{q.question}</td>
                <td className="py-3 pr-4 text-xs">
                  {q.num_ground_truth} IDs
                </td>
                <td className="py-3 pr-4 text-xs">
                  {q.num_retrieved} IDs ({overlap} match)
                </td>
                <td className="py-3 pr-4 text-xs">{q.time_taken.toFixed(2)}s</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
