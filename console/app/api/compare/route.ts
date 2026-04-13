import { NextRequest, NextResponse } from "next/server";
import { loadAllExperiments, loadLabels } from "@/lib/results";

export const dynamic = "force-dynamic";

/**
 * GET /api/compare?ids=id1,id2,...
 *
 * Returns the aggregated data for each requested experiment,
 * plus a unified list of all metric keys found across them.
 */
export function GET(request: NextRequest) {
  const idsParam = request.nextUrl.searchParams.get("ids");
  if (!idsParam) {
    return NextResponse.json({ error: "Missing ids parameter" }, { status: 400 });
  }

  const requestedIds = idsParam.split(",").map((s) => s.trim()).filter(Boolean);
  if (requestedIds.length < 2) {
    return NextResponse.json({ error: "Need at least 2 experiment ids" }, { status: 400 });
  }

  const allExperiments = loadAllExperiments();
  const matched = requestedIds
    .map((id) => allExperiments.find((e) => e.id === id))
    .filter(Boolean);

  if (matched.length < 2) {
    return NextResponse.json({ error: "Could not find at least 2 of the requested experiments" }, { status: 404 });
  }

  // Collect all _mean metric keys across experiments
  const allMetricKeys = new Set<string>();
  for (const exp of matched) {
    if (!exp!.aggregated) continue;
    for (const key of Object.keys(exp!.aggregated)) {
      if (key.endsWith("_mean") && typeof exp!.aggregated[key] === "number") {
        allMetricKeys.add(key);
      }
    }
  }

  // Sort metrics: recall, ndcg, success, coverage, alignment, time last
  const priorityTerms = ["recall", "ndcg", "success", "coverage", "alpha", "alignment", "exact_match", "officeqa"];
  const sortedMetricKeys = [...allMetricKeys].sort((a, b) => {
    const aLower = a.toLowerCase();
    const bLower = b.toLowerCase();

    const aIsTime = aLower.includes("time");
    const bIsTime = bLower.includes("time");
    if (aIsTime !== bIsTime) return aIsTime ? 1 : -1;

    let aPri = priorityTerms.length;
    let bPri = priorityTerms.length;
    for (let i = 0; i < priorityTerms.length; i++) {
      if (aLower.includes(priorityTerms[i]) && aPri === priorityTerms.length) aPri = i;
      if (bLower.includes(priorityTerms[i]) && bPri === priorityTerms.length) bPri = i;
    }
    if (aPri !== bPri) return aPri - bPri;

    // Sub-sort by number in name (e.g., @1 < @5 < @20)
    const aNum = parseInt((a.match(/\d+/) || ["0"])[0]);
    const bNum = parseInt((b.match(/\d+/) || ["0"])[0]);
    if (aNum !== bNum) return aNum - bNum;

    return a.localeCompare(b);
  });

  const labels = loadLabels();
  const experiments = matched.map((exp) => {
    const agg = exp!.aggregated;
    const metrics: Record<string, number | null> = {};
    for (const key of sortedMetricKeys) {
      const val = agg?.[key];
      metrics[key] = typeof val === "number" ? val : null;
    }

    return {
      id: exp!.id,
      label: labels[exp!.id] ?? "",
      dataset: exp!.dataset,
      agent_name: exp!.agent_name,
      mode: exp!.mode,
      num_trials: exp!.num_trials,
      timestamp: exp!.timestamp,
      metrics,
    };
  });

  return NextResponse.json({
    metricKeys: sortedMetricKeys,
    experiments,
  });
}
