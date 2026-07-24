import { NextRequest, NextResponse } from "next/server";
import { loadAllExperiments, loadLabels, EFFORT_RANK } from "@/lib/results";

export const dynamic = "force-dynamic";

/** "bright/earth_science" -> "Earth Science" (the subset, title-cased). */
function subsetLabel(dataset: string): string {
  const parts = dataset.split("/");
  const subset = parts.length > 1 ? parts.slice(1).join(" ") : parts[0];
  return subset.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * GET /api/compare?ids=id1,id2,...[&aggregate=1]
 *
 * Returns the aggregated data for each requested experiment,
 * plus a unified list of all metric keys found across them.
 *
 * With aggregate=1, experiments are instead grouped by effort level (falling
 * back to agent name) and each group's metrics are macro-averaged across its
 * constituent experiments — one synthetic "experiment" per group, with the
 * per-dataset values in the trials slot so the sweep charts can break them out.
 */
export function GET(request: NextRequest) {
  const idsParam = request.nextUrl.searchParams.get("ids");
  const aggregate = request.nextUrl.searchParams.get("aggregate") === "1";
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

  // Effort sweep: every experiment carries a distinct known effort level.
  // Order columns low → medium → high so the comparison reads as a sweep.
  const isEffortSweep =
    matched.every((e) => e!.effort != null && e!.effort in EFFORT_RANK) &&
    new Set(matched.map((e) => e!.effort)).size === matched.length;
  if (isEffortSweep) {
    matched.sort((a, b) => EFFORT_RANK[a!.effort!] - EFFORT_RANK[b!.effort!]);
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

  if (aggregate) {
    // Group by effort level (agent name for runs without one); one synthetic
    // experiment per group, macro-averaged across its constituent datasets.
    const groups = new Map<string, typeof matched>();
    for (const exp of matched) {
      const key = exp!.effort ?? exp!.agent_name;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(exp);
    }
    const ordered = [...groups.entries()].sort((a, b) => {
      const ra = a[0] in EFFORT_RANK ? EFFORT_RANK[a[0]] : 99;
      const rb = b[0] in EFFORT_RANK ? EFFORT_RANK[b[0]] : 99;
      return ra - rb || a[0].localeCompare(b[0]);
    });
    // Sort each group's runs by dataset so per-dataset breakdowns line up
    // across columns.
    for (const [, exps] of ordered) {
      exps.sort((a, b) => a!.dataset.localeCompare(b!.dataset));
    }

    // Validate the groups: a cross-dataset average is only comparable when
    // every group covers the same datasets, each exactly once.
    let warning: string | null = null;
    const groupDatasets = ordered.map(([key, exps]) => ({
      key,
      datasets: exps.map((e) => e!.dataset),
    }));
    for (const g of groupDatasets) {
      const dupes = [...new Set(g.datasets.filter((d, i) => g.datasets.indexOf(d) !== i))];
      if (dupes.length > 0) {
        warning = `Unbalanced comparison: '${g.key}' includes the same dataset more than once (${dupes.join(", ")}), so its average double-counts ${dupes.length === 1 ? "that dataset" : "those datasets"}.`;
        break;
      }
    }
    if (!warning) {
      const union = [...new Set(groupDatasets.flatMap((g) => g.datasets))];
      const incomplete = groupDatasets.filter((g) => g.datasets.length !== union.length);
      if (incomplete.length > 0) {
        const largest = groupDatasets.reduce((a, b) => (b.datasets.length > a.datasets.length ? b : a));
        const g = incomplete.find((x) => x.key !== largest.key) ?? incomplete[0];
        const missing = union.filter((d) => !g.datasets.includes(d));
        warning = `Unbalanced comparison: '${largest.key}' averages ${largest.datasets.length} datasets but '${g.key}' averages ${g.datasets.length} (missing: ${missing.join(", ")}).`;
      }
    }

    const experiments = ordered.map(([key, exps]) => {
      const metrics: Record<string, number | null> = {};
      const metricsStd: Record<string, number | null> = {};
      const metricsTrials: Record<string, { trial: number; label: string; value: number | null }[]> = {};
      for (const mk of sortedMetricKeys) {
        const vals = exps.map((e) => {
          const v = e!.aggregated?.[mk];
          return typeof v === "number" ? v : null;
        });
        const nums = vals.filter((v): v is number => v !== null);
        const mean = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
        metrics[mk] = mean;
        // Std across trials of the cross-dataset average: average trial N over
        // all datasets, then take the std of those per-trial averages. Only
        // the first min-trial-count trials are pooled so every per-trial
        // average covers every dataset — pooling unequal trial counts would
        // bias the std. Dataset difficulty differences don't inflate it —
        // only run-to-run variance does.
        const base = mk.replace(/_mean$/, "");
        const perRunTrialVals = exps
          .map((e) => {
            const rawTrials = Array.isArray(e!.aggregated?.["trials"])
              ? (e!.aggregated!["trials"] as Record<string, unknown>[])
              : [];
            return rawTrials
              .map((t) => t[base])
              .filter((v): v is number => typeof v === "number");
          })
          .filter((vs) => vs.length > 0);
        const minTrials =
          perRunTrialVals.length > 0 ? Math.min(...perRunTrialVals.map((vs) => vs.length)) : 0;
        const trialMeans: number[] = [];
        for (let t = 0; t < minTrials; t++) {
          const vs = perRunTrialVals.map((vals) => vals[t]);
          trialMeans.push(vs.reduce((a, b) => a + b, 0) / vs.length);
        }
        if (trialMeans.length > 1) {
          const tm = trialMeans.reduce((a, b) => a + b, 0) / trialMeans.length;
          metricsStd[mk] = Math.sqrt(
            trialMeans.reduce((a, b) => a + (b - tm) ** 2, 0) / trialMeans.length,
          );
        } else {
          metricsStd[mk] = null;
        }
        metricsTrials[mk] = exps.map((e, i) => ({
          trial: i + 1,
          label: subsetLabel(e!.dataset),
          value: vals[i],
        }));
      }
      const first = exps[0]!;
      return {
        id: `avg-${key}`,
        label: "",
        dataset: first.dataset,
        agent_name: first.agent_name,
        mode: first.mode,
        num_trials: first.num_trials,
        timestamp: first.timestamp,
        effort: first.effort,
        metrics,
        metricsStd,
        metricsTrials,
        constituents: exps.map((e) => ({
          id: e!.id,
          dataset: e!.dataset,
          label: labels[e!.id] ?? "",
        })),
      };
    });

    const aggIsEffortSweep =
      experiments.every((e) => e.effort != null && e.effort in EFFORT_RANK) &&
      new Set(experiments.map((e) => e.effort)).size === experiments.length;

    return NextResponse.json({
      metricKeys: sortedMetricKeys,
      experiments,
      isEffortSweep: aggIsEffortSweep,
      isAggregate: true,
      ...(warning ? { warning } : {}),
    });
  }

  const experiments = matched.map((exp) => {
    const agg = exp!.aggregated;
    const metrics: Record<string, number | null> = {};
    const metricsStd: Record<string, number | null> = {};
    const metricsTrials: Record<string, { trial: number; value: number | null }[]> = {};
    // Per-trial breakdown: the aggregated file's "trials" list holds each
    // trial's raw avg_* values (the *_mean key minus its suffix).
    const rawTrials = Array.isArray(agg?.["trials"])
      ? (agg!["trials"] as Record<string, unknown>[])
      : [];
    for (const key of sortedMetricKeys) {
      const val = agg?.[key];
      metrics[key] = typeof val === "number" ? val : null;
      // Trial variance for each *_mean key lives under the *_std sibling.
      const std = agg?.[key.replace(/_mean$/, "_std")];
      metricsStd[key] = typeof std === "number" ? std : null;
      const base = key.replace(/_mean$/, "");
      metricsTrials[key] = rawTrials.map((t, i) => ({
        trial: typeof t["trial"] === "number" ? (t["trial"] as number) : i + 1,
        value: typeof t[base] === "number" ? (t[base] as number) : null,
      }));
    }

    return {
      id: exp!.id,
      label: labels[exp!.id] ?? "",
      dataset: exp!.dataset,
      agent_name: exp!.agent_name,
      mode: exp!.mode,
      num_trials: exp!.num_trials,
      timestamp: exp!.timestamp,
      effort: exp!.effort,
      metrics,
      metricsStd,
      metricsTrials,
    };
  });

  return NextResponse.json({
    metricKeys: sortedMetricKeys,
    experiments,
    isEffortSweep,
  });
}
