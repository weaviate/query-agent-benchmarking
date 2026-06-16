import { NextRequest, NextResponse } from "next/server";
import { buildQueryComparison } from "@/lib/results";

export const dynamic = "force-dynamic";

/**
 * GET /api/compare/queries?ids=id1,id2,...
 *
 * Returns a per-query comparison across the requested experiments, aligning
 * queries by question text and classifying each as all-correct / all-wrong /
 * mixed (disagreement). Loaded on demand by the compare view.
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

  const comparison = buildQueryComparison(requestedIds);
  if (!comparison) {
    return NextResponse.json(
      { error: "Could not find at least 2 of the requested experiments" },
      { status: 404 }
    );
  }

  return NextResponse.json(comparison);
}
