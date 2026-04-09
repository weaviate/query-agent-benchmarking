import { NextRequest, NextResponse } from "next/server";
import { loadExperiment } from "@/lib/results";

export function GET(request: NextRequest) {
  const id = request.nextUrl.searchParams.get("id");
  const trialNum = request.nextUrl.searchParams.get("trial");

  if (!id || !trialNum) {
    return NextResponse.json({ error: "Missing id or trial param" }, { status: 400 });
  }

  const experiment = loadExperiment(id);
  if (!experiment) {
    return NextResponse.json({ error: "Experiment not found" }, { status: 404 });
  }

  const trial = experiment.trials.find((t) => t.trialNumber === parseInt(trialNum));
  if (!trial?.results) {
    return NextResponse.json({ error: "Trial not found" }, { status: 404 });
  }

  return NextResponse.json(trial.results);
}
