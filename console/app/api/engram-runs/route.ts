import { NextResponse } from "next/server";
import { loadAllEngramManifests } from "@/lib/engram-runs";

export const dynamic = "force-dynamic";

export function GET() {
  const manifests = loadAllEngramManifests();
  return NextResponse.json(manifests);
}
