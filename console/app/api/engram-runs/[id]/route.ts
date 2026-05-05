import { NextRequest, NextResponse } from "next/server";
import { loadEngramManifest } from "@/lib/engram-runs";

export const dynamic = "force-dynamic";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const manifest = loadEngramManifest(id);
  if (!manifest) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json(manifest);
}
