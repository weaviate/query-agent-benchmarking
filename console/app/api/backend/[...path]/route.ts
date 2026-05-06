import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 600; // 10 minutes for long-running operations

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const target = `${BACKEND_URL}/${path.join("/")}`;
  const url = new URL(target);
  request.nextUrl.searchParams.forEach((val, key) => url.searchParams.set(key, val));

  const resp = await fetch(url.toString(), {
    signal: AbortSignal.timeout(600_000),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const target = `${BACKEND_URL}/${path.join("/")}`;
  const body = await request.json();

  // For the streaming populate endpoint, pass through as a stream
  if (path.join("/") === "populate-db-stream") {
    const resp = await fetch(target, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(600_000),
    });

    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({ status: "error" }));
      return NextResponse.json(data, { status: resp.status });
    }

    // Stream the response through to the client
    return new Response(resp.body, {
      headers: {
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
      },
    });
  }

  const resp = await fetch(target, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(600_000),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
