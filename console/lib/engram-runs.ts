import fs from "fs";
import path from "path";

// ============================================================================
// Types
// ============================================================================

export interface EngramRunEntry {
  run_id: string;
  tenant_id: string;
  session_id: string;
  session_date: string;
  submitted_at: number;
  status?: string;
  created?: number;
  updated?: number;
  deleted?: number;
  run_duration_seconds?: number;
}

export interface EngramTenantStats {
  tenant_id: string;
  num_sessions: number;
  elapsed_seconds: number;
  total_created: number;
  total_updated: number;
  total_deleted: number;
}

export interface EngramManifest {
  type: "engram_ingest";
  timestamp: string;
  dataset: string;
  submit_elapsed_seconds: number;
  total_sessions: number;
  total_tenants: number;
  tenant_session_counts: Record<string, number>;
  runs: EngramRunEntry[];
  stats?: EngramTenantStats[];
}

export interface EngramManifestSummary {
  id: string;
  filename: string;
  timestamp: string;
  dataset: string;
  total_sessions: number;
  total_tenants: number;
  submit_elapsed_seconds: number;
  total_created: number;
  total_updated: number;
  total_deleted: number;
}

// ============================================================================
// File loading
// ============================================================================

const RESULTS_DIR = path.join(process.cwd(), "results");

function getEngramManifestFiles(): string[] {
  if (!fs.existsSync(RESULTS_DIR)) {
    return [];
  }
  return fs
    .readdirSync(RESULTS_DIR)
    .filter((f) => f.startsWith("engram-ingest-") && f.endsWith(".json"))
    .sort()
    .reverse(); // newest first
}

export function loadAllEngramManifests(): EngramManifestSummary[] {
  const files = getEngramManifestFiles();
  const summaries: EngramManifestSummary[] = [];

  for (const filename of files) {
    try {
      const filePath = path.join(RESULTS_DIR, filename);
      const data = JSON.parse(fs.readFileSync(filePath, "utf-8")) as EngramManifest;
      if (data.type !== "engram_ingest") continue;

      // Aggregate memory operation counts from stats or runs
      let totalCreated = 0;
      let totalUpdated = 0;
      let totalDeleted = 0;

      if (data.stats) {
        for (const s of data.stats) {
          totalCreated += s.total_created;
          totalUpdated += s.total_updated;
          totalDeleted += s.total_deleted;
        }
      } else {
        for (const r of data.runs) {
          totalCreated += r.created ?? 0;
          totalUpdated += r.updated ?? 0;
          totalDeleted += r.deleted ?? 0;
        }
      }

      const id = encodeURIComponent(filename.replace(/\.json$/, ""));
      summaries.push({
        id,
        filename,
        timestamp: data.timestamp,
        dataset: data.dataset,
        total_sessions: data.total_sessions,
        total_tenants: data.total_tenants,
        submit_elapsed_seconds: data.submit_elapsed_seconds,
        total_created: totalCreated,
        total_updated: totalUpdated,
        total_deleted: totalDeleted,
      });
    } catch {
      continue;
    }
  }

  return summaries;
}

export function loadEngramManifest(id: string): EngramManifest | null {
  const filename = decodeURIComponent(id) + ".json";
  const filePath = path.join(RESULTS_DIR, filename);
  try {
    if (!fs.existsSync(filePath)) return null;
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8")) as EngramManifest;
    return data.type === "engram_ingest" ? data : null;
  } catch {
    return null;
  }
}
