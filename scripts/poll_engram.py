"""Poll Engram run status from an ingestion manifest.

Manifest runs are in submission order and Engram processes each user's runs
in FIFO order, so we can binary-search per tenant to find the latest completed
run rather than checking every run individually. Multiple tenants are searched
concurrently.

Usage:
  # Single pass — print counts and exit
  uv run python3 scripts/poll_engram.py engram-ingest-longmemeval-s-20260625-103721.json

  # Loop every N seconds with a tqdm progress bar until all runs finish
  uv run python3 scripts/poll_engram.py engram-ingest-longmemeval-s-20260625-103721.json --poll 10
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from engram import EngramClient
from tqdm import tqdm

from query_agent_benchmarking.internal.adapters.results.serialization import RESULTS_DIR
from query_agent_benchmarking.internal.config.loader import load_config

load_dotenv()

_TERMINAL = {"completed", "failed", "deleted"}

_DB_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "query_agent_benchmarking"
    / "internal"
    / "config"
    / "database_loader_config.yml"
)


def _make_client() -> EngramClient:
    cfg = load_config(_DB_CONFIG_PATH)
    base_url = cfg.get("engram_base_url", "https://dev-engram.labs.weaviate.io")
    return EngramClient(api_key=os.environ["ENGRAM_API_KEY"], base_url=base_url)


def _is_terminal(client: EngramClient, run_id: str) -> bool:
    try:
        return client.runs.get(run_id).status in _TERMINAL
    except Exception:
        return True  # treat errors as done so they don't block the search


def _binary_search_frontier(client: EngramClient, runs: list[dict], known_frontier: int) -> int:
    """Find the rightmost terminal run index using binary search.

    Exploits the FIFO ordering guarantee: if run[k] is terminal then
    run[0..k-1] are also terminal; if run[k] is still running then
    run[k+1..] are also still running.

    Args:
        runs: Tenant's runs in submission order.
        known_frontier: Index of the last confirmed-terminal run (-1 = none).

    Returns:
        Updated frontier (rightmost terminal index, or known_frontier unchanged).
    """
    lo = known_frontier + 1
    hi = len(runs) - 1
    result = known_frontier

    while lo <= hi:
        mid = (lo + hi) // 2
        if _is_terminal(client, runs[mid]["run_id"]):
            result = mid
            lo = mid + 1   # could be more completed to the right
        else:
            hi = mid - 1   # nothing to the right can be complete yet

    return result


def _group_by_tenant(all_runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by tenant_id, preserving manifest (submission) order."""
    groups: dict[str, list[dict]] = {}
    for r in all_runs:
        groups.setdefault(r["tenant_id"], []).append(r)
    return groups


def _seed_frontiers(tenant_runs: dict[str, list[dict]]) -> dict[str, int]:
    """Pre-seed frontiers from manifest status fields — no API calls needed.

    Scans from the start of each tenant's runs and advances the frontier for
    every run that already carries a terminal status in the manifest.
    Stops at the first non-terminal entry (monotonicity guarantee).
    """
    frontiers: dict[str, int] = {}
    for tid, runs in tenant_runs.items():
        frontier = -1
        for i, r in enumerate(runs):
            if r.get("status") in _TERMINAL:
                frontier = i
            else:
                break
        frontiers[tid] = frontier
    return frontiers


def _advance_all_frontiers(
    client: EngramClient,
    tenant_runs: dict[str, list[dict]],
    frontiers: dict[str, int],
) -> dict[str, int]:
    """Binary-search all incomplete tenants concurrently."""
    pending_tenants = [
        tid for tid, f in frontiers.items()
        if f < len(tenant_runs[tid]) - 1
    ]
    if not pending_tenants:
        return frontiers

    with ThreadPoolExecutor(max_workers=len(pending_tenants)) as executor:
        future_to_tid = {
            executor.submit(
                _binary_search_frontier, client, tenant_runs[tid], frontiers[tid]
            ): tid
            for tid in pending_tenants
        }
        for future in as_completed(future_to_tid):
            tid = future_to_tid[future]
            frontiers[tid] = future.result()

    return frontiers


def _count_done(frontiers: dict[str, int]) -> int:
    return sum(f + 1 for f in frontiers.values())


def _total_runs(tenant_runs: dict[str, list[dict]]) -> int:
    return sum(len(runs) for runs in tenant_runs.values())


def _all_complete(tenant_runs: dict[str, list[dict]], frontiers: dict[str, int]) -> bool:
    return all(frontiers[tid] == len(runs) - 1 for tid, runs in tenant_runs.items())


def _print_user_lists(tenant_runs: dict[str, list[dict]], frontiers: dict[str, int]) -> None:
    completed = [tid for tid, runs in tenant_runs.items() if frontiers[tid] == len(runs) - 1]
    in_progress = [tid for tid in tenant_runs if tid not in completed]
    q = '"'
    print(f"completed: {','.join(q + tid + q for tid in completed)}")
    print(f"in_progress: {','.join(q + tid + q for tid in in_progress)}")


def run_single_pass(
    client: EngramClient,
    tenant_runs: dict[str, list[dict]],
    frontiers: dict[str, int],
    show_tenants: bool = False,
) -> None:
    frontiers = _advance_all_frontiers(client, tenant_runs, frontiers)
    total = _total_runs(tenant_runs)
    done = _count_done(frontiers)
    print(f"{done}/{total} runs done  (still running: {total - done})")
    if show_tenants:
        _print_user_lists(tenant_runs, frontiers)


def run_progress_loop(
    client: EngramClient,
    tenant_runs: dict[str, list[dict]],
    frontiers: dict[str, int],
    interval: float,
    show_tenants: bool = False,
) -> None:
    total = _total_runs(tenant_runs)

    frontiers = _advance_all_frontiers(client, tenant_runs, frontiers)
    bar = tqdm(total=total, initial=_count_done(frontiers), desc="Engram runs", unit="run")

    try:
        while not _all_complete(tenant_runs, frontiers):
            time.sleep(interval)
            prev_done = _count_done(frontiers)
            frontiers = _advance_all_frontiers(client, tenant_runs, frontiers)
            new_done = _count_done(frontiers)
            if new_done > prev_done:
                bar.update(new_done - prev_done)
    finally:
        bar.close()

    done = _count_done(frontiers)
    print(f"\nAll {done}/{total} runs finished")
    if show_tenants:
        _print_user_lists(tenant_runs, frontiers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Engram run status from an ingestion manifest")
    parser.add_argument(
        "manifest",
        help="Manifest filename in console/results/ (e.g. engram-ingest-longmemeval-s-20260625-103721.json)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        metavar="N",
        help="Poll every N seconds with a live progress bar until all runs finish",
    )
    parser.add_argument(
        "--tenants",
        action="store_true",
        help="On exit, print comma-separated lists of completed and in-progress tenants",
    )
    args = parser.parse_args()

    manifest_path = RESULTS_DIR / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    all_runs = manifest.get("runs", [])
    if not all_runs:
        raise SystemExit("Manifest contains no runs.")

    tenant_runs = _group_by_tenant(all_runs)
    frontiers = _seed_frontiers(tenant_runs)

    client = _make_client()

    if args.poll is not None:
        run_progress_loop(client, tenant_runs, frontiers, interval=args.poll, show_tenants=args.tenants)
    else:
        run_single_pass(client, tenant_runs, frontiers, show_tenants=args.tenants)


if __name__ == "__main__":
    main()
