"""Tests for the effort sweep's per-run config overrides.

Verifies that ``effort_sweep_overrides`` entries (keyed by effort level or
baseline agent name) are applied to the right runs, without touching any
real infrastructure — ``_run_search_eval`` is stubbed out.
"""

import asyncio

import pytest

from query_agent_benchmarking.internal.core.services import search_benchmark


@pytest.fixture
def captured_configs(monkeypatch):
    """Stub _run_search_eval and capture the config each sweep run receives."""
    configs: dict[str, dict] = {}

    async def fake_run_search_eval(cfg):
        configs[cfg["effort"]] = cfg
        return {"avg_recall_at_5_mean": 1.0}

    monkeypatch.setattr(search_benchmark, "_run_search_eval", fake_run_search_eval)
    return configs


def test_per_run_overrides_applied(captured_configs):
    config = {
        "search_agent_name": "query-agent-search-mode",
        "max_concurrent": 3,
        "sleep_between_requests": 1,
        "effort_sweep": True,
        "effort_sweep_baselines": ["hybrid-search"],
        "effort_sweep_overrides": {
            "hybrid-search": {"max_concurrent": 5, "sleep_between_requests": 0},
            "medium": {"max_concurrent": 5, "sleep_between_requests": 0},
            "high": {"max_concurrent": 2, "sleep_between_requests": 2},
            "ultrahigh": {"max_concurrent": 2, "sleep_between_requests": 2},
        },
    }

    results = asyncio.run(search_benchmark._run_effort_sweep(config))

    assert set(results) == {"medium", "high", "ultrahigh", "hybrid"}
    assert all(entry["error"] is None for entry in results.values())

    medium = captured_configs["medium"]
    assert (medium["max_concurrent"], medium["sleep_between_requests"]) == (5, 0)

    # Baseline runs are keyed by their display label but may be configured
    # under the full agent name.
    hybrid = captured_configs["hybrid"]
    assert (hybrid["max_concurrent"], hybrid["sleep_between_requests"]) == (5, 0)
    assert hybrid["search_agent_name"] == "hybrid-search"

    for level in ("high", "ultrahigh"):
        cfg = captured_configs[level]
        assert (cfg["max_concurrent"], cfg["sleep_between_requests"]) == (2, 2)

    # The sweep-control keys must not leak into individual runs.
    for cfg in captured_configs.values():
        assert "effort_sweep" not in cfg
        assert "effort_sweep_overrides" not in cfg


def test_baselines_default_to_hybrid_search(captured_configs):
    config = {"effort_sweep": True}

    results = asyncio.run(search_benchmark._run_effort_sweep(config))

    assert set(results) == {"medium", "high", "ultrahigh", "hybrid"}
    assert captured_configs["hybrid"]["search_agent_name"] == "hybrid-search"


def test_empty_baselines_disable_baseline_runs(captured_configs):
    config = {"effort_sweep": True, "effort_sweep_baselines": []}

    results = asyncio.run(search_benchmark._run_effort_sweep(config))

    assert set(results) == {"medium", "high", "ultrahigh"}


def test_runs_without_overrides_use_shared_settings(captured_configs):
    config = {
        "search_agent_name": "query-agent-search-mode",
        "max_concurrent": 3,
        "sleep_between_requests": 1,
        "effort_sweep": True,
        "effort_sweep_overrides": {"medium": {"max_concurrent": 8}},
    }

    asyncio.run(search_benchmark._run_effort_sweep(config))

    assert captured_configs["medium"]["max_concurrent"] == 8
    assert captured_configs["medium"]["sleep_between_requests"] == 1
    for level in ("high", "ultrahigh"):
        cfg = captured_configs[level]
        assert (cfg["max_concurrent"], cfg["sleep_between_requests"]) == (3, 1)
