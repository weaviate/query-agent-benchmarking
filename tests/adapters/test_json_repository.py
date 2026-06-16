"""Tests for the JSON file result repository."""

import json
import pytest
from unittest.mock import patch

from query_agent_benchmarking.internal.adapters.results.json_file_repository import (
    JsonFileResultRepository,
)
from query_agent_benchmarking.internal.core.domain.models import (
    AgentSearch,
    InMemoryQuery,
    QueryResult,
    ObjectID,
    InMemoryAskQuery,
    AskResult,
)


@pytest.fixture
def repo_with_tmp_dir(tmp_path):
    """Create a repository that saves to a temp directory."""
    repo = JsonFileResultRepository()
    with patch("query_agent_benchmarking.internal.adapters.results.serialization.RESULTS_DIR", tmp_path):
        yield repo, tmp_path


@pytest.fixture
def sample_config():
    return {
        "dataset_identifier": "beir/scifact",
        "agent_name": "test-agent",
        "num_trials": 1,
    }


class TestJsonFileResultRepository:
    def test_save_trial_results(self, repo_with_tmp_dir, sample_config):
        repo, tmp_path = repo_with_tmp_dir
        query = InMemoryQuery(question="Q1", dataset_ids=["d1"])
        results = [
            QueryResult(
                query=query,
                query_ground_truth_id=["d1"],
                retrieved_ids=[ObjectID(object_id="d1"), ObjectID(object_id="d2")],
                time_taken=0.5,
            )
        ]

        repo.save_trial_results(results, sample_config, trial_number=1)

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["metadata"]["mode"] == "search"
        assert len(data["queries"]) == 1
        assert data["queries"][0]["question"] == "Q1"
        # Agents that don't report a search plan (searches=None) omit the keys
        # entirely, so the serialized shape is unchanged from before the feature.
        assert "num_searches" not in data["queries"][0]
        assert "searches" not in data["queries"][0]

    def test_save_trial_results_with_search_plan(self, repo_with_tmp_dir, sample_config):
        repo, tmp_path = repo_with_tmp_dir
        query = InMemoryQuery(question="red cars under 30k", dataset_ids=["d1"])
        results = [
            QueryResult(
                query=query,
                query_ground_truth_id=["d1"],
                retrieved_ids=[ObjectID(object_id="d1")],
                time_taken=0.5,
                searches=[
                    AgentSearch(
                        collection="FilteredCars",
                        query="red cars",
                        filters={
                            "combine": "AND",
                            "filters": [
                                {
                                    "filter_type": "integer",
                                    "property_name": "price",
                                    "operator": "<",
                                    "value": 30000,
                                }
                            ],
                        },
                    )
                ],
            )
        ]

        repo.save_trial_results(results, sample_config, trial_number=1)

        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text())
        q = data["queries"][0]
        assert q["num_searches"] == 1
        assert q["searches"][0]["collection"] == "FilteredCars"
        assert q["searches"][0]["query"] == "red cars"
        assert q["searches"][0]["filters"]["combine"] == "AND"
        assert q["searches"][0]["filters"]["filters"][0]["property_name"] == "price"

    def test_save_trial_metrics(self, repo_with_tmp_dir, sample_config):
        repo, tmp_path = repo_with_tmp_dir
        metrics = {"avg_recall_at_5": 0.85, "avg_query_time": 0.3}

        repo.save_trial_metrics(metrics, sample_config, trial_number=1)

        files = list(tmp_path.glob("*metrics.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["avg_recall_at_5"] == 0.85

    def test_save_aggregated_results(self, repo_with_tmp_dir, sample_config):
        repo, tmp_path = repo_with_tmp_dir
        aggregated = {"num_trials": 3, "avg_recall_at_5_mean": 0.82}

        repo.save_aggregated_results(aggregated, sample_config)

        files = list(tmp_path.glob("*results.json"))
        assert len(files) == 1

    def test_save_ask_trial_results(self, repo_with_tmp_dir, sample_config):
        repo, tmp_path = repo_with_tmp_dir
        query = InMemoryAskQuery(question="Q1", ground_truth_answer="A1")
        results = [
            AskResult(query=query, system_answer="A1 answer", time_taken=1.0)
        ]

        repo.save_ask_trial_results(results, sample_config, trial_number=1, alignment_scores=[1])

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["metadata"]["mode"] == "ask"
