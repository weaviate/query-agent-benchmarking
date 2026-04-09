"""Tests for IR metric functions (pure functions, no external dependencies)."""

import pytest

from query_agent_benchmarking.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
    calculate_nDCG_at_k,
    calculate_coverage,
    calculate_alpha_ndcg,
)
from query_agent_benchmarking.domain.models import NuggetInfo


# ============================================================================
# Recall@K Tests
# ============================================================================

class TestRecallAtK:
    def test_perfect_recall(self):
        target_ids = ["doc1", "doc2", "doc3"]
        retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=5) == 1.0

    def test_partial_recall(self):
        target_ids = ["doc1", "doc2", "doc3", "doc4"]
        retrieved_ids = ["doc1", "doc5", "doc2", "doc6", "doc7"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=5) == 0.5

    def test_zero_recall(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc3", "doc4", "doc5"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=3) == 0.0

    def test_recall_at_1_is_binary(self):
        """Recall@1 returns found_count (0 or 1), not proportion."""
        target_ids = ["doc1", "doc2", "doc3"]
        retrieved_ids = ["doc1", "doc4", "doc5"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=1) == 1

    def test_recall_at_1_miss(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc3", "doc4"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=1) == 0

    def test_empty_retrieved(self):
        target_ids = ["doc1", "doc2"]
        assert calculate_recall_at_k(target_ids, [], k=5) == 0.0

    def test_empty_target(self):
        """Edge case: empty target set causes ZeroDivisionError."""
        retrieved_ids = ["doc1", "doc2"]
        with pytest.raises(ZeroDivisionError):
            calculate_recall_at_k([], retrieved_ids, k=5)

    def test_k_larger_than_retrieved(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc1"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=10) == 0.5

    def test_string_coercion(self):
        """IDs should be coerced to strings for comparison."""
        target_ids = [1, 2, 3]
        retrieved_ids = ["1", "2", "4"]
        assert calculate_recall_at_k(target_ids, retrieved_ids, k=3) == pytest.approx(2 / 3)


# ============================================================================
# Success@K Tests
# ============================================================================

class TestSuccessAtK:
    def test_hit(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc3", "doc1", "doc4"]
        assert calculate_success_at_k(target_ids, retrieved_ids, k=3) == 1

    def test_miss(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc3", "doc4", "doc5"]
        assert calculate_success_at_k(target_ids, retrieved_ids, k=3) == 0

    def test_hit_at_exact_k(self):
        target_ids = ["doc1"]
        retrieved_ids = ["doc2", "doc3", "doc1"]
        assert calculate_success_at_k(target_ids, retrieved_ids, k=3) == 1

    def test_hit_beyond_k(self):
        target_ids = ["doc1"]
        retrieved_ids = ["doc2", "doc3", "doc4", "doc1"]
        assert calculate_success_at_k(target_ids, retrieved_ids, k=3) == 0

    def test_empty_retrieved(self):
        assert calculate_success_at_k(["doc1"], [], k=5) == 0


# ============================================================================
# nDCG@K Tests
# ============================================================================

class TestNDCGAtK:
    def test_perfect_ranking(self):
        target_ids = ["doc1", "doc2", "doc3"]
        retrieved_ids = ["doc1", "doc2", "doc3"]
        assert calculate_nDCG_at_k(target_ids, retrieved_ids, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        target_ids = ["doc1"]
        retrieved_ids = ["doc2", "doc3", "doc4"]
        assert calculate_nDCG_at_k(target_ids, retrieved_ids, k=3) == 0.0

    def test_partial_ranking(self):
        target_ids = ["doc1", "doc2"]
        retrieved_ids = ["doc3", "doc1", "doc2"]
        result = calculate_nDCG_at_k(target_ids, retrieved_ids, k=3)
        assert 0.0 < result < 1.0

    def test_empty_retrieved(self):
        assert calculate_nDCG_at_k(["doc1"], [], k=5) == 0.0

    def test_single_relevant_at_top(self):
        target_ids = ["doc1"]
        retrieved_ids = ["doc1", "doc2", "doc3"]
        assert calculate_nDCG_at_k(target_ids, retrieved_ids, k=3) == pytest.approx(1.0)


# ============================================================================
# Coverage@K Tests
# ============================================================================

class TestCoverageAtK:
    def test_full_coverage(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc1"]),
            NuggetInfo(nugget_id="n2", text="t2", relevant_corpus_ids=["doc2"]),
        ]
        retrieved_ids = ["doc1", "doc2", "doc3"]
        assert calculate_coverage(retrieved_ids, nugget_data, k=3) == 1.0

    def test_partial_coverage(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc1"]),
            NuggetInfo(nugget_id="n2", text="t2", relevant_corpus_ids=["doc4"]),
        ]
        retrieved_ids = ["doc1", "doc2", "doc3"]
        assert calculate_coverage(retrieved_ids, nugget_data, k=3) == 0.5

    def test_no_coverage(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc4"]),
        ]
        retrieved_ids = ["doc1", "doc2"]
        assert calculate_coverage(retrieved_ids, nugget_data, k=2) == 0.0

    def test_empty_nugget_data(self):
        assert calculate_coverage(["doc1"], [], k=5) == 0.0

    def test_coverage_respects_k(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc3"]),
        ]
        retrieved_ids = ["doc1", "doc2", "doc3"]
        assert calculate_coverage(retrieved_ids, nugget_data, k=2) == 0.0
        assert calculate_coverage(retrieved_ids, nugget_data, k=3) == 1.0


# ============================================================================
# Alpha-nDCG@K Tests
# ============================================================================

class TestAlphaNDCG:
    def test_perfect_diverse_ranking(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc1"]),
            NuggetInfo(nugget_id="n2", text="t2", relevant_corpus_ids=["doc2"]),
        ]
        retrieved_ids = ["doc1", "doc2"]
        result = calculate_alpha_ndcg(retrieved_ids, nugget_data, alpha=0.5, k=2)
        assert result == pytest.approx(1.0)

    def test_no_relevant_docs(self):
        nugget_data = [
            NuggetInfo(nugget_id="n1", text="t1", relevant_corpus_ids=["doc5"]),
        ]
        retrieved_ids = ["doc1", "doc2"]
        assert calculate_alpha_ndcg(retrieved_ids, nugget_data, k=2) == 0.0

    def test_empty_inputs(self):
        assert calculate_alpha_ndcg([], [], k=5) == 0.0
        nugget_data = [NuggetInfo(nugget_id="n1", text="t", relevant_corpus_ids=["d1"])]
        assert calculate_alpha_ndcg([], nugget_data, k=5) == 0.0
