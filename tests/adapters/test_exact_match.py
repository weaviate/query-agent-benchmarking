"""Tests for exact match metric."""

from query_agent_benchmarking.metrics.exact_match import calculate_exact_match


class TestExactMatch:
    def test_identical_strings(self):
        assert calculate_exact_match("hello world", "hello world") is True

    def test_case_insensitive_default(self):
        assert calculate_exact_match("Hello World", "hello world") is True

    def test_case_sensitive(self):
        assert calculate_exact_match("Hello", "hello", case_sensitive=True) is False

    def test_whitespace_normalization(self):
        assert calculate_exact_match("  hello   world  ", "hello world") is True

    def test_different_strings(self):
        assert calculate_exact_match("hello", "goodbye") is False

    def test_empty_strings(self):
        assert calculate_exact_match("", "") is True

    def test_multiline_whitespace_normalized(self):
        """Newlines are normalized to spaces by the whitespace normalization."""
        assert calculate_exact_match("hello\nworld", "hello world") is True
