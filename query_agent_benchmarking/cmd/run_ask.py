"""Thin entry point for ask benchmarks — delegates to the service layer."""

from query_agent_benchmarking.internal.core.services.ask_benchmark import run_ask_eval

__all__ = ["run_ask_eval"]
