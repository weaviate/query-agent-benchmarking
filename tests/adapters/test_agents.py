"""Tests for agent adapters."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from query_agent_benchmarking.internal.core.domain.models import ObjectID
from query_agent_benchmarking.internal.core.ports.ask_agent import AskResponse
from query_agent_benchmarking.internal.core.ports.search_agent import SearchAgent
from query_agent_benchmarking.internal.core.ports.ask_agent import AskAgent
from query_agent_benchmarking.internal.adapters.agents.external_service import (
    ExternalSearchService,
    ExternalAskService,
)
from query_agent_benchmarking.internal.adapters.agents.weaviate_search import (
    parse_agent_name,
    parse_target_vector_names,
)


# ============================================================================
# parse_agent_name tests
# ============================================================================


class TestParseAgentName:
    def test_simple_name(self):
        assert parse_agent_name("hybrid-search") == ("hybrid-search", None)

    def test_name_with_target_vector(self):
        assert parse_agent_name("hybrid-search[text_content_weaviate]") == (
            "hybrid-search",
            "text_content_weaviate",
        )

    def test_name_with_multi_target_vector(self):
        base, config = parse_agent_name(
            "hybrid-search[text_content_weaviate+image_content_weaviate]"
        )
        assert base == "hybrid-search"
        assert config == "text_content_weaviate+image_content_weaviate"

    def test_vector_search_with_target(self):
        assert parse_agent_name("vector-search[my_vec]") == ("vector-search", "my_vec")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_agent_name("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_agent_name("   ")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid agent_name format"):
            parse_agent_name("hybrid-search[")

    def test_nested_brackets_raises(self):
        with pytest.raises(ValueError, match="Invalid agent_name format"):
            parse_agent_name("hybrid-search[[vec]]")

    def test_strips_whitespace(self):
        assert parse_agent_name("  hybrid-search  ") == ("hybrid-search", None)


# ============================================================================
# parse_target_vector_names tests
# ============================================================================


class TestParseTargetVectorNames:
    def test_single(self):
        assert parse_target_vector_names("text_content_weaviate") == [
            "text_content_weaviate"
        ]

    def test_multiple(self):
        names = parse_target_vector_names("text_content_weaviate+image_content_weaviate")
        assert names == ["text_content_weaviate", "image_content_weaviate"]

    def test_strips_whitespace(self):
        assert parse_target_vector_names(" a + b ") == ["a", "b"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Invalid target vector config"):
            parse_target_vector_names("")

    def test_trailing_plus_raises(self):
        with pytest.raises(ValueError, match="Invalid target vector config"):
            parse_target_vector_names("a+")

    def test_duplicates_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            parse_target_vector_names("vec+vec")


# ============================================================================
# ExternalSearchService tests
# ============================================================================


class TestExternalSearchService:
    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="host is required"):
            ExternalSearchService(host="")

    def test_run_sync(self):
        svc = ExternalSearchService(host="http://localhost:8080/search")

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": ["doc1", "doc2", "doc3"]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            response = svc.run("test query")

        results = response.retrieved_ids
        assert len(results) == 3
        assert all(isinstance(r, ObjectID) for r in results)
        assert results[0].object_id == "doc1"
        # No search plan reported by the external service.
        assert response.searches is None
        mock_client.post.assert_called_once_with(
            "http://localhost:8080/search", json={"query": "test query"}
        )

    def test_run_empty_results(self):
        svc = ExternalSearchService(host="http://localhost:8080/search")

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            response = svc.run("test query")

        assert response.retrieved_ids == []
        assert response.searches is None

    @pytest.mark.asyncio
    async def test_run_async(self):
        svc = ExternalSearchService(host="http://localhost:8080/search")

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": ["a", "b"]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            response = await svc.run_async("async query")

        results = response.retrieved_ids
        assert len(results) == 2
        assert results[0].object_id == "a"
        assert response.searches is None

    def test_conforms_to_search_agent_protocol(self):
        svc = ExternalSearchService(host="http://example.com")
        assert isinstance(svc, SearchAgent)


# ============================================================================
# ExternalAskService tests
# ============================================================================


class TestExternalAskService:
    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="host is required"):
            ExternalAskService(host="")

    def test_run_sync(self):
        svc = ExternalAskService(host="http://localhost:8080/ask")

        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": "The answer is 42"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = svc.run("What is the answer?")

        assert isinstance(result, AskResponse)
        assert result.final_answer == "The answer is 42"
        mock_client.post.assert_called_once_with(
            "http://localhost:8080/ask",
            json={"question": "What is the answer?"},
        )

    def test_run_with_oracle_context_id(self):
        svc = ExternalAskService(host="http://localhost:8080/ask")

        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": "result"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            svc.run("query", oracle_context_id="ctx-123")

        mock_client.post.assert_called_once_with(
            "http://localhost:8080/ask",
            json={"question": "query", "oracle_context_id": "ctx-123"},
        )

    def test_run_missing_answer_returns_empty(self):
        svc = ExternalAskService(host="http://localhost:8080/ask")

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = svc.run("query")

        assert result.final_answer == ""

    def test_conforms_to_ask_agent_protocol(self):
        svc = ExternalAskService(host="http://example.com")
        assert isinstance(svc, AskAgent)


# ============================================================================
# Collection resolver tests
# ============================================================================


class TestCollectionResolver:
    def test_both_raises(self):
        from query_agent_benchmarking.internal.adapters.agents.collection_resolver import (
            resolve_collection_info,
        )
        from query_agent_benchmarking.internal.core.domain.models import DocsCollection

        with pytest.raises(ValueError, match="Cannot specify both"):
            resolve_collection_info(
                dataset_name="enron",
                docs_collection=DocsCollection(
                    collection_name="MyCol", content_key="text", id_key="my_id"
                ),
            )

    def test_neither_raises(self):
        from query_agent_benchmarking.internal.adapters.agents.collection_resolver import (
            resolve_collection_info,
        )

        with pytest.raises(ValueError, match="Must specify either"):
            resolve_collection_info()

    def test_custom_collection(self):
        from query_agent_benchmarking.internal.adapters.agents.collection_resolver import (
            resolve_collection_info,
        )
        from query_agent_benchmarking.internal.core.domain.models import DocsCollection

        info = resolve_collection_info(
            docs_collection=DocsCollection(
                collection_name="CustomDocs", content_key="text", id_key="doc_id"
            )
        )
        assert info["collection"] == "CustomDocs"
        assert info["id_property"] == "doc_id"
        assert info["uses_named_vectors"] is False
