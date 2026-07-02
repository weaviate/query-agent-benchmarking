"""
Query execution logic for both search and ask modes.

Contains the core iteration, batching, semaphore concurrency, and timing
logic. Depends only on port interfaces, not on concrete implementations.
"""

import asyncio
import time
from typing import Any, Optional

from tqdm import tqdm

from query_agent_benchmarking.internal.core.domain.models import (
    AgentSearch,
    InMemoryQuery,
    InMemoryAskQuery,
    ObjectID,
    QueryResult,
    SearchAgentResponse,
    AskResult,
)


# ============================================================================
# Search Mode Query Execution
# ============================================================================

def _unpack_search_response(
    response: object,
) -> tuple[list[ObjectID], Optional[list[AgentSearch]]]:
    """Normalize a search agent's return value into (retrieved_ids, searches).

    Agents may return either a plain ``list[ObjectID]`` (legacy / BYOS /
    direct-search adapters) or a ``SearchAgentResponse`` that also carries the
    structured searches the agent performed. ``searches`` is ``None`` when the
    agent does not report a search plan.
    """
    if isinstance(response, SearchAgentResponse):
        return response.retrieved_ids, response.searches
    return response, None  # type: ignore[return-value]

def run_search_queries(
    queries: list[InMemoryQuery],
    query_agent: Any,
) -> list[QueryResult]:
    """Synchronous search query execution."""
    results = []
    start = time.time()
    for i, query in enumerate(tqdm(queries, desc="Running search queries")):
        query_start_time = time.time()
        stringified_ids = [str(dataset_id) for dataset_id in query.dataset_ids]
        response = query_agent.run(query.question, tenant=query.tenant_id)
        retrieved_ids, searches = _unpack_search_response(response)
        query_time_taken = time.time() - query_start_time

        results.append(QueryResult(
            query=query,
            query_ground_truth_id=stringified_ids,
            retrieved_ids=retrieved_ids,
            time_taken=query_time_taken,
            searches=searches,
        ))

        if i % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i}/{len(queries)}) ---\033[0m")
            print(f"Latest query: {query.question}")
            print(f"Ground truth: {query.dataset_ids}")
            print(f"Latest response: {results[i].retrieved_ids}")
            print(f"Time taken: {query_time_taken:.2f} seconds")

    print(f"\033[95mExperiment completed {len(results)} queries in {time.time() - start:.2f} seconds.\033[0m")
    return results


async def run_search_queries_async(
    queries: list[InMemoryQuery],
    query_agent: Any,
    batch_size: int = 10,
    max_concurrent: int = 3,
) -> list[QueryResult]:
    """Asynchronous search query execution with batching and concurrency control."""
    results = []
    start = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_query(query, index, retry_count=0, max_retries=3):
        async with semaphore:
            query_start_time = time.time()
            stringified_ids = [str(dataset_id) for dataset_id in query.dataset_ids]
            try:
                if retry_count > 0:
                    delay = min(2 ** retry_count, 10)
                    print(f"\nRetrying query {index} (attempt {retry_count + 1}) after {delay}s delay...")
                    await asyncio.sleep(delay)
                elif index > 0:
                    await asyncio.sleep(0.1)

                print(f"Running search query {index}: {query.question}")
                response = await query_agent.run_async(query.question, tenant=query.tenant_id)
                retrieved_ids, searches = _unpack_search_response(response)
                query_time_taken = time.time() - query_start_time

                return QueryResult(
                    query=query,
                    query_ground_truth_id=stringified_ids,
                    retrieved_ids=retrieved_ids,
                    time_taken=query_time_taken,
                    searches=searches,
                )
            except Exception as e:
                query_time_taken = time.time() - query_start_time
                print(f"\n\033[91mError processing query {index}: {str(e)}\033[0m")
                return QueryResult(
                    query=query,
                    query_ground_truth_id=stringified_ids,
                    retrieved_ids=[],
                    time_taken=query_time_taken,
                )

    total_batches = (len(queries) + batch_size - 1) // batch_size
    print(f"\033[94mProcessing {len(queries)} search queries in {total_batches} batches "
          f"(batch_size={batch_size}, max_concurrent={max_concurrent})\033[0m")

    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        print(f"\nStarting batch {batch_idx // batch_size + 1}/{total_batches}")

        tasks = [process_query(query, batch_idx + i) for i, query in enumerate(batch)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        batch_time = time.time() - batch_start
        completed = min(batch_idx + batch_size, len(queries))
        batch_successes = sum(1 for r in batch_results if "error" not in r)
        batch_errors = len(batch_results) - batch_successes

        print(f"\n\033[93m--- Batch {batch_idx // batch_size + 1} Complete ({completed}/{len(queries)}) ---\033[0m")
        print(f"Successes: {batch_successes}, Errors: {batch_errors}")
        print(f"Batch completed in {batch_time:.2f} seconds")

        successful_results = [r for r in batch_results if "error" not in r]
        if successful_results:
            sample = successful_results[0]
            print(f"Sample query: {sample.query.question[:100]}...")
            print(f"Sample response: {sample.retrieved_ids[:200]}...")

        if batch_idx + batch_size < len(queries):
            await asyncio.sleep(1)

    total_time = time.time() - start
    total_successes = sum(1 for r in results if "error" not in r)
    total_errors = len(results) - total_successes

    print("\n\033[95mAsync search experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")

    return results


# ============================================================================
# Ask Mode Query Execution
# ============================================================================

def run_ask_queries(
    queries: list[InMemoryAskQuery],
    ask_agent: Any,
    sleep_between_requests: float = 0.0,
) -> list[AskResult]:
    """Synchronous ask query execution."""
    results = []
    start = time.time()

    for i, query in enumerate(tqdm(queries, desc="Running ask queries")):
        query_start_time = time.time()
        try:
            response = ask_agent.run(
                query=query.question,
                oracle_context_id=query.oracle_context_id,
                tenant_id=query.tenant_id,
                question_date=query.question_date,
            )
            query_time_taken = time.time() - query_start_time
            results.append(AskResult(
                query=query,
                system_answer=response.final_answer,
                time_taken=query_time_taken,
                retrieved_context=response.raw_response,
            ))
        except Exception as e:
            query_time_taken = time.time() - query_start_time
            print(f"\n\033[91mError processing ask query {i}: {str(e)}\033[0m")
            results.append(AskResult(
                query=query,
                system_answer=f"[ERROR] {str(e)}",
                time_taken=query_time_taken,
            ))

        if i % 10 == 0:
            print(f"\n\033[93m--- Progress Update ({i}/{len(queries)}) ---\033[0m")
            print(f"Latest query: {query.question[:100]}...")
            print(f"Latest answer: {results[i].system_answer[:200]}...")
            print(f"Time taken: {query_time_taken:.2f} seconds")

        if sleep_between_requests > 0 and i < len(queries) - 1:
            time.sleep(sleep_between_requests)

    print(f"\033[95mAsk experiment completed {len(results)} queries in {time.time() - start:.2f} seconds.\033[0m")
    return results


async def run_ask_queries_async(
    queries: list[InMemoryAskQuery],
    ask_agent: Any,
    batch_size: int = 10,
    max_concurrent: int = 3,
) -> list[AskResult]:
    """Asynchronous ask query execution with batching and concurrency control."""
    results = []
    start = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_query(query: InMemoryAskQuery, index: int):
        async with semaphore:
            query_start_time = time.time()
            try:
                if index > 0:
                    await asyncio.sleep(0.1)
                print(f"Running ask query {index}: {query.question[:50]}...")
                response = await ask_agent.run_async(
                    query=query.question,
                    oracle_context_id=query.oracle_context_id,
                    tenant_id=query.tenant_id,
                    question_date=query.question_date,
                )
                query_time_taken = time.time() - query_start_time
                return AskResult(
                    query=query,
                    system_answer=response.final_answer,
                    time_taken=query_time_taken,
                    retrieved_context=response.raw_response,
                )
            except Exception as e:
                query_time_taken = time.time() - query_start_time
                print(f"\n\033[91mError processing ask query {index}: {str(e)}\033[0m")
                return AskResult(
                    query=query,
                    system_answer=f"[ERROR] {str(e)}",
                    time_taken=query_time_taken,
                )

    total_batches = (len(queries) + batch_size - 1) // batch_size
    print(f"\033[94mProcessing {len(queries)} ask queries in {total_batches} batches "
          f"(batch_size={batch_size}, max_concurrent={max_concurrent})\033[0m")

    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        print(f"\nStarting batch {batch_idx // batch_size + 1}/{total_batches}")

        tasks = [process_query(query, batch_idx + i) for i, query in enumerate(batch)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        batch_time = time.time() - batch_start
        completed = min(batch_idx + batch_size, len(queries))
        batch_errors = sum(1 for r in batch_results if r.system_answer.startswith("[ERROR]"))
        batch_successes = len(batch_results) - batch_errors

        print(f"\n\033[93m--- Batch {batch_idx // batch_size + 1} Complete ({completed}/{len(queries)}) ---\033[0m")
        print(f"Successes: {batch_successes}, Errors: {batch_errors}")
        print(f"Batch completed in {batch_time:.2f} seconds")

        if batch_idx + batch_size < len(queries):
            await asyncio.sleep(1)

    total_time = time.time() - start
    total_errors = sum(1 for r in results if r.system_answer.startswith("[ERROR]"))
    total_successes = len(results) - total_errors

    print("\n\033[95mAsync ask experiment completed!\033[0m")
    print(f"\033[95mResults: {total_successes} successful, {total_errors} failed out of {len(results)} total\033[0m")
    print(f"\033[95mTotal time: {total_time:.2f} seconds\033[0m")
    print(f"\033[95mAverage time per query: {total_time/len(results):.2f} seconds\033[0m")

    return results
