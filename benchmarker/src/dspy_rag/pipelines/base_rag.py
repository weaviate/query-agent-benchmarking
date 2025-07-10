import abc

import dspy

from benchmarker.src.dspy_rag.models import DSPyAgentRAGResponse

class BaseRAG(dspy.Module):
    def __init__(self, collection_name: str, target_property_name: str) -> None:
        self.collection_name = collection_name
        self.target_property_name = target_property_name

    @staticmethod
    def _merge_usage(*usages: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for usage in usages:
            # Skip None values
            if usage is None:
                continue
            for lm_id, stats in usage.items():
                bucket = merged.setdefault(
                    lm_id, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["prompt_tokens"] += stats.get("prompt_tokens", 0)
                bucket["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    @abc.abstractmethod
    def forward(self, question: str) -> DSPyAgentRAGResponse: ...
    
    @abc.abstractmethod
    async def aforward(self, question: str) -> DSPyAgentRAGResponse: ...