"""
System prompt registry for ask mode benchmarks.

Maps dataset names to system prompts that instruct the Query Agent
to produce answers in the format expected by the evaluation metric.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "multihoprag": (
        "Respond with only a short, succinct factual answer. "
        "Do not include explanations, context, or full sentences. "
        "For example: 'Sam Bankman-Fried', 'yes', 'before', or 'insufficient information'."
    ),
}


def get_system_prompt(dataset_name: str) -> str | None:
    """Look up the system prompt for a dataset, or return None if not registered."""
    return SYSTEM_PROMPTS.get(dataset_name)
