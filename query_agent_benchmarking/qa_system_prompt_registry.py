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
    "officeqa": (
        "Respond with only the final numerical or short factual answer. "
        "Do not include explanations, units descriptions, or full sentences. "
        "For numerical answers, use the same format as the source document "
        "(e.g., '2,602' for thousands-separated numbers, '1608.80%' for percentages). "
        "Wrap your answer in <FINAL_ANSWER> tags, e.g. <FINAL_ANSWER>2,602</FINAL_ANSWER>."
    ),
}


def get_system_prompt(dataset_name: str) -> str | None:
    """Look up the system prompt for a dataset, or return None if not registered."""
    return SYSTEM_PROMPTS.get(dataset_name)
