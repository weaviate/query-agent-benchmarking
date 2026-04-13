"""
LongMemEval-specific LLM-as-Judge using type-specific prompts.

Implements the evaluation protocol from the LongMemEval paper (Wu et al., ICLR 2025).
Uses GPT-4o with separate prompt-engineered instructions per question type:
- temporal-reasoning: allows off-by-one errors for day/week/month counts
- knowledge-update: correct as long as the updated answer is present
- single-session-preference: rubric-based, just needs to recall personal info correctly
- other types (single-session-user, single-session-assistant, multi-session):
  standard containment check with credit for equivalent answers or intermediate steps

Output format: simple yes/no (no chain-of-thought reasoning).
"""

import os
from typing import Optional

from openai import OpenAI

# ============================================================================
# Type-specific prompt templates (from the LongMemEval paper, Appendix G)
# ============================================================================

_STANDARD_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL_REASONING_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer."
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not."
    "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}"
    "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
)

_TEMPLATE_MAP = {
    "single-session-user": _STANDARD_TEMPLATE,
    "single-session-assistant": _STANDARD_TEMPLATE,
    "multi-session": _STANDARD_TEMPLATE,
    "temporal-reasoning": _TEMPORAL_REASONING_TEMPLATE,
    "knowledge-update": _KNOWLEDGE_UPDATE_TEMPLATE,
    "single-session-preference": _PREFERENCE_TEMPLATE,
}


def _build_prompt(
    question_type: str,
    question: str,
    correct_answer: str,
    system_answer: str,
    question_id: Optional[str] = None,
) -> str:
    """Build the type-specific evaluation prompt.

    Args:
        question_type: The LongMemEval question category.
        question: The question text.
        correct_answer: The ground truth answer.
        system_answer: The model-generated answer to evaluate.
        question_id: Optional question ID to detect abstention questions (_abs suffix).

    Returns:
        The formatted prompt string.
    """
    is_abstention = question_id is not None and "_abs" in question_id
    if is_abstention:
        template = _ABSTENTION_TEMPLATE
    else:
        template = _TEMPLATE_MAP.get(question_type, _STANDARD_TEMPLATE)

    return template.format(
        question=question,
        answer=correct_answer,
        response=system_answer,
    )


class LongMemEvalJudge:
    """LLM-as-Judge implementing the LongMemEval evaluation protocol.

    Uses OpenAI's API directly (no DSPy) to match the paper's setup:
    - GPT-4o with temperature=0
    - Simple yes/no output (no chain-of-thought)
    - Type-specific prompts per question category

    Satisfies the LLMJudge protocol from core/ports/llm_judge.py.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_key: Optional[str] = None,
        default_question_type: str = "multi-session",
    ):
        """Initialize the LongMemEval judge.

        Args:
            model: OpenAI model ID. Defaults to gpt-4o-2024-08-06 per the paper.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            default_question_type: Fallback type when question_type is not provided.
        """
        self.model = model
        self.default_question_type = default_question_type
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _call_judge(self, prompt: str) -> bool:
        """Call the LLM judge and parse yes/no response."""
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            n=1,
        )
        content = completion.choices[0].message.content or ""
        return "yes" in content.strip().lower()

    def _get_usage(self, completion) -> tuple[int, int]:
        """Extract token usage from an OpenAI completion."""
        if completion.usage:
            return completion.usage.prompt_tokens, completion.usage.completion_tokens
        return 0, 0

    # ------------------------------------------------------------------
    # LLMJudge protocol methods
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        system_answer: str,
        correct_answer: str,
        question_type: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> bool:
        """Evaluate if system answer is correct using type-specific prompts.

        Args:
            question: The question that was asked.
            system_answer: The answer generated by the system.
            correct_answer: The ground truth answer.
            question_type: LongMemEval question category. Falls back to default_question_type.
            question_id: Optional question ID (used to detect abstention questions).

        Returns:
            True if the judge considers the answer correct, False otherwise.
        """
        qtype = question_type or self.default_question_type
        prompt = _build_prompt(qtype, question, correct_answer, system_answer, question_id)
        return self._call_judge(prompt)

    def evaluate_with_details(
        self,
        question: str,
        system_answer: str,
        correct_answer: str,
        question_type: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> dict:
        """Evaluate with detailed results including token usage.

        Args:
            question: The question that was asked.
            system_answer: The answer generated by the system.
            correct_answer: The ground truth answer.
            question_type: LongMemEval question category.
            question_id: Optional question ID (used to detect abstention questions).

        Returns:
            Dictionary with 'aligned', 'question_type', 'is_abstention',
            'input_tokens', 'output_tokens', and 'reasoning' keys.
        """
        qtype = question_type or self.default_question_type
        is_abstention = question_id is not None and "_abs" in question_id
        prompt = _build_prompt(qtype, question, correct_answer, system_answer, question_id)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            n=1,
        )

        response_text = (completion.choices[0].message.content or "").strip()
        aligned = "yes" in response_text.lower()
        input_tokens, output_tokens = self._get_usage(completion)

        return {
            "aligned": aligned,
            "question_type": qtype,
            "is_abstention": is_abstention,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "votes": 1,
            "ensemble_k": 1,
            "reasoning": None,  # LongMemEval judge uses no chain-of-thought
        }
