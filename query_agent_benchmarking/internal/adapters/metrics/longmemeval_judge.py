"""
LongMemEval-specific LLM-as-Judge using type-specific prompts.

Implements the evaluation protocol from the LongMemEval paper (Wu et al., ICLR 2025).
Uses DSPy for LLM inference with separate prompt-engineered instructions per question type:
- temporal-reasoning: allows off-by-one errors for day/week/month counts
- knowledge-update: correct as long as the updated answer is present
- single-session-preference: rubric-based, just needs to recall personal info correctly
- other types (single-session-user, single-session-assistant, multi-session):
  standard containment check with credit for equivalent answers or intermediate steps

Output format: simple yes/no (no chain-of-thought reasoning).
"""

import asyncio
import os
from typing import Optional

import dspy

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
    "\n\nIs the model response correct? "
    "First reason through each rule in the criteria step by step, then give a boolean verdict"
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
    "\n\nIs the model response correct? "
    "First reason through each rule in the criteria step by step, then give a boolean verdict"
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer. "
    "The response does not need to match the correct answer word-for-word. If the response "
    "captures the same core fact or meaning as the correct answer (e.g., 'in a shoe rack' "
    "vs. 'in a shoe rack in my closet'), it should be considered correct. Only answer no if "
    "the response gives a fundamentally different answer or misses the key updated fact."
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? "
    "First reason through each rule in the criteria step by step, then give a boolean verdict"
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? "
    "First reason through each rule in the criteria step by step, then give a boolean verdict"
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not."
    "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}"
    "\n\nDoes the model correctly identify the question as unanswerable? "
    "First reason through each rule in the criteria step by step, then give a boolean verdict"
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


class LongMemEvalJudgment(dspy.Signature):
    """Judge whether a model response is correct given type-specific evaluation criteria.
    First reason through each rule in the criteria step by step, then give a boolean verdict.
    """

    evaluation_prompt: str = dspy.InputField(
        description="The full evaluation prompt with type-specific instructions, question, correct answer, and model response."
    )
    reasoning: str = dspy.OutputField(
        description="Go through each rule in the evaluation criteria one by one. For each rule, determine whether the response satisfies it and why. Then give the overall reason for your final verdict."
    )
    judgment: bool = dspy.OutputField(
        description="True if the model response is correct, False otherwise."
    )


class LongMemEvalJudge:
    """LLM-as-Judge implementing the LongMemEval evaluation protocol.

    Uses DSPy for LLM inference with type-specific prompts per question category,
    following the paper's setup:
    - Type-specific prompts per question category
    - Optional ensemble voting: call the judge ensemble_k times, majority vote wins

    Satisfies the LLMJudge protocol from core/ports/llm_judge.py.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5.4",
        api_key: Optional[str] = None,
        default_question_type: str = "multi-session",
        cache: bool = False,
        ensemble_k: int = 1,
    ):
        """Initialize the LongMemEval judge.

        Args:
            model: LLM model ID in DSPy/litellm format (e.g. "openai/gpt-4o-2024-08-06").
            api_key: API key for the LLM provider. Falls back to OPENAI_API_KEY env var.
            default_question_type: Fallback type when question_type is not provided.
            cache: Whether to cache LLM responses. Disabled by default for evaluation.
        """
        self.model = model
        self.default_question_type = default_question_type
        self.ensemble_k = ensemble_k

        self.lm = dspy.LM(
            model,
            cache=cache,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

        self.judge = dspy.Predict(LongMemEvalJudgment)

    async def _call_judge_async(self, prompt: str) -> tuple[bool, str, int, int]:
        """Async version of _call_judge using dspy.Predict.acall()."""
        with dspy.context(lm=self.lm):
            response = await self.judge.acall(evaluation_prompt=prompt)
        aligned = bool(response.judgment)
        reasoning = response.reasoning or ""

        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.get_lm_usage()
            if usage and self.model in usage:
                input_tokens = usage[self.model].get("prompt_tokens", 0)
                output_tokens = usage[self.model].get("completion_tokens", 0)
        except Exception:
            pass

        return aligned, reasoning, input_tokens, output_tokens

    def _call_judge(self, prompt: str) -> tuple[bool, str, int, int]:
        """Call the LLM judge and parse yes/no response.

        Returns:
            Tuple of (aligned, reasoning, input_tokens, output_tokens).
        """
        with dspy.context(lm=self.lm):
            response = self.judge(evaluation_prompt=prompt)
        aligned = bool(response.judgment)
        reasoning = response.reasoning or ""

        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.get_lm_usage()
            if usage and self.model in usage:
                input_tokens = usage[self.model].get("prompt_tokens", 0)
                output_tokens = usage[self.model].get("completion_tokens", 0)
        except Exception:
            pass  # Token tracking is best-effort

        return aligned, reasoning, input_tokens, output_tokens

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
        votes = sum(self._call_judge(prompt)[0] for _ in range(self.ensemble_k))
        return votes >= self.ensemble_k / 2

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

        vote_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for _ in range(self.ensemble_k):
            aligned_i, reasoning_i, in_tok, out_tok = self._call_judge(prompt)
            vote_results.append({"vote": aligned_i, "reasoning": reasoning_i})
            total_input_tokens += in_tok
            total_output_tokens += out_tok

        votes = sum(v["vote"] for v in vote_results)
        majority_aligned = votes >= self.ensemble_k / 2

        return {
            "aligned": majority_aligned,
            "question_type": qtype,
            "is_abstention": is_abstention,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "votes": votes,
            "ensemble_k": self.ensemble_k,
            "reasoning": vote_results,
        }

    async def evaluate_with_details_async(
        self,
        question: str,
        system_answer: str,
        correct_answer: str,
        question_type: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> dict:
        """Async version of evaluate_with_details. Runs all ensemble_k calls concurrently."""
        qtype = question_type or self.default_question_type
        is_abstention = question_id is not None and "_abs" in question_id
        prompt = _build_prompt(qtype, question, correct_answer, system_answer, question_id)

        call_results = await asyncio.gather(*[
            self._call_judge_async(prompt) for _ in range(self.ensemble_k)
        ])

        vote_results = [{"vote": a, "reasoning": r} for a, r, _, _ in call_results]
        votes = sum(v["vote"] for v in vote_results)

        return {
            "aligned": votes >= self.ensemble_k / 2,
            "question_type": qtype,
            "is_abstention": is_abstention,
            "input_tokens": sum(c[2] for c in call_results),
            "output_tokens": sum(c[3] for c in call_results),
            "votes": votes,
            "ensemble_k": self.ensemble_k,
            "reasoning": vote_results,
        }
