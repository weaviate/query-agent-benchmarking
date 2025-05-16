from __future__ import annotations

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

RETRIES = 3

class LMJudgeAgentDeps:
    def __init__(
            self,
            question: str,
            system_response: str
    ):
        self.question = question
        self.system_response = system_response

    def build_prompt(self) -> str:
        return f"""
        Please evaluate how well the `system_response` answers the question.
        Please report your assessment as a rating on a scale of 1 to 5, with 1 denoting a terrible answer and 5 denoting a fantastic answer.

        `question`
        {self.question}

        `system_response`
        {self.system_response}
        """

class LMJudgeResult(BaseModel):
    reasoning: str
    rating: float
    
lm_as_judge_agent = Agent(
    deps_type=LMJudgeAgentDeps,
    result_type=LMJudgeResult,
    retries=RETRIES,
    result_tool_name="lm_as_judge_result",
)

@lm_as_judge_agent.system_prompt
async def system_prompt(ctx: RunContext[LMJudgeAgentDeps]) -> str:
    return ctx.deps.build_prompt()

@lm_as_judge_agent.result_validator
async def validate_rating(
    ctx: RunContext[LMJudgeAgentDeps], result: LMJudgeResult
):
    """Validate that the rating is in between 1 and 5."""
    if result.rating < 1 or result.rating > 5:
        raise ModelRetry(f"Rating must be between 1 and 5, got {result.rating}")
    return result

async def main():
    deps = LMJudgeAgentDeps(
        question="What is the capital of France?",
        system_response="Paris is the capital of France."
    )

    result = await lm_as_judge_agent.run(
        deps=deps,
        model="openai:gpt-4o",
    )

    print("LMJudge Agent Result:")
    print(result.data)


if __name__ == "__main__":
    asyncio.run(main())