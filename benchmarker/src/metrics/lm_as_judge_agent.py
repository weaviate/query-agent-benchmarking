import dspy
from pydantic import BaseModel, ValidationError

class LMJudgeResult(BaseModel):
    reasoning: str
    rating: float

class LMJudgeSignature(dspy.Signature):
    """Evaluate how well a system response answers a given question on a scale of 1-5."""
    
    question = dspy.InputField(desc="The question that was asked")
    system_response = dspy.InputField(desc="The system's response to evaluate")
    reasoning = dspy.OutputField(desc="Detailed reasoning for the rating")
    rating = dspy.OutputField(desc="Rating from 1 (terrible) to 5 (fantastic)")

class LMJudgeAgent(dspy.Module):
    def __init__(self, retries: int = 3):
        super().__init__()
        self.retries = retries
        self.judge = dspy.ChainOfThought(LMJudgeSignature)
    
    def validate_rating(self, rating_str: str) -> float:
        """Validate that the rating is between 1 and 5."""
        try:
            rating = float(rating_str)
            if rating < 1 or rating > 5:
                raise ValueError(f"Rating must be between 1 and 5, got {rating}")
            return rating
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid rating format: {rating_str}") from e
    
    def forward(self, question: str, system_response: str) -> LMJudgeResult:
        """Run the LM judge evaluation with retries."""
        
        for attempt in range(self.retries):
            try:
                # Run the judge
                prediction = self.judge(
                    question=question,
                    system_response=system_response
                )
                
                # Validate and convert rating
                rating = self.validate_rating(prediction.rating)
                
                # Create and validate result
                result = LMJudgeResult(
                    reasoning=prediction.reasoning,
                    rating=rating
                )
                
                return result
                
            except (ValueError, ValidationError) as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Failed after {self.retries} attempts. Last error: {e}")
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                continue
        
        raise RuntimeError(f"Failed after {self.retries} attempts")

lm_as_judge_agent = LMJudgeAgent()

def main():
    # Configure DSPy with OpenAI GPT-4
    lm = dspy.OpenAI(model="gpt-4o")
    dspy.settings.configure(lm=lm)
    
    # Create the agent
    agent = LMJudgeAgent(retries=3)
    
    # Example usage
    question = "What is the capital of France?"
    system_response = "Paris is the capital of France."
    
    try:
        result = agent(question=question, system_response=system_response)
        
        print("LMJudge Agent Result:")
        print(f"Reasoning: {result.reasoning}")
        print(f"Rating: {result.rating}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()