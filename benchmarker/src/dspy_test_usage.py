import os
import dspy

# Define a simple signature
class SimpleQA(dspy.Signature):
    """Answer a question."""
    question = dspy.InputField()
    answer = dspy.OutputField()

def main():
    # Configure DSPy with explicit usage tracking
    print("Setting up DSPy with usage tracking...")
    lm = dspy.LM('openai/gpt-4o', api_key=os.getenv("OPENAI_API_KEY"), cache=False)
    dspy.configure(lm=lm, track_usage=True)
    
    # Create a predictor
    qa = dspy.Predict(SimpleQA)
    
    # Run inference
    print("Running inference...")
    test_question = "What is the capital of France?"
    response = qa(question=test_question)
    
    # Check usage stats
    print(f"Response: {response.answer}")
    print(f"Usage tracking enabled: {dspy.settings.track_usage}")
    print(f"Usage data: {response.get_lm_usage()}")
    
    # Try an alternative way to get usage
    print("Trying alternative ways to access usage...")
    
    # Method 1: Access through dspy.settings
    print(f"Current dspy.settings: {vars(dspy.settings)}")
    
    # Method 2: Check if _lm_usage attribute exists
    if hasattr(response, '_lm_usage'):
        print(f"Response has _lm_usage attribute: {response._lm_usage}")
    else:
        print("Response does not have _lm_usage attribute")
    
    # Method 3: Check all attributes of response
    print(f"All response attributes: {dir(response)}")
    
    # Method 4: Check specific dspy metrics functionality
    print("Testing metrics API if available...")
    try:
        from dspy.primitives.metrics import get_metrics
        metrics = get_metrics(response)
        print(f"Metrics: {metrics}")
    except (ImportError, AttributeError) as e:
        print(f"Metrics API not available: {e}")

if __name__ == "__main__":
    main() 