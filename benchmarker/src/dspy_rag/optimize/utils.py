from typing import Dict, Any
import os
import yaml
import dspy
import weaviate

def load_optimization_config(config_path: str) -> Dict[str, Any]:
    """
    Load and process configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with processed configuration
        
    Raises:
        ValueError: If file not found or YAML parsing fails
    """
    try:
        # Load YAML configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Apply quick test configuration if enabled
        if config.get("quick_test", {}).get("enabled", False):
            quick_config = config["quick_test"]
            for key, value in quick_config.items():
                if key != "enabled":
                    config[key] = value
            print("Applied quick test configuration")
            
        return config
        
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def setup_dspy():
    """Configure DSPy."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")


def setup_weaviate():
    """Set up Weaviate client connection."""
    cluster_url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not cluster_url or not api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables are required")
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )
    
    print(f"Connected to Weaviate cluster: {cluster_url}")
    return client