#!/usr/bin/env python3
"""
DSPy Optimization Runner for Weaviate Query Agent Benchmarker

This script runs DSPy optimization on RAG programs using the benchmarker's evaluation infrastructure.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

import dspy
import weaviate

from benchmarker.src.dspy_rag.optimization import (
    OptimizationConfig,
    DSPyOptimizer
)


def load_optimization_config(config_path: str) -> Dict[str, Any]:
    """Load optimization configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def apply_quick_test_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply quick test configuration if enabled."""
    if config.get("quick_test", {}).get("enabled", False):
        quick_config = config["quick_test"]
        for key, value in quick_config.items():
            if key != "enabled":
                config[key] = value
        print("Applied quick test configuration")
    return config


def create_optimization_config_from_dict(config_dict: Dict[str, Any]) -> OptimizationConfig:
    """Create OptimizationConfig from dictionary."""
    return OptimizationConfig(
        dataset_name=config_dict["dataset_name"],
        agent_name=config_dict["agent_name"],
        optimizer_type=config_dict["optimizer_type"],
        metric_type=config_dict["metric_type"],
        train_ratio=config_dict["train_ratio"],
        max_train_samples=config_dict.get("max_train_samples"),
        max_dev_samples=config_dict.get("max_dev_samples"),
        max_bootstrapped_demos=config_dict["max_bootstrapped_demos"],
        max_labeled_demos=config_dict["max_labeled_demos"],
        num_candidate_programs=config_dict["num_candidate_programs"],
        num_threads=config_dict["num_threads"],
        recall_weight=config_dict["recall_weight"],
        lm_judge_weight=config_dict["lm_judge_weight"],
        lm_judge_model=config_dict["lm_judge_model"],
        save_optimized_program=config_dict["save_optimized_program"],
        output_dir=config_dict["output_dir"],
        experiment_name=config_dict.get("experiment_name")
    )


def setup_dspy():
    """Configure DSPy."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4o", api_key=openai_api_key)
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


def print_configuration_summary(config: Dict[str, Any]):
    """Print a summary of the optimization configuration."""
    print("=" * 60)
    print("DSPy Optimization Configuration")
    print("=" * 60)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Agent: {config['agent_name']}")
    print(f"Optimizer: {config['optimizer_type']}")
    print(f"Metric: {config['metric_type']}")
    print(f"Output Directory: {config['output_dir']}")
    
    if config.get("quick_test", {}).get("enabled"):
        print("Quick Test Mode: ENABLED")
    
    print("=" * 60)


def main():
    # Load configuration
    config_path = Path(__file__).parent / "optimization_config.yml"
    
    try:
        print(f"Loading configuration from: {config_path}")
        config = load_optimization_config(config_path)
        config = apply_quick_test_config(config)
        print_configuration_summary(config)
        
        # Create optimization configuration
        opt_config = create_optimization_config_from_dict(config)
        
        # Setup DSPy and Weaviate
        setup_dspy()
        weaviate_client = setup_weaviate()
        
        # Run optimization
        optimizer = DSPyOptimizer(opt_config, weaviate_client)
        results = optimizer.run_optimization()
        
        # Print results
        print("\n" + "=" * 60)
        print("Optimization Summary")
        print("=" * 60)
        
        performance = results["performance"]
        timing = results["timing"]
        
        print(f"Mean Score: {performance['mean_dev_score']:.4f}")
        print(f"Std Dev: {performance['std_dev_score']:.4f}")
        print(f"Total Time: {timing['total_time']:.2f} seconds")
        
        if opt_config.save_optimized_program:
            print(f"Results saved to: {opt_config.output_dir}")
        
        print("=" * 60)
        
        return results
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        return None
        
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        try:
            if 'weaviate_client' in locals():
                weaviate_client.close()
        except:
            pass


if __name__ == "__main__":
    results = main()
    if results:
        print("\nOptimization completed successfully!")
    else:
        print("\nOptimization failed or was interrupted")
        exit(1)