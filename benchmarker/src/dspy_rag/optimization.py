"""
DSPy optimization integration for the Weaviate Query Agent Benchmarker.

This module provides DSPy-compatible metrics and optimization utilities
for RAG programs, integrated with the existing evaluation infrastructure.
"""

import asyncio
import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import dspy
from dspy import Example
from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    COPRO,
    MIPROv2
)

from benchmarker.src.dataset import in_memory_dataset_loader, split_dataset
from benchmarker.src.dspy_rag.rag_programs import RAG_VARIANTS
from benchmarker.src.metrics.ir_metrics import calculate_recall
from benchmarker.src.metrics.lm_as_judge_agent import lm_as_judge_agent, LMJudgeAgentDeps
from benchmarker.src.utils import qa_source_parser, make_json_serializable


# ============================================================================
# DSPy Example Creation and Metrics
# ============================================================================

def create_dspy_examples_from_dataset(
    queries: List[Dict], 
    dataset_name: str,
    include_answers: bool = False
) -> List[Example]:
    """
    Convert dataset queries to DSPy Examples.
    
    Args:
        queries: List of query dictionaries from dataset loader
        dataset_name: Name of the dataset for context
        include_answers: Whether to include ground truth answers (for supervised optimization)
        
    Returns:
        List of DSPy Example objects
    """
    examples = []
    
    for query in queries:
        example = Example()
        example = example.with_inputs("question")
        
        example["question"] = query["question"]
        
        # Add dataset_ids for recall evaluation
        if "dataset_ids" in query:
            example.dataset_ids = query["dataset_ids"]
        
        # Add nugget data if available (for FreshStack datasets)
        if "nugget_data" in query:
            example.nugget_data = query["nugget_data"]
            
        # Add ground truth answer if available and requested
        if include_answers and "answer" in query:
            example.answer = query["answer"]
    
        examples.append(example)
    
    return examples


class RecallMetric:
    """DSPy-compatible recall metric."""
    
    def __init__(self, weaviate_client, dataset_name: str, weight: float = 1.0):
        self.weaviate_client = weaviate_client
        self.dataset_name = dataset_name
        self.weight = weight
        
        # Set up collection based on dataset
        if dataset_name == "enron":
            self.collection = weaviate_client.collections.get("EnronEmails")
        elif dataset_name == "wixqa":
            self.collection = weaviate_client.collections.get("WixKB")
        elif dataset_name.startswith("freshstack-"):
            subset = dataset_name.split("-")[1].capitalize()
            self.collection = weaviate_client.collections.get(f"FreshStack{subset}")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def __call__(self, example: Example, prediction, trace=None) -> float:
        """
        Calculate recall for a single example.
        
        Args:
            example: DSPy Example with question and dataset_ids
            prediction: Model prediction with sources
            trace: Optional trace information
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, self.collection)
            else:
                retrieved_ids = []
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use nugget-based evaluation if available
            nugget_data = getattr(example, 'nugget_data', None)
            
            recall_score = calculate_recall(
                target_ids=target_ids,
                retrieved_ids=retrieved_ids,
                nugget_data=nugget_data
            )
            
            return recall_score * self.weight
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0


class LMJudgeMetric:
    """DSPy-compatible LM-as-a-Judge metric."""
    
    def __init__(self, weight: float = 1.0, model: str = "openai:gpt-4o"):
        self.weight = weight
        self.model = model
    
    def __call__(self, example: Example, prediction, trace=None) -> float:
        """
        Calculate LM judge score for a single example.
        
        Args:
            example: DSPy Example with question
            prediction: Model prediction with final_answer
            trace: Optional trace information
            
        Returns:
            Normalized LM judge score (0.0 to 1.0)
        """
        try:
            # Extract answer from prediction
            if hasattr(prediction, 'final_answer'):
                answer = prediction.final_answer
            elif hasattr(prediction, 'answer'):
                answer = prediction.answer
            else:
                answer = str(prediction)
            
            # Skip if no answer provided
            if not answer or answer.strip() == "":
                return 0.0
            
            # Run LM judge evaluation
            deps = LMJudgeAgentDeps(
                question=example.question,
                system_response=answer
            )
            
            # Run async evaluation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    lm_as_judge_agent.run(deps=deps, model=self.model)
                )
                # Normalize from 1-5 scale to 0-1 scale
                normalized_score = (result.data.rating - 1) / 4
                return normalized_score * self.weight
            finally:
                loop.close()
                
        except Exception as e:
            print(f"Error calculating LM judge score: {e}")
            return 0.0


class CompositeMetric:
    """Composite metric combining multiple evaluation criteria."""
    
    def __init__(
        self, 
        weaviate_client,
        dataset_name: str,
        recall_weight: float = 0.5,
        lm_judge_weight: float = 0.5,
        lm_judge_model: str = "openai:gpt-4o"
    ):
        self.recall_metric = RecallMetric(weaviate_client, dataset_name, recall_weight)
        self.lm_judge_metric = LMJudgeMetric(lm_judge_weight, lm_judge_model)
        self.total_weight = recall_weight + lm_judge_weight
        
    def __call__(self, example: Example, prediction, trace=None) -> float:
        """
        Calculate composite score combining recall and LM judge metrics.
        
        Args:
            example: DSPy Example
            prediction: Model prediction
            trace: Optional trace information
            
        Returns:
            Composite score (0.0 to 1.0)
        """
        recall_score = self.recall_metric(example, prediction, trace)
        lm_judge_score = self.lm_judge_metric(example, prediction, trace)
        
        composite_score = (recall_score + lm_judge_score) / self.total_weight
        return composite_score


def create_metric(
    metric_type: str,
    weaviate_client,
    dataset_name: str,
    **kwargs
) -> Union[RecallMetric, LMJudgeMetric, CompositeMetric]:
    """
    Factory function for creating metric instances.
    
    Args:
        metric_type: Type of metric ("recall", "lm_judge", "composite")
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for metric configuration
        
    Returns:
        Configured metric instance
    """
    if metric_type == "recall":
        return RecallMetric(weaviate_client, dataset_name, **kwargs)
    elif metric_type == "lm_judge":
        return LMJudgeMetric(**kwargs)
    elif metric_type == "composite":
        return CompositeMetric(weaviate_client, dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# ============================================================================
# Optimization Configuration and Main Class
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for DSPy optimization."""
    
    # Dataset and agent configuration
    dataset_name: str
    agent_name: str
    
    # Optimization parameters
    optimizer_type: str = "copro"  # copro, bootstrap_few_shot, bootstrap_random_search, mipro
    metric_type: str = "recall"  # recall, lm_judge, composite
    
    # Train/dev split
    train_ratio: float = 0.7
    max_train_samples: Optional[int] = None
    max_dev_samples: Optional[int] = None
    
    # Optimizer-specific parameters
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    num_candidate_programs: int = 10  # breadth for COPRO
    num_threads: int = 4
    
    # COPRO-specific parameters
    copro_depth: int = 3  # Number of optimization rounds
    copro_init_temperature: float = 1.4  # Higher = more creative prompts
    
    # Metric configuration
    recall_weight: float = 0.5
    lm_judge_weight: float = 0.5
    lm_judge_model: str = "openai:gpt-4o"
    
    # Output configuration
    save_optimized_program: bool = True
    output_dir: str = "optimization_results"
    experiment_name: Optional[str] = None


class DSPyOptimizer:
    """Main class for running DSPy optimization on RAG programs."""
    
    def __init__(self, config: OptimizationConfig, weaviate_client):
        self.config = config
        self.weaviate_client = weaviate_client
        self.optimized_program = None
        self.optimization_results = {}
        
        # Validate configuration
        self._validate_config()
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _validate_config(self):
        """Validate the optimization configuration."""
        if self.config.agent_name not in RAG_VARIANTS:
            raise ValueError(f"Agent {self.config.agent_name} not supported for optimization. "
                           f"Must be one of: {list(RAG_VARIANTS.keys())}")
        
        valid_optimizers = ["copro", "bootstrap_few_shot", "bootstrap_random_search", "mipro"]
        if self.config.optimizer_type not in valid_optimizers:
            raise ValueError(f"Optimizer {self.config.optimizer_type} not supported. "
                           f"Must be one of: {valid_optimizers}")
        
        valid_metrics = ["recall", "lm_judge", "composite", "fast_recall"]
        if self.config.metric_type not in valid_metrics:
            raise ValueError(f"Metric {self.config.metric_type} not supported. "
                           f"Must be one of: {valid_metrics}")
    
    def prepare_data(self) -> Tuple[List[dspy.Example], List[dspy.Example]]:
        """
        Load and prepare training and development datasets.
        
        Returns:
            Tuple of (train_examples, dev_examples)
        """
        print(f"\033[94mLoading dataset: {self.config.dataset_name}\033[0m")
        
        # Load dataset
        _, queries = in_memory_dataset_loader(self.config.dataset_name)
        
        # Split into train/dev
        train_queries, dev_queries = split_dataset(
            queries, 
            train_ratio=self.config.train_ratio,
            shuffle=True
        )
        
        # Limit samples if specified
        if self.config.max_train_samples:
            train_queries = train_queries[:self.config.max_train_samples]
        if self.config.max_dev_samples:
            dev_queries = dev_queries[:self.config.max_dev_samples]
        
        # Convert to DSPy examples
        train_examples = create_dspy_examples_from_dataset(
            train_queries, 
            self.config.dataset_name,
            include_answers=False  # We don't have ground truth answers
        )
        
        dev_examples = create_dspy_examples_from_dataset(
            dev_queries,
            self.config.dataset_name,
            include_answers=False
        )
        
        print(f"\033[92mPrepared {len(train_examples)} training examples and {len(dev_examples)} dev examples\033[0m")
        
        return train_examples, dev_examples
    
    def create_program(self):
        """Create the RAG program to be optimized."""
        print(f"\033[94mCreating {self.config.agent_name} program\033[0m")
        
        # Get collection info based on dataset
        if self.config.dataset_name == "enron":
            collection_name = "EnronEmails"
            target_property_name = ""
        elif self.config.dataset_name == "wixqa":
            collection_name = "WixKB"
            target_property_name = "contents"
        elif self.config.dataset_name.startswith("freshstack-"):
            subset = self.config.dataset_name.split("-")[1].capitalize()
            collection_name = f"FreshStack{subset}"
            target_property_name = "docs_text"
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
        
        # Create RAG program
        rag_class = RAG_VARIANTS[self.config.agent_name]
        program = rag_class(
            collection_name=collection_name,
            target_property_name=target_property_name
        )
        
        return program
    
    def create_metric(self):
        """Create the evaluation metric."""
        print(f"\033[94mCreating {self.config.metric_type} metric\033[0m")
        
        metric_kwargs = {}
        if self.config.metric_type == "composite":
            metric_kwargs.update({
                "recall_weight": self.config.recall_weight,
                "lm_judge_weight": self.config.lm_judge_weight,
                "lm_judge_model": self.config.lm_judge_model
            })
        elif self.config.metric_type == "lm_judge":
            metric_kwargs.update({
                "model": self.config.lm_judge_model
            })
        
        return create_metric(
            metric_type=self.config.metric_type,
            weaviate_client=self.weaviate_client,
            dataset_name=self.config.dataset_name,
            **metric_kwargs
        )
    
    def create_optimizer(self, metric):
        """Create the DSPy optimizer."""
        print(f"\033[94mCreating {self.config.optimizer_type} optimizer\033[0m")
        
        if self.config.optimizer_type == "bootstrap_few_shot":
            return BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos
            )
        
        elif self.config.optimizer_type == "bootstrap_random_search":
            return BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                num_candidate_programs=self.config.num_candidate_programs
            )
        
        elif self.config.optimizer_type == "copro":
            return COPRO(
                metric=metric,
                breadth=self.config.num_candidate_programs,
                depth=self.config.copro_depth,
                init_temperature=self.config.copro_init_temperature,
                track_stats=True  # Track optimization statistics
            )
        
        elif self.config.optimizer_type == "mipro":
            return MIPROv2(
                metric=metric,
                num_candidates=self.config.num_candidate_programs,
                init_temperature=1.0
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        print(f"\033[95m{'='*60}\033[0m")
        print("\033[95mStarting DSPy Optimization\033[0m")
        print(f"\033[95m{'='*60}\033[0m")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Agent: {self.config.agent_name}")
        print(f"Optimizer: {self.config.optimizer_type}")
        print(f"Metric: {self.config.metric_type}")
        print(f"\033[95m{'='*60}\033[0m")
        
        try:
            # Prepare data
            train_examples, dev_examples = self.prepare_data()
            
            # Create program and metric
            program = self.create_program()
            metric = self.create_metric()
            
            # Create optimizer
            optimizer = self.create_optimizer(metric)
            
            # Run optimization
            print(f"\033[94mStarting optimization with {len(train_examples)} training examples...\033[0m")
            optimization_start = time.time()
            
            # Configure evaluation kwargs for COPRO
            eval_kwargs = dict(
                num_threads=self.config.num_threads, 
                display_progress=True, 
                display_table=0
            )
            
            if self.config.optimizer_type == "copro":
                # COPRO uses a different compilation approach
                self.optimized_program = optimizer.compile(
                    program.deepcopy(), 
                    trainset=train_examples,
                    eval_kwargs=eval_kwargs
                )
            else:
                # Standard compilation for other optimizers
                self.optimized_program = optimizer.compile(
                    student=program,
                    trainset=train_examples
                )
            
            optimization_time = time.time() - optimization_start
            
            # Evaluate optimized program
            print("\033[94mEvaluating optimized program on dev set...\033[0m")
            dev_scores = []
            for example in dev_examples:
                try:
                    prediction = self.optimized_program.forward(example.question)
                    score = metric(example, prediction)
                    dev_scores.append(score)
                except Exception as e:
                    print(f"Error evaluating example: {e}")
                    dev_scores.append(0.0)
            
            # Compile results
            total_time = time.time() - start_time
            
            self.optimization_results = {
                "config": {
                    "dataset_name": self.config.dataset_name,
                    "agent_name": self.config.agent_name,
                    "optimizer_type": self.config.optimizer_type,
                    "metric_type": self.config.metric_type,
                    "train_samples": len(train_examples),
                    "dev_samples": len(dev_examples)
                },
                "performance": {
                    "dev_scores": dev_scores,
                    "mean_dev_score": sum(dev_scores) / len(dev_scores) if dev_scores else 0,
                    "std_dev_score": (sum((x - sum(dev_scores)/len(dev_scores))**2 for x in dev_scores) / len(dev_scores))**0.5 if len(dev_scores) > 1 else 0
                },
                "timing": {
                    "total_time": total_time,
                    "optimization_time": optimization_time,
                    "data_prep_time": optimization_start - start_time
                },
                "optimization_details": {
                    "num_candidate_programs": self.config.num_candidate_programs,
                    "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                    "max_labeled_demos": self.config.max_labeled_demos
                }
            }
            
            # Add COPRO-specific statistics if available
            if self.config.optimizer_type == "copro" and hasattr(self.optimized_program, 'results_best'):
                self.optimization_results["copro_stats"] = {
                    "results_best": getattr(self.optimized_program, 'results_best', None),
                    "results_latest": getattr(self.optimized_program, 'results_latest', None),
                    "total_calls": getattr(self.optimized_program, 'total_calls', None)
                }
            
            # Save results
            if self.config.save_optimized_program:
                self._save_results()
            
            # Print summary
            self._print_summary()
            
            return self.optimization_results
            
        except Exception as e:
            print(f"\033[91mOptimization failed: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self):
        """Save optimization results and optimized program."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config.experiment_name or f"{self.config.agent_name}_{self.config.optimizer_type}"
        
        # Save results JSON
        results_file = self.output_dir / f"{experiment_name}_{timestamp}_results.json"
        with open(results_file, 'w') as f:
            json.dump(make_json_serializable(self.optimization_results), f, indent=2)
        
        # Save optimized program
        program_file = self.output_dir / f"{experiment_name}_{timestamp}_program.pkl"
        self.optimized_program.save(str(program_file))
        
        print(f"\033[92mResults saved to: {results_file}\033[0m")
        print(f"\033[92mOptimized program saved to: {program_file}\033[0m")
    
    def _print_summary(self):
        """Print optimization summary."""
        results = self.optimization_results
        
        print(f"\n\033[95m{'='*60}\033[0m")
        print("\033[95mOptimization Complete!\033[0m")
        print(f"\033[95m{'='*60}\033[0m")
        
        print("\033[96mPerformance:\033[0m")
        print(f"  Mean Dev Score: {results['performance']['mean_dev_score']:.4f}")
        print(f"  Std Dev Score: {results['performance']['std_dev_score']:.4f}")
        print(f"  Dev Samples: {len(results['performance']['dev_scores'])}")
        
        print("\n\033[96mTiming:\033[0m")
        print(f"  Total Time: {results['timing']['total_time']:.2f} seconds")
        print(f"  Optimization Time: {results['timing']['optimization_time']:.2f} seconds")
        print(f"  Data Prep Time: {results['timing']['data_prep_time']:.2f} seconds")
        
        # Print COPRO-specific statistics if available
        if "copro_stats" in results and results["copro_stats"]["total_calls"]:
            print("\n\033[96mCOPRO Statistics:\033[0m")
            print(f"  Total Metric Calls: {results['copro_stats']['total_calls']}")
            if results["copro_stats"]["results_best"]:
                print("  Best Results per Depth: Available")
            if results["copro_stats"]["results_latest"]:
                print("  Latest Results per Depth: Available")
        
        print(f"\033[95m{'='*60}\033[0m")


# ============================================================================
# Convenience Functions
# ============================================================================

def run_optimization_benchmark(
    dataset_name: str,
    agent_name: str,
    weaviate_client,
    optimizer_type: str = "bootstrap_few_shot",
    metric_type: str = "composite",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run optimization with minimal configuration.
    
    Args:
        dataset_name: Name of the dataset
        agent_name: Name of the RAG agent
        weaviate_client: Weaviate client instance
        optimizer_type: Type of optimizer to use
        metric_type: Type of metric to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimization results dictionary
    """
    config = OptimizationConfig(
        dataset_name=dataset_name,
        agent_name=agent_name,
        optimizer_type=optimizer_type,
        metric_type=metric_type,
        **kwargs
    )
    
    optimizer = DSPyOptimizer(config, weaviate_client)
    return optimizer.run_optimization()


def quick_optimization_example():
    """Example of running a quick optimization for testing."""
    import weaviate
    
    # Connect to Weaviate
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    try:
        # Configure DSPy
        dspy.configure(
            lm=dspy.LM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), cache=False),
            track_usage=True
        )
        
        # Run optimization with COPRO
        results = run_optimization_benchmark(
            dataset_name="freshstack-godot",
            agent_name="search-only-with-query-writer",
            weaviate_client=weaviate_client,
            optimizer_type="copro",
            metric_type="recall",
            max_train_samples=5,
            max_dev_samples=5,
            num_candidate_programs=4,
            copro_depth=2,
            copro_init_temperature=1.2,
            num_threads=5
        )
        
        return results
        
    finally:
        weaviate_client.close()


if __name__ == "__main__":
    # Run example optimization
    results = quick_optimization_example()
    print("Optimization completed successfully!") 