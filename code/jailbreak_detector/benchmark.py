#!/usr/bin/env python3
"""
Benchmark script for jailbreak detection models

This script evaluates multiple LLM models on their ability to detect jailbreak patterns
and compares their performance against human evaluations.
"""

import argparse
import json
import os
import logging
import time
from typing import List, Dict, Any
import pandas as pd

from detector import JailbreakDetector
from evaluator import JailbreakEvaluator
from utils import check_dependencies, get_available_device, format_duration


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark jailbreak detection models"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/llm-as-a-judge/output_cleaned_text.csv",
        help="Path to dataset CSV file with human evaluations"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=[
            "meta-llama/Llama-2-7b-chat-hf",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "mistralai/Mistral-7B-Instruct-v0.2"
        ],
        help="List of models to benchmark"
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of samples to use from the dataset (all if not specified)"
    )
    parser.add_argument(
        "--device", 
        default=None,
        help="Device for model inference (cuda, cpu, mps, auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Use quantization for reduced memory usage"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for model generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def prepare_dataset(dataset_path: str, sample_size: int = None) -> List[str]:
    """
    Prepare dataset for benchmark.
    
    Args:
        dataset_path: Path to dataset CSV file
        sample_size: Number of samples to use
        
    Returns:
        List of responses to analyze
    """
    logger = logging.getLogger(__name__)
    
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    
    # Log dataset statistics
    logger.info(f"Dataset size: {len(dataset)} rows")
    logger.info(f"Human evaluation distribution: {dataset['human_eval'].value_counts().to_dict()}")
    
    # Sample from dataset if requested
    if sample_size is not None and sample_size < len(dataset):
        # Stratified sampling to maintain class distribution
        jailbroken = dataset[dataset['human_eval'] == 1]
        not_jailbroken = dataset[dataset['human_eval'] == 0]
        
        # Calculate proportions
        jb_proportion = len(jailbroken) / len(dataset)
        jb_samples = int(sample_size * jb_proportion)
        not_jb_samples = sample_size - jb_samples
        
        # Sample
        jb_sampled = jailbroken.sample(jb_samples, random_state=42)
        not_jb_sampled = not_jailbroken.sample(not_jb_samples, random_state=42)
        
        # Combine
        dataset = pd.concat([jb_sampled, not_jb_sampled]).sample(frac=1, random_state=42)
        
        logger.info(f"Sampled {len(dataset)} rows for benchmark")
        logger.info(f"Sampled distribution: {dataset['human_eval'].value_counts().to_dict()}")
    
    # Extract responses
    responses = dataset['cleaned text'].tolist()
    
    return responses


def benchmark_model(
    model_name: str, 
    responses: List[str], 
    output_dir: str,
    device: str = None,
    batch_size: int = 1,
    quantize: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.1
) -> str:
    """
    Benchmark a single model.
    
    Args:
        model_name: Name of the model to benchmark
        responses: List of responses to analyze
        output_dir: Directory to save results
        device: Device for model inference
        batch_size: Batch size for model inference
        quantize: Whether to use quantization
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for model generation
        
    Returns:
        Path to results file
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Benchmarking model: {model_name}")
    start_time = time.time()
    
    # Initialize detector
    detector = JailbreakDetector(
        model_name=model_name,
        device=device if device else get_available_device(),
        batch_size=batch_size,
        quantize=quantize,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Run detection on responses
    results = detector.analyze_batch(responses)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_results.json")
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Log summary
    duration = time.time() - start_time
    jailbroken_count = sum(1 for r in results if r["jailbroken"])
    average_time = duration / len(responses)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Total duration: {format_duration(duration)}")
    logger.info(f"Average time per response: {format_duration(average_time)}")
    logger.info(f"Results: {jailbroken_count}/{len(responses)} responses detected as jailbroken")
    logger.info(f"Results saved to {result_path}")
    
    return result_path


def main():
    """Main entry point for the benchmark script."""
    args = parse_arguments()
    
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Check dependencies
        check_dependencies()
        
        # Prepare dataset
        responses = prepare_dataset(args.data, args.sample_size)
        
        # Set up evaluator
        evaluator = JailbreakEvaluator(results_dir=args.output_dir)
        
        # Benchmark each model
        results_files = []
        for model_name in args.models:
            result_path = benchmark_model(
                model_name=model_name,
                responses=responses,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                quantize=args.quantize,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            results_files.append((model_name, result_path))
        
        # Evaluate each model against human evaluations
        evaluation_results = []
        for model_name, result_path in results_files:
            result = evaluator.evaluate_model(model_name, result_path, args.data)
            evaluation_results.append(result)
        
        # Compare models
        comparison_df = evaluator.compare_models(evaluation_results)
        
        # Plot comparisons
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            evaluator.plot_comparison(comparison_df, metric)
        
        logger.info("Benchmark complete!")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 