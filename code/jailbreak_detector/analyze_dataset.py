#!/usr/bin/env python3
"""
Dataset Analysis Script for Jailbreak Detection

This script analyzes the output_cleaned_text.csv dataset using specified LLM models
and compares the results with human evaluations.
"""

import argparse
import json
import os
import logging
import time
import pandas as pd
from typing import List, Dict, Any
import torch

from detector import JailbreakDetector
from evaluator import JailbreakEvaluator
from utils import check_dependencies, get_available_device, format_duration


def setup_logging(log_file=None):
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze jailbreak patterns in the output_cleaned_text.csv dataset"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/llm-as-a-judge/output_cleaned_text.csv",
        help="Path to the output_cleaned_text.csv dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="analysis_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--model", "-m",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model to use for detection"
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of samples to use (all if not specified)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--no-quantize", "-nq",
        action="store_true",
        help="Disable 4-bit quantization (uses more memory but may be more accurate)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for model inference (cuda, cpu, mps, auto)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--save-disagreements",
        action="store_true",
        help="Save cases where model and human evaluations disagree"
    )
    parser.add_argument(
        "--models-list", "-ml",
        default=None,
        help="Path to JSON file containing a list of models to evaluate"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply all optimizations (compile, flash attention, etc.)"
    )
    # Add multi-GPU specific arguments
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "sequential"],
        default=None,
        help="Multi-GPU device mapping strategy"
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard ID for dataset splitting (0-based)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for dataset splitting"
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge results from multiple shards"
    )
    parser.add_argument(
        "--shards-pattern",
        default="{model_name}_shard{shard_id}_results.json",
        help="Pattern for shard result files"
    )
    
    return parser.parse_args()


def load_dataset(dataset_path: str, sample_size: int = None, shard_id: int = None, num_shards: int = None) -> pd.DataFrame:
    """
    Load the dataset for analysis.
    
    Args:
        dataset_path: Path to the dataset
        sample_size: Number of samples to use
        shard_id: Shard ID for dataset splitting
        num_shards: Total number of shards for dataset splitting
        
    Returns:
        Loaded dataset
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    
    # Log dataset information
    logger.info(f"Dataset size: {len(dataset)} rows")
    logger.info(f"Human evaluation distribution: {dataset['human_eval'].value_counts().to_dict()}")
    
    # Apply sharding if specified
    if shard_id is not None and num_shards is not None:
        if shard_id >= num_shards:
            raise ValueError(f"Shard ID {shard_id} must be less than number of shards {num_shards}")
            
        # Calculate shard size
        total_size = len(dataset)
        shard_size = total_size // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < num_shards - 1 else total_size
        
        # Apply sharding
        dataset = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
        logger.info(f"Processing shard {shard_id+1}/{num_shards} with {len(dataset)} samples")
    
    # Sample from dataset if requested
    elif sample_size is not None and sample_size < len(dataset):
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
        sampled_dataset = pd.concat([jb_sampled, not_jb_sampled]).sample(frac=1, random_state=42)
        
        logger.info(f"Sampled {len(sampled_dataset)} rows from dataset")
        logger.info(f"Sampled distribution: {sampled_dataset['human_eval'].value_counts().to_dict()}")
        
        return sampled_dataset
    
    return dataset


def analyze_with_model(
    model_name: str, 
    dataset: pd.DataFrame, 
    output_dir: str,
    batch_size: int = 8,
    quantize: bool = True,
    device: str = None,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    optimize: bool = False,
    device_map: str = None,
    shard_id: int = None
) -> Dict[str, Any]:
    """
    Analyze the dataset with a specific model.
    
    Args:
        model_name: Name of the model to use
        dataset: Dataset to analyze
        output_dir: Directory to save results
        batch_size: Batch size for model inference
        quantize: Whether to use 4-bit quantization
        device: Device for model inference
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        optimize: Whether to apply all optimizations
        device_map: Multi-GPU device mapping strategy
        shard_id: Shard ID for output filename (if using sharding)
        
    Returns:
        Analysis results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing dataset with model: {model_name}")
    
    # Extract responses from dataset
    responses = dataset['cleaned text'].tolist()
    
    # Start timing
    start_time = time.time()
    
    # Set device if not specified
    if device is None:
        device = get_available_device()
    
    # Initialize detector with optimizations
    detector = JailbreakDetector(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        quantize=quantize,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        # Apply optimizations if requested
        offload_to_cpu=optimize and device == "cuda",
        use_bettertransformer=optimize,
        use_flash_attention=optimize,
        use_compile=optimize and device == "cuda",
        use_cache=True,
        device_map=device_map
    )
    
    try:
        # Run analysis
        logger.info(f"Running analysis on {len(responses)} responses")
        results = detector.analyze_batch(responses)
        
        # Calculate time statistics
        total_time = time.time() - start_time
        avg_time_per_response = total_time / len(responses)
        
        logger.info(f"Analysis completed in {format_duration(total_time)}")
        logger.info(f"Average time per response: {format_duration(avg_time_per_response)}")
        
        # Determine jailbroken count
        jailbroken_count = sum(1 for r in results if r["jailbroken"])
        logger.info(f"Model detected {jailbroken_count}/{len(results)} responses as jailbroken")
        
        # Save results
        model_safe_name = model_name.replace('/', '_')
        
        # Adjust filename for sharded processing
        if shard_id is not None:
            results_filename = f"{model_safe_name}_shard{shard_id}_results.json"
        else:
            results_filename = f"{model_safe_name}_results.json"
            
        results_path = os.path.join(output_dir, results_filename)
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Clean up to free memory
        detector.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "model_name": model_name,
            "results_path": results_path,
            "total_time": total_time,
            "avg_time_per_response": avg_time_per_response,
            "jailbroken_count": jailbroken_count,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing with model {model_name}: {str(e)}", exc_info=True)
        return {
            "model_name": model_name,
            "error": str(e)
        }


def merge_shard_results(model_name: str, output_dir: str, num_shards: int, pattern: str = "{model_name}_shard{shard_id}_results.json") -> str:
    """
    Merge results from multiple shards.
    
    Args:
        model_name: Name of the model
        output_dir: Directory containing shard results
        num_shards: Number of shards to merge
        pattern: Pattern for shard result filenames
        
    Returns:
        Path to merged results file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Merging results from {num_shards} shards for model {model_name}")
    
    model_safe_name = model_name.replace('/', '_')
    all_results = []
    
    for shard_id in range(num_shards):
        filename = pattern.format(model_name=model_safe_name, shard_id=shard_id)
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Shard file {filepath} not found")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                shard_results = json.load(f)
                all_results.extend(shard_results)
                logger.info(f"Loaded {len(shard_results)} results from {filepath}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {filepath}")
    
    # Save merged results
    merged_filepath = os.path.join(output_dir, f"{model_safe_name}_results.json")
    with open(merged_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"Merged {len(all_results)} results saved to {merged_filepath}")
    return merged_filepath


def main():
    """Main entry point for the analysis script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    try:
        # Check dependencies
        check_dependencies()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # If merging shards, perform merge operation and exit
        if args.merge_shards:
            if args.num_shards is None:
                # Try to detect number of shards from existing files
                model_safe_name = args.model.replace('/', '_')
                pattern = args.shards_pattern.format(model_name=model_safe_name, shard_id="*")
                
                # Use simple logic to estimate number of shards
                import glob
                shard_files = glob.glob(os.path.join(args.output_dir, pattern.replace("*", "[0-9]*")))
                
                if not shard_files:
                    logger.error("No shard files found and --num-shards not specified")
                    return 1
                
                num_shards = max([int(f.split("shard")[-1].split("_")[0]) for f in shard_files]) + 1
                logger.info(f"Detected {num_shards} shards based on existing files")
            else:
                num_shards = args.num_shards
                
            merged_file = merge_shard_results(
                model_name=args.model,
                output_dir=args.output_dir,
                num_shards=num_shards,
                pattern=args.shards_pattern
            )
            
            # Run evaluation on merged results
            evaluator = JailbreakEvaluator(results_dir=args.output_dir)
            evaluator.evaluate_model(
                model_name=args.model,
                results_path=merged_file,
                dataset_path=args.data
            )
            
            logger.info("Merge and evaluation complete")
            return 0
        
        # Load dataset
        dataset = load_dataset(args.data, args.sample_size, args.shard_id, args.num_shards)
        
        # Determine models to evaluate
        models_to_evaluate = [args.model]
        if args.models_list:
            try:
                with open(args.models_list, 'r', encoding='utf-8') as f:
                    models_config = json.load(f)
                    if isinstance(models_config, list):
                        models_to_evaluate = models_config
                    logger.info(f"Loaded {len(models_to_evaluate)} models from {args.models_list}")
            except Exception as e:
                logger.error(f"Error loading models list: {str(e)}")
        
        # Track results for each model
        evaluation_results = []
        
        # Initialize evaluator
        evaluator = JailbreakEvaluator(results_dir=args.output_dir)
        
        # Analyze with each model
        for model_name in models_to_evaluate:
            logger.info(f"Starting evaluation with model: {model_name}")
            
            # Run analysis
            result = analyze_with_model(
                model_name=model_name,
                dataset=dataset,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                quantize=not args.no_quantize,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                optimize=args.optimize,
                device_map=args.device_map,
                shard_id=args.shard_id
            )
            
            # Skip if there was an error
            if "error" in result:
                logger.error(f"Skipping evaluation for {model_name} due to analysis error")
                continue
            
            # Skip evaluation if using sharding (will be done after merging)
            if args.shard_id is not None:
                logger.info(f"Skipping evaluation for shard {args.shard_id} (will be done after merging)")
                continue
            
            # Evaluate model against human evaluations
            logger.info(f"Evaluating {model_name} against human evaluations")
            eval_result = evaluator.evaluate_model(
                model_name=model_name,
                results_path=result["results_path"],
                dataset_path=args.data
            )
            
            # Save disagreements if requested
            if args.save_disagreements:
                # Load model results
                with open(result["results_path"], 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                
                # Convert to DataFrame
                model_df = pd.DataFrame(model_results)
                
                # Merge with dataset
                merged_df = evaluator.merge_results(model_df, dataset)
                
                # Identify disagreements
                disagreements = evaluator.identify_disagreements(merged_df)
                
                logger.info(f"Found {len(disagreements)} disagreements for {model_name}")
            
            evaluation_results.append(eval_result)
        
        # Compare models if more than one was evaluated and not using sharding
        if len(evaluation_results) > 1 and args.shard_id is None:
            logger.info(f"Comparing {len(evaluation_results)} models")
            comparison_df = evaluator.compare_models(evaluation_results)
            
            # Plot comparisons
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                evaluator.plot_comparison(comparison_df, metric)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 