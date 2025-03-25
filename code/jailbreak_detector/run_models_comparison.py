#!/usr/bin/env python3
"""
Model Comparison Runner Script

This script automates the process of running multiple models on the dataset
and comparing their performance for jailbreak detection.
"""

import subprocess
import argparse
import logging
import json
import os
import time
from typing import List, Dict
import pandas as pd

from evaluator import JailbreakEvaluator
from utils import format_duration


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
        description="Run comparison of multiple models for jailbreak detection"
    )
    parser.add_argument(
        "--models-list", "-ml",
        default="models_list.json",
        help="Path to JSON file containing models to compare"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="models_comparison",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/llm-as-a-judge/output_cleaned_text.csv",
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=50,
        help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for model inference (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply all optimizations for inference"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file"
    )
    
    return parser.parse_args()


def load_models_list(path: str) -> List[str]:
    """Load list of models from JSON file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading models list from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        models = json.load(f)
    
    if not isinstance(models, list):
        raise TypeError("Models list should be a JSON array of model names")
    
    logger.info(f"Loaded {len(models)} models")
    return models


def run_model_analysis(
    model_name: str, 
    output_dir: str, 
    data_path: str, 
    sample_size: int,
    device: str = None,
    optimize: bool = False
) -> Dict:
    """
    Run analysis for a single model.
    
    Args:
        model_name: Name of the model to analyze
        output_dir: Directory for output files
        data_path: Path to dataset
        sample_size: Number of samples to use
        device: Device for inference
        optimize: Whether to apply optimizations
        
    Returns:
        Result info dictionary
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running analysis for model: {model_name}")
    
    start_time = time.time()
    
    # Build command
    cmd = [
        "python", "analyze_dataset.py",
        "--model", model_name,
        "--data", data_path,
        "--output-dir", output_dir,
        "--batch-size", "8",
        "--sample-size", str(sample_size),
        "--save-disagreements"
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    if optimize:
        cmd.append("--optimize")
    
    # Create model-specific log file
    model_log_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_run.log")
    os.makedirs(os.path.dirname(os.path.abspath(model_log_file)), exist_ok=True)
    
    # Run analysis process
    logger.info(f"Executing: {' '.join(cmd)}")
    with open(model_log_file, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for completion
        process.wait()
    
    # Check result
    if process.returncode != 0:
        logger.error(f"Analysis failed for model {model_name} with return code {process.returncode}")
        return {
            "model_name": model_name,
            "success": False,
            "return_code": process.returncode,
            "log_file": model_log_file
        }
    
    # Successfully completed
    duration = time.time() - start_time
    logger.info(f"Analysis completed for {model_name} in {format_duration(duration)}")
    
    return {
        "model_name": model_name,
        "success": True,
        "duration": duration,
        "log_file": model_log_file,
        "results_file": os.path.join(output_dir, f"{model_name.replace('/', '_')}_results.json"),
        "evaluation_file": os.path.join(output_dir, f"{model_name.replace('/', '_')}_evaluation.json")
    }


def collect_results(run_results: List[Dict], output_dir: str) -> pd.DataFrame:
    """
    Collect and combine results from successful runs.
    
    Args:
        run_results: List of run result dictionaries
        output_dir: Directory containing result files
        
    Returns:
        DataFrame with combined results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize evaluator
    evaluator = JailbreakEvaluator(results_dir=output_dir)
    
    # Collect successful evaluations
    evaluation_results = []
    
    for result in run_results:
        if result["success"] and os.path.exists(result.get("evaluation_file", "")):
            try:
                with open(result["evaluation_file"], 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    evaluation_results.append(eval_data)
            except Exception as e:
                logger.error(f"Error loading evaluation for {result['model_name']}: {str(e)}")
    
    if not evaluation_results:
        logger.error("No successful evaluations found")
        return None
    
    # Create comparison
    comparison_df = evaluator.compare_models(evaluation_results)
    
    # Plot comparisons
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        try:
            evaluator.plot_comparison(comparison_df, metric)
        except Exception as e:
            logger.error(f"Error creating plot for {metric}: {str(e)}")
    
    logger.info(f"Generated comparison for {len(comparison_df)} models")
    return comparison_df


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load models list
        models = load_models_list(args.models_list)
        
        # Run analysis for each model
        run_results = []
        for model_name in models:
            result = run_model_analysis(
                model_name=model_name,
                output_dir=args.output_dir,
                data_path=args.data,
                sample_size=args.sample_size,
                device=args.device,
                optimize=args.optimize
            )
            run_results.append(result)
        
        # Collect and combine results
        comparison_df = collect_results(run_results, args.output_dir)
        
        if comparison_df is not None:
            # Log the final comparison table
            logger.info("\nModel Comparison Results:")
            logger.info("\n" + comparison_df.to_string())
            
            # Determine best model by F1 score
            best_model = comparison_df.loc[comparison_df['f1'].idxmax()]
            logger.info(f"\nBest model by F1 score: {best_model['model_name']} (F1: {best_model['f1']:.4f})")
            
            # Save run summary
            summary_path = os.path.join(args.output_dir, "run_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.time(),
                    "models_count": len(models),
                    "successful_runs": sum(1 for r in run_results if r["success"]),
                    "failed_runs": sum(1 for r in run_results if not r["success"]),
                    "sample_size": args.sample_size,
                    "best_model": {
                        "name": best_model['model_name'],
                        "f1": float(best_model['f1']),
                        "accuracy": float(best_model['accuracy']),
                        "precision": float(best_model['precision']),
                        "recall": float(best_model['recall'])
                    },
                    "run_results": run_results
                }, f, indent=2)
            
            logger.info(f"Run summary saved to {summary_path}")
        
        logger.info("Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 