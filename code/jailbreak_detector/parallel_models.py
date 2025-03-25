#!/usr/bin/env python3
"""
Parallel Models Execution Script

This script runs multiple models in parallel across different GPUs
to maximize throughput when evaluating multiple models.
"""

import argparse
import subprocess
import os
import json
import logging
import time
import sys
from typing import List, Dict, Any
import multiprocessing
import torch

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
        description="Run multiple models in parallel across different GPUs"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/llm-as-a-judge/output_cleaned_text.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="parallel_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--models-list", "-ml",
        default="models_list.json",
        help="Path to JSON file containing a list of models to evaluate"
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of samples to use from dataset"
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
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of models to run concurrently (defaults to number of GPUs)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply all optimizations"
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save per-model log files"
    )
    
    return parser.parse_args()


def detect_gpus():
    """Detect available GPUs and return their IDs."""
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    return [str(i) for i in range(gpu_count)]


def load_models_list(path: str) -> List[str]:
    """Load list of models from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Models list file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        models = json.load(f)
    
    if not isinstance(models, list):
        raise TypeError("Models list should be a JSON array of model names")
    
    return models


def process_model(model_name: str, gpu_id: str, args: argparse.Namespace):
    """
    Process a model on a specific GPU.
    
    Args:
        model_name: Name of the model to use
        gpu_id: GPU ID to use
        args: Command line arguments
    
    Returns:
        Dict with processing results
    """
    logger = logging.getLogger(f"process-{model_name}-gpu{gpu_id}")
    logger.info(f"Starting processing of model {model_name} on GPU {gpu_id}")
    
    start_time = time.time()
    
    # Create log file
    log_file = os.path.join(
        args.output_dir,
        args.log_dir,
        f"{model_name.replace('/', '_')}_gpu{gpu_id}.log"
    )
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "analyze_dataset.py",
        "--model", model_name,
        "--data", args.data,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--log-file", log_file
    ]
    
    if args.sample_size:
        cmd.extend(["--sample-size", str(args.sample_size)])
    
    if args.optimize:
        cmd.append("--optimize")
        
    if args.no_quantize:
        cmd.append("--no-quantize")
    
    # Set environment for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run process
    try:
        process = subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        success = process.returncode == 0
        duration = time.time() - start_time
        
        logger.info(f"Finished processing model {model_name} on GPU {gpu_id} in {duration:.2f} seconds")
        
        return {
            "model_name": model_name,
            "gpu_id": gpu_id,
            "success": success,
            "duration": duration,
            "log_file": log_file
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing model {model_name} on GPU {gpu_id}: {str(e)}")
        
        return {
            "model_name": model_name,
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
            "log_file": log_file
        }


def worker_process(queue, results, model_name, gpu_id, args):
    """Worker process function for multiprocessing."""
    try:
        result = process_model(model_name, gpu_id, args)
        results.append(result)
        queue.put(1)  # Signal task completion
    except Exception as e:
        print(f"Error in worker for {model_name} on GPU {gpu_id}: {str(e)}")
        queue.put(0)  # Signal task failure


def run_models_in_parallel(models: List[str], gpu_ids: List[str], args: argparse.Namespace) -> List[Dict]:
    """
    Run multiple models in parallel across different GPUs.
    
    Args:
        models: List of model names
        gpu_ids: List of GPU IDs
        args: Command line arguments
        
    Returns:
        List of result dictionaries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {len(models)} models in parallel across {len(gpu_ids)} GPUs")
    
    # Determine maximum concurrency
    max_concurrency = args.max_concurrency or len(gpu_ids)
    logger.info(f"Maximum concurrency: {max_concurrency}")
    
    # Create shared resources for workers
    manager = multiprocessing.Manager()
    results = manager.list()
    queue = multiprocessing.Queue()
    
    # Track processes
    processes = []
    running = 0
    processed = 0
    
    # Start initial batch of processes
    for i, model_name in enumerate(models):
        if running >= max_concurrency:
            break
            
        gpu_id = gpu_ids[i % len(gpu_ids)]
        logger.info(f"Starting model {model_name} on GPU {gpu_id}")
        
        p = multiprocessing.Process(
            target=worker_process,
            args=(queue, results, model_name, gpu_id, args)
        )
        p.start()
        processes.append(p)
        running += 1
        processed += 1
    
    # Process remaining models
    while processed < len(models) or running > 0:
        # Wait for a process to complete
        signal = queue.get()
        running -= 1
        
        # Start a new process if there are more models
        if processed < len(models):
            model_name = models[processed]
            gpu_id = gpu_ids[processed % len(gpu_ids)]
            logger.info(f"Starting model {model_name} on GPU {gpu_id}")
            
            p = multiprocessing.Process(
                target=worker_process,
                args=(queue, results, model_name, gpu_id, args)
            )
            p.start()
            processes.append(p)
            running += 1
            processed += 1
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Convert manager list to regular list
    return list(results)


def compare_results(output_dir: str):
    """
    Compare results from all models.
    
    Args:
        output_dir: Directory containing results
    """
    # Build command
    cmd = [
        sys.executable, "run_models_comparison.py",
        "--output-dir", output_dir
    ]
    
    # Run process
    subprocess.run(cmd, check=True)


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Get GPU IDs
        if args.gpu_ids == "auto":
            gpu_ids = detect_gpus()
            if not gpu_ids:
                logger.error("No GPUs detected")
                return 1
        else:
            gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids.split(",")]
        
        logger.info(f"Using GPUs: {', '.join(gpu_ids)}")
        
        # Load models list
        models = load_models_list(args.models_list)
        logger.info(f"Loaded {len(models)} models from {args.models_list}")
        
        # Run models in parallel
        results = run_models_in_parallel(models, gpu_ids, args)
        
        # Log results summary
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        logger.info(f"Results summary: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            logger.warning("Failed models:")
            for result in failed:
                logger.warning(f"  {result['model_name']} - Check log file: {result.get('log_file', 'N/A')}")
        
        # Save summary
        summary_path = os.path.join(args.output_dir, "parallel_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Convert list to serializable format
            serializable_results = []
            for r in results:
                serializable_dict = dict(r)
                serializable_results.append(serializable_dict)
                
            json.dump({
                "timestamp": time.time(),
                "total_models": len(models),
                "successful": len(successful),
                "failed": len(failed),
                "gpu_ids": gpu_ids,
                "results": serializable_results
            }, f, indent=2)
            
        logger.info(f"Summary saved to {summary_path}")
        
        # Compare results
        if len(successful) > 1:
            logger.info("Comparing model results...")
            try:
                compare_results(args.output_dir)
                logger.info("Model comparison complete")
            except Exception as e:
                logger.error(f"Error comparing model results: {str(e)}")
        
        logger.info("Parallel processing complete")
        
    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 