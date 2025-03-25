#!/usr/bin/env python3
"""
Multi-GPU Dataset Analysis Script

This script analyzes the output_cleaned_text.csv dataset across multiple GPUs by sharding
the data and running separate processes for each GPU, then merging the results.
"""

import argparse
import subprocess
import os
import json
import logging
import time
import sys
from typing import List, Dict, Any
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
        description="Analyze the dataset across multiple GPUs using sharding"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/llm-as-a-judge/output_cleaned_text.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="multi_gpu_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--model", "-m",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model to use for detection"
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
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
        "--optimize",
        action="store_true",
        help="Apply all optimizations (compile, flash attention, etc.)"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--single-gpu-per-process",
        action="store_true",
        help="Use a single GPU per process (instead of all specified GPUs for each process)"
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "sequential"],
        default=None,
        help="Multi-GPU device mapping strategy (for multi-GPU per process mode)"
    )
    parser.add_argument(
        "--models-list", "-ml",
        default=None,
        help="Path to JSON file containing a list of models to evaluate"
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


def run_shard_process(
    model_name: str,
    shard_id: int,
    num_shards: int,
    gpu_id: str,
    args: argparse.Namespace,
    log_file: str
) -> subprocess.Popen:
    """
    Run a process to analyze a specific shard on a given GPU.
    
    Args:
        model_name: Name of the model to use
        shard_id: Shard ID to process
        num_shards: Total number of shards
        gpu_id: GPU ID to use
        args: Command line arguments
        log_file: Path to log file for this process
    
    Returns:
        Process object
    """
    # Build command
    cmd = [
        sys.executable, "analyze_dataset.py",
        "--model", model_name,
        "--data", args.data,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--shard-id", str(shard_id),
        "--num-shards", str(num_shards),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--log-file", log_file
    ]
    
    if args.optimize:
        cmd.append("--optimize")
        
    if args.no_quantize:
        cmd.append("--no-quantize")
    
    # Set environment for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Create log directory
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    # Start process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process


def merge_results(model_name: str, num_shards: int, output_dir: str):
    """
    Merge results from sharded processing.
    
    Args:
        model_name: Name of the model
        num_shards: Number of shards
        output_dir: Output directory
    """
    # Build command
    cmd = [
        sys.executable, "analyze_dataset.py",
        "--model", model_name,
        "--output-dir", output_dir,
        "--merge-shards",
        "--num-shards", str(num_shards)
    ]
    
    # Run process
    process = subprocess.run(cmd, check=True)
    return process.returncode == 0


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
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
        num_gpus = len(gpu_ids)
        
        # Get models
        if args.models_list:
            models = load_models_list(args.models_list)
            logger.info(f"Loaded {len(models)} models from {args.models_list}")
        else:
            models = [args.model]
        
        # Process each model
        for model_name in models:
            logger.info(f"Processing model: {model_name}")
            
            # Determine number of shards and GPUs per shard
            num_shards = num_gpus if args.single_gpu_per_process else 1
            
            # Run shard processes
            processes = []
            for shard_id in range(num_shards):
                # Determine which GPU(s) to use for this shard
                if args.single_gpu_per_process:
                    # One GPU per process
                    gpu_id = gpu_ids[shard_id]
                    logger.info(f"Running shard {shard_id} on GPU {gpu_id}")
                else:
                    # All GPUs for each process
                    gpu_id = ",".join(gpu_ids)
                    logger.info(f"Running shard {shard_id} on GPUs {gpu_id}")
                
                # Create log file
                log_file = os.path.join(
                    args.output_dir, 
                    "logs", 
                    f"{model_name.replace('/', '_')}_shard{shard_id}.log"
                )
                
                # Start process
                process = run_shard_process(
                    model_name=model_name,
                    shard_id=shard_id,
                    num_shards=num_shards,
                    gpu_id=gpu_id,
                    args=args,
                    log_file=log_file
                )
                
                processes.append((process, shard_id, log_file))
            
            # Monitor processes
            model_start_time = time.time()
            
            for process, shard_id, log_file in processes:
                logger.info(f"Waiting for shard {shard_id} to complete...")
                
                # Wait for process to complete while streaming output
                for line in process.stdout:
                    logger.debug(f"[Shard {shard_id}] {line.strip()}")
                
                process.wait()
                
                if process.returncode != 0:
                    logger.error(f"Shard {shard_id} failed with return code {process.returncode}")
                else:
                    logger.info(f"Shard {shard_id} completed successfully")
            
            # Merge results
            logger.info("Merging results from all shards...")
            if merge_results(model_name, num_shards, args.output_dir):
                logger.info("Results merged successfully")
            else:
                logger.error("Failed to merge results")
            
            model_duration = time.time() - model_start_time
            logger.info(f"Model {model_name} processed in {model_duration:.2f} seconds")
            
        logger.info("All models processed successfully")
        
    except Exception as e:
        logger.error(f"Error during multi-GPU processing: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 