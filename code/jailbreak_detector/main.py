#!/usr/bin/env python3
"""
Jailbreak Detection Script

This script analyzes LLM responses to determine if they show signs of successful jailbreaking.
It loads a specified LLM from Hugging Face and uses it to evaluate whether responses
indicate that a model has been compromised or manipulated to bypass safety measures.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Union
import logging

from detector import JailbreakDetector


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
        description="Detect jailbreak patterns in LLM responses"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input file containing LLM responses (JSON or text)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results.json",
        help="Path to output file for detection results"
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model to use for detection"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "text"],
        default="json",
        help="Format of the output file"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        help="Device for model inference (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_responses(input_path: str) -> List[str]:
    """
    Load responses from input file.
    Supports JSON and plain text formats.
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading responses from {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as file:
        if input_path.endswith(".json"):
            data = json.load(file)
            # Handle different possible JSON structures
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) for item in data):
                    # Extract response field from each item if it exists
                    responses = []
                    for item in data:
                        if "response" in item:
                            responses.append(item["response"])
                        else:
                            # Try to find a likely response field
                            text_fields = [v for k, v in item.items() 
                                          if isinstance(v, str) and len(v) > 50]
                            if text_fields:
                                responses.append(text_fields[0])
                    return responses
            elif isinstance(data, dict):
                # Try to extract responses from a dictionary structure
                if "responses" in data:
                    return data["responses"]
                else:
                    # Look for array fields that might contain responses
                    for key, value in data.items():
                        if isinstance(value, list) and all(isinstance(x, str) for x in value):
                            return value
            
            # If we can't determine the structure, return an error
            raise ValueError("Could not parse JSON structure. Please provide a list of response strings.")
        else:
            # For text files, we assume one response per line
            return [line.strip() for line in file if line.strip()]


def save_results(results: List[Dict[str, Any]], output_path: str, format_type: str):
    """Save detection results to the specified output file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {output_path}")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if format_type == "json":
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    elif format_type == "csv":
        import csv
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            if results:
                writer = csv.DictWriter(file, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    elif format_type == "text":
        with open(output_path, "w", encoding="utf-8") as file:
            for result in results:
                file.write(f"Response: {result['response'][:100]}...\n")
                file.write(f"Jailbroken: {result['jailbroken']}\n")
                file.write(f"Confidence: {result['confidence']}\n")
                if 'explanation' in result:
                    file.write(f"Explanation: {result['explanation']}\n")
                file.write("\n" + "-"*50 + "\n\n")


def main():
    """Main entry point for the jailbreak detection script."""
    args = parse_arguments()
    
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting jailbreak detection using model: {args.model}")
    
    try:
        # Load responses from input file
        responses = load_responses(args.input)
        logger.info(f"Loaded {len(responses)} responses for analysis")
        
        # Initialize the detector with the specified model
        detector = JailbreakDetector(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size
        )
        
        # Run detection on all responses
        results = detector.analyze_batch(responses)
        
        # Save results to output file
        save_results(results, args.output, args.format)
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
        
        # Print summary
        jailbroken_count = sum(1 for r in results if r["jailbroken"])
        logger.info(f"Summary: {jailbroken_count}/{len(results)} responses detected as jailbroken")
        
    except Exception as e:
        logger.error(f"Error during jailbreak detection: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 