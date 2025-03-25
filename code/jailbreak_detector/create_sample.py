#!/usr/bin/env python3
"""
Generate sample data for testing the jailbreak detector.
"""

import os
import json
import argparse
import logging

from utils import create_sample_data


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
        description="Generate sample data for jailbreak detection testing"
    )
    parser.add_argument(
        "--output", "-o",
        default="sample_responses.json",
        help="Path to output file for sample data"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of sample responses to generate"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the sample data generation script."""
    args = parse_arguments()
    logger = setup_logging()
    
    try:
        logger.info(f"Generating {args.count} sample responses")
        
        # Create the sample data file
        output_path = create_sample_data(args.output, args.count)
        
        logger.info(f"Sample data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 