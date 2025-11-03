#!/usr/bin/env python3
"""
CLI script for fitting content to constraints.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from cs4.core.constraint_fitter import ConstraintFitter
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Fit base content to satisfy constraints"
    )
    parser.add_argument(
        "--domain",
        choices=["blog", "story", "news"],
        default="blog",
        help="Content domain"
    )
    parser.add_argument(
        "--constraints-path",
        required=True,
        help="Path to constraints CSV"
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help="Path to base generated content CSV"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., fitted_content.csv)"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_FITTING_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_FITTING_MODEL})"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts on failure"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Config.LOGS_DIR / "constraint_fitting.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting constraint fitting for domain: {args.domain}")
    logger.info(f"Constraints: {args.constraints_path}")
    logger.info(f"Base content: {args.base_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        constraints_df = pd.read_csv(args.constraints_path, encoding="utf-8")
        base_df = pd.read_csv(args.base_path, encoding="utf-8")
        logger.info(f"Loaded {len(constraints_df)} constraints, {len(base_df)} base samples")
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        sys.exit(1)
    
    # Initialize LLM client
    try:
        if args.provider == "openai":
            client = OpenAIClient(log_usage=True)
        else:
            client = AnthropicClient(log_usage=True)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    # Initialize fitter
    fitter = ConstraintFitter(
        llm_client=client,
        model=args.model,
        content_type=args.domain,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retry_attempts=args.retry_attempts
    )
    
    # Fit content to constraints
    try:
        result_df = fitter.fit_batch(
            constraints_df=constraints_df,
            base_df=base_df,
            output_path=args.output_path
        )
        logger.info(f"Successfully fitted content for {len(result_df)} samples")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Constraint fitting failed: {e}")
        sys.exit(1)
    
    logger.info("Constraint fitting complete!")


if __name__ == "__main__":
    main()
