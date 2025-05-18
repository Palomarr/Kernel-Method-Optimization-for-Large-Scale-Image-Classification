#!/usr/bin/env python
"""
Main runner script for Kernel Method Optimization.

This script provides a unified interface to run all the different components
of the project including:
- Hyperparameter optimization
- Benchmarks and evaluations
- CNN comparisons
"""
import argparse
import os
import sys
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kernel Method Optimization for Image Classification"
    )
    parser.add_argument(
        'action',
        choices=['optimize', 'evaluate', 'benchmark', 'compare-cnn', 'all'],
        help="Action to perform"
    )
    parser.add_argument(
        '--no-optimize', 
        action='store_true',
        help="Skip optimization when running benchmark or evaluation"
    )
    parser.add_argument(
        '--samples', 
        type=int,
        default=10000,
        help="Maximum number of samples to use for training"
    )
    
    return parser.parse_args()

def run_optimization():
    """Run hyperparameter optimization."""
    logger.info("Running hyperparameter optimization...")
    subprocess.run(
        ["python", "scripts/run_optimization.py"],
        check=True
    )

def run_benchmark(use_optimized=True):
    """Run benchmark experiments."""
    logger.info("Running benchmark experiments...")
    cmd = ["python", "scripts/run_experiments.py"]
    if not use_optimized:
        cmd.append("no-optimize")
    subprocess.run(cmd, check=True)

def run_evaluation(use_optimized=True):
    """Run full evaluation."""
    logger.info("Running full evaluation...")
    subprocess.run(
        ["python", "scripts/run_evaluations.py"],
        check=True
    )

def run_cnn_comparison():
    """Run comparison with CNN baseline."""
    logger.info("Running comparison with CNN baseline...")
    subprocess.run(
        ["python", "scripts/compare_with_cnn.py"],
        check=True
    )

def main():
    """Run the requested action."""
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    if args.action == 'optimize' or args.action == 'all':
        run_optimization()
    
    if args.action == 'benchmark' or args.action == 'all':
        run_benchmark(use_optimized=not args.no_optimize)
    
    if args.action == 'evaluate' or args.action == 'all':
        run_evaluation(use_optimized=not args.no_optimize)
    
    if args.action == 'compare-cnn' or args.action == 'all':
        run_cnn_comparison()
    
    logger.info("Completed all requested actions.")

if __name__ == "__main__":
    main() 