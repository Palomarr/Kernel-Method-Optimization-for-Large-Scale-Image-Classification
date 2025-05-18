#!/usr/bin/env python3
"""
Run hyperparameter optimization and benchmark experiments.

This script:
1. Runs hyperparameter optimization for RFF and Nyström methods
2. Runs the benchmark experiments using the optimized parameters
"""
import os
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run optimization and benchmarking."""
    start_time = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Run hyperparameter optimization
    logger.info("Step 1: Running hyperparameter optimization...")
    try:
        subprocess.run(
            ["python", "kernel_methods/hyperparameter_tuning.py"], 
            check=True
        )
        logger.info("Hyperparameter optimization completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Hyperparameter optimization failed with error code {e.returncode}")
        logger.error("Continuing with default parameters.")
    
    # Step 2: Run benchmark with optimized parameters
    logger.info("Step 2: Running benchmark experiments with optimized parameters...")
    try:
        subprocess.run(
            ["python", "kernel_methods/experiments.py"], 
            check=True
        )
        logger.info("Benchmark experiments completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark experiments failed with error code {e.returncode}")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logger.info(f"All tasks completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Print a summary of the results and where to find them
    logger.info("\nResults Summary:")
    logger.info("=================")
    logger.info("Optimization results:")
    logger.info("- RFF parameter optimization: results/rff_optimization_results.csv")
    logger.info("- Nyström parameter optimization: results/nystrom_optimization_results.csv")
    logger.info("\nBenchmark results:")
    logger.info("- Overall comparison: results/benchmark_comparison.png")
    logger.info("- Confusion matrices: results/confusion_matrix_*.png")
    logger.info("- Raw benchmark data: results/benchmark_results.pkl")

if __name__ == "__main__":
    main() 