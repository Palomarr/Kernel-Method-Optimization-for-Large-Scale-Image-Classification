#!/usr/bin/env python
"""
Run comprehensive evaluations on the Sign Language MNIST dataset.
This script runs the full benchmark using both RFF and Nyström methods.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kernel.kernel_approximation_classifier import KernelApproximationClassifier
from evaluation.benchmark import run_benchmark, plot_benchmark_results, plot_confusion_matrix
from data_utils.data_loader import load_reshaped_data
from evaluation.experiments import run_comparative_analysis

def main():
    """Run comprehensive evaluations on the Sign Language MNIST dataset."""
    start_time = time.time()
    logger.info("Starting comprehensive evaluation on Sign Language MNIST dataset...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Part 1: Run the full comparative analysis from experiments.py
    logger.info("Running full comparative analysis using optimized parameters...")
    run_comparative_analysis(use_optimized_params=True)
    
    # Part 2: Generate additional visualizations
    logger.info("Generating additional visualizations...")
    
    # Load benchmark results
    try:
        with open('results/benchmark_results.pkl', 'rb') as f:
            benchmark_results = pickle.load(f)
        
        # Generate confusion matrices for each method
        logger.info("Generating confusion matrices...")
        for method_name, conf_matrices in benchmark_results['confusion_matrices'].items():
            # Get largest sample size data
            largest_size = max(conf_matrices.keys())
            cm = conf_matrices[largest_size]
            
            # Get class labels (assuming 0-9 for the digits in Sign Language MNIST)
            class_labels = list(range(10))
            
            # Create and save confusion matrix plot
            fig = plot_confusion_matrix(
                cm, class_labels, 
                title=f"Confusion Matrix - {method_name} (n={largest_size})",
                figsize=(10, 8)
            )
            fig.savefig(f"results/confusion_matrix_{method_name.replace(' ', '_').lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Generate additional performance comparison plots
        logger.info("Generating performance comparison plots...")
        
        # Plot accuracy vs. training time
        plt.figure(figsize=(10, 6))
        for method in benchmark_results['accuracy']:
            # Get non-NaN values
            valid_indices = [i for i, acc in enumerate(benchmark_results['accuracy'][method]) 
                            if not np.isnan(acc)]
            if not valid_indices:
                continue
                
            accuracies = [benchmark_results['accuracy'][method][i] for i in valid_indices]
            train_times = [benchmark_results['training_time'][method][i] for i in valid_indices]
            
            plt.scatter(train_times, accuracies, label=method, s=100)
            
        plt.xlabel('Training Time (s)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Training Time Trade-off')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/accuracy_vs_training_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot accuracy vs. prediction time
        plt.figure(figsize=(10, 6))
        for method in benchmark_results['accuracy']:
            # Get non-NaN values
            valid_indices = [i for i, acc in enumerate(benchmark_results['accuracy'][method]) 
                            if not np.isnan(acc)]
            if not valid_indices:
                continue
                
            accuracies = [benchmark_results['accuracy'][method][i] for i in valid_indices]
            pred_times = [benchmark_results['prediction_time'][method][i] for i in valid_indices]
            
            plt.scatter(pred_times, accuracies, label=method, s=100)
            
        plt.xlabel('Prediction Time (s)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Prediction Time Trade-off')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/accuracy_vs_prediction_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary table as CSV
        logger.info("Generating summary table...")
        import pandas as pd
        
        # Create summary data
        summary_data = []
        sample_sizes = benchmark_results['sample_sizes']
        
        for method in benchmark_results['accuracy']:
            for i, size in enumerate(sample_sizes):
                if i < len(benchmark_results['accuracy'][method]) and not np.isnan(benchmark_results['accuracy'][method][i]):
                    summary_data.append({
                        'Method': method,
                        'Sample Size': size,
                        'Accuracy': benchmark_results['accuracy'][method][i],
                        'Training Time (s)': benchmark_results['training_time'][method][i],
                        'Prediction Time (s)': benchmark_results['prediction_time'][method][i]
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv('results/method_comparison_summary.csv', index=False)
            
            # Print the best performing methods
            logger.info("\nBest performing methods by accuracy:")
            best_methods = df.sort_values('Accuracy', ascending=False).head(5)
            for _, row in best_methods.iterrows():
                logger.info(f"{row['Method']} with {row['Sample Size']} samples: "
                           f"Accuracy = {row['Accuracy']:.4f}, "
                           f"Training = {row['Training Time (s)']:.2f}s, "
                           f"Prediction = {row['Prediction Time (s)']:.4f}s")
    
    except Exception as e:
        logger.error(f"Error generating additional visualizations: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Comprehensive evaluation completed in {total_time:.2f} seconds.")
    logger.info("Results saved to the 'results' directory.")

if __name__ == "__main__":
    main() 