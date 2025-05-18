#!/usr/bin/env python
"""
Compare kernel approximation methods with CNN baseline on Sign Language MNIST.
This script runs benchmarks comparing RFF, Nyström, and CNN performance.
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import pickle
import psutil
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from kernel_methods.models.kernel_approximation_classifier import KernelApproximationClassifier
from models.neural.cnn_baseline import PyTorchCNNClassifier
from kernel_methods.utils.data_loader import load_reshaped_data
from evaluation.benchmark import plot_confusion_matrix

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    """Run comparison between kernel methods and CNN baseline."""
    start_time = time.time()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    logger.info("Loading Sign Language MNIST dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_reshaped_data()
    
    # Print dataset shapes
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape if X_val is not None else 'None'}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    # Preprocess data for kernel methods
    logger.info("Preprocessing data...")
    
    if X_train.ndim > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
        X_test_flat = X_test
    
    # Scale data for kernel methods
    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    if X_val is not None:
        X_val_flat_scaled = scaler.transform(X_val_flat)
    X_test_flat_scaled = scaler.transform(X_test_flat)
    
    # Define maximum sample size to use
    max_samples = min(10000, X_train.shape[0])
    logger.info(f"Using maximum {max_samples} samples for training")
    
    # Results storage
    results = {
        'method': [],
        'accuracy': [],
        'training_time': [],
        'prediction_time': [],
        'peak_memory_mb': []
    }
    
    # Try to load optimized parameters
    rff_params = {
        'n_components': 1000,
        'gamma': 0.01
    }
    nystrom_params = {
        'n_components': 500,
        'gamma': 0.01,
        'sampling': 'kmeans'
    }
    optimizer_params = {
        'learning_rate': 0.01,
        'batch_size': 128,
        'max_iter': 300
    }
    
    try:
        if os.path.exists('results/rff_best_params.pkl'):
            with open('results/rff_best_params.pkl', 'rb') as f:
                rff_best_params = pickle.load(f)
            rff_params = rff_best_params['approx_params']
            rff_optimizer_params = rff_best_params['optimizer_params']
            logger.info(f"Loaded optimized RFF parameters: {rff_params}")
        else:
            logger.info("No optimized RFF parameters found. Using defaults.")
            rff_optimizer_params = optimizer_params
        
        if os.path.exists('results/nystrom_best_params.pkl'):
            with open('results/nystrom_best_params.pkl', 'rb') as f:
                nystrom_best_params = pickle.load(f)
            nystrom_params = nystrom_best_params['approx_params']
            nystrom_optimizer_params = nystrom_best_params['optimizer_params']
            logger.info(f"Loaded optimized Nyström parameters: {nystrom_params}")
        else:
            logger.info("No optimized Nyström parameters found. Using defaults.")
            nystrom_optimizer_params = optimizer_params
    except Exception as e:
        logger.error(f"Error loading optimized parameters: {e}")
        logger.info("Using default parameters instead.")
        rff_optimizer_params = optimizer_params
        nystrom_optimizer_params = optimizer_params
    
    # Evaluate Random Fourier Features
    logger.info("\n" + "="*50)
    logger.info("Evaluating Random Fourier Features")
    logger.info("="*50)
    
    try:
        # Create RFF classifier
        rff_clf = KernelApproximationClassifier(
            approximation='rff',
            approx_params=rff_params,
            optimizer_params=rff_optimizer_params
        )
        
        # Measure memory before training
        gc.collect()
        mem_before = get_memory_usage()
        
        # Train RFF classifier
        start_train = time.time()
        rff_clf.fit(X_train_flat_scaled[:max_samples], y_train[:max_samples])
        train_time = time.time() - start_train
        
        # Measure peak memory
        mem_after = get_memory_usage()
        peak_memory = mem_after - mem_before
        
        # Make predictions
        start_pred = time.time()
        y_pred = rff_clf.predict(X_test_flat_scaled)
        pred_time = time.time() - start_pred
        
        # Calculate accuracy
        accuracy = np.mean(y_test == y_pred)
        
        # Save results
        results['method'].append(f"RFF ({rff_params['n_components']} components)")
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['prediction_time'].append(pred_time)
        results['peak_memory_mb'].append(peak_memory)
        
        # Print results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Training time: {train_time:.2f} seconds")
        logger.info(f"Prediction time: {pred_time:.4f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        fig = plot_confusion_matrix(cm, list(range(10)), 
                                   title=f"Confusion Matrix - RFF ({rff_params['n_components']} components)")
        fig.savefig("results/confusion_matrix_rff.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error evaluating RFF: {e}")
    
    # Evaluate Nyström method
    logger.info("\n" + "="*50)
    logger.info("Evaluating Nyström Method")
    logger.info("="*50)
    
    try:
        # Create Nyström classifier
        nystrom_clf = KernelApproximationClassifier(
            approximation='nystrom',
            approx_params=nystrom_params,
            optimizer_params=nystrom_optimizer_params
        )
        
        # Measure memory before training
        gc.collect()
        mem_before = get_memory_usage()
        
        # Train Nyström classifier
        start_train = time.time()
        nystrom_clf.fit(X_train_flat_scaled[:max_samples], y_train[:max_samples])
        train_time = time.time() - start_train
        
        # Measure peak memory
        mem_after = get_memory_usage()
        peak_memory = mem_after - mem_before
        
        # Make predictions
        start_pred = time.time()
        y_pred = nystrom_clf.predict(X_test_flat_scaled)
        pred_time = time.time() - start_pred
        
        # Calculate accuracy
        accuracy = np.mean(y_test == y_pred)
        
        # Save results
        results['method'].append(f"Nyström ({nystrom_params['n_components']} components)")
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['prediction_time'].append(pred_time)
        results['peak_memory_mb'].append(peak_memory)
        
        # Print results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Training time: {train_time:.2f} seconds")
        logger.info(f"Prediction time: {pred_time:.4f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        fig = plot_confusion_matrix(cm, list(range(10)), 
                                   title=f"Confusion Matrix - Nyström ({nystrom_params['n_components']} components)")
        fig.savefig("results/confusion_matrix_nystrom.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error evaluating Nyström: {e}")
    
    # Evaluate CNN baseline
    logger.info("\n" + "="*50)
    logger.info("Evaluating PyTorch CNN Baseline")
    logger.info("="*50)
    
    try:
        # Check number of classes and range
        unique_classes = np.unique(y_train)
        logger.info(f"Number of unique classes in training set: {len(unique_classes)}")
        logger.info(f"Unique classes: {unique_classes}")
        
        # Fix the class labels to be contiguous
        class_map = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
        y_train_mapped = np.array([class_map[y] for y in y_train])
        if X_val is not None and y_val is not None:
            y_val_mapped = np.array([class_map[y] for y in y_val])
        else:
            y_val_mapped = None
        y_test_mapped = np.array([class_map[y] for y in y_test])
        
        logger.info(f"Remapped classes to contiguous range 0-{len(unique_classes)-1}")
        
        # Try different epochs based on sample size
        epochs = 15 if max_samples >= 5000 else 10
        
        # Create CNN classifier - now using PyTorch
        cnn_clf = PyTorchCNNClassifier(
            input_shape=(28, 28, 1),
            n_classes=len(unique_classes),
            epochs=epochs,
            batch_size=32,
            verbose=1,
            early_stopping=True
        )
        
        # Measure memory before training
        gc.collect()
        mem_before = get_memory_usage()
        
        # Train CNN classifier
        start_train = time.time()
        if X_val is not None:
            cnn_clf.fit(X_train[:max_samples], y_train_mapped[:max_samples], 
                        X_val=X_val, y_val=y_val_mapped)
        else:
            # Use a portion of training data as validation
            train_idx = int(0.8 * max_samples)
            cnn_clf.fit(X_train[:train_idx], y_train_mapped[:train_idx],
                       X_val=X_train[train_idx:max_samples], y_val=y_train_mapped[train_idx:max_samples])
        train_time = time.time() - start_train
        
        # Measure peak memory
        mem_after = get_memory_usage()
        peak_memory = mem_after - mem_before
        
        # Make predictions
        start_pred = time.time()
        y_pred_mapped = cnn_clf.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Map predictions back to original classes
        reverse_map = {new_label: old_label for old_label, new_label in class_map.items()}
        y_pred = np.array([reverse_map[y] for y in y_pred_mapped])
        
        # Calculate accuracy
        accuracy = np.mean(y_test == y_pred)
        
        # Save results
        results['method'].append("PyTorch CNN")
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['prediction_time'].append(pred_time)
        results['peak_memory_mb'].append(peak_memory)
        
        # Print results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Training time: {train_time:.2f} seconds")
        logger.info(f"Prediction time: {pred_time:.4f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        fig = plot_confusion_matrix(cm, list(range(10)), title="Confusion Matrix - PyTorch CNN")
        fig.savefig("results/confusion_matrix_pytorch_cnn.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error evaluating PyTorch CNN: {e}")
    
    # Create comparison plots
    logger.info("\nCreating comparison plots...")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv("results/kernel_vs_cnn_comparison.csv", index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['method'], df['accuracy'])
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['method'], df['training_time'])
    plt.xlabel('Method')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/training_time_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot prediction time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['method'], df['prediction_time'])
    plt.xlabel('Method')
    plt.ylabel('Prediction Time (s)')
    plt.title('Prediction Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/prediction_time_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot memory usage comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['method'], df['peak_memory_mb'])
    plt.xlabel('Method')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/memory_usage_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create a scatter plot of accuracy vs time
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(df['method']):
        plt.scatter(df['training_time'][i], df['accuracy'][i], s=100, label=method)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Time Trade-off')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_time_tradeoff.png", dpi=300, bbox_inches='tight')
    
    # Print summary table
    logger.info("\nMethod Comparison Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Method':<30} {'Accuracy':<10} {'Train Time':<12} {'Pred Time':<12} {'Memory (MB)':<12}")
    logger.info("-" * 80)
    
    for i, method in enumerate(df['method']):
        logger.info(f"{method:<30} {df['accuracy'][i]:<10.4f} {df['training_time'][i]:<12.2f} "
                   f"{df['prediction_time'][i]:<12.4f} {df['peak_memory_mb'][i]:<12.2f}")
    
    total_time = time.time() - start_time
    logger.info(f"\nComparison completed in {total_time:.2f} seconds.")
    logger.info("Results saved to 'results' directory.")

if __name__ == "__main__":
    main() 