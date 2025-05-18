#!/usr/bin/env python
"""
Hyperparameter optimization for kernel approximation methods.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import time
import logging
from tqdm import tqdm

from models.kernel_approximation_classifier import KernelApproximationClassifier
from utils.data_loader import load_reshaped_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def flatten_data(X):
    """Flatten 3D data to 2D."""
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X

def plot_optimization_results(results_df, method_name):
    """
    Plot optimization results to visualize parameter relationships.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing optimization results
    method_name : str
        Name of the method (RFF or Nyström)
    """
    # Create output directory
    os.makedirs('results/figures', exist_ok=True)
    
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy vs. n_components
    sns.boxplot(x='n_components', y='accuracy', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title(f'{method_name}: Accuracy by Number of Components')
    axes[0, 0].set_ylim(0, 1)
    
    # Plot accuracy vs. gamma
    sns.boxplot(x='gamma', y='accuracy', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title(f'{method_name}: Accuracy by Gamma Value')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot training time vs. n_components
    sns.boxplot(x='n_components', y='training_time', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title(f'{method_name}: Training Time by Number of Components')
    
    # For Nyström, plot accuracy by sampling method
    if 'sampling' in results_df.columns:
        sns.boxplot(x='sampling', y='accuracy', data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title(f'{method_name}: Accuracy by Sampling Method')
        axes[1, 1].set_ylim(0, 1)
    else:
        # Plot accuracy vs. learning rate
        sns.boxplot(x='learning_rate', y='accuracy', data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title(f'{method_name}: Accuracy by Learning Rate')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/{method_name.lower()}_parameter_analysis.png', dpi=300)
    
    # Create heatmap for component and gamma interaction
    plt.figure(figsize=(10, 8))
    pivot_table = results_df.pivot_table(
        values='accuracy', 
        index='gamma', 
        columns='n_components', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
    plt.title(f'{method_name}: Accuracy by Components and Gamma')
    plt.tight_layout()
    plt.savefig(f'results/figures/{method_name.lower()}_heatmap.png', dpi=300)
    
    # Create the top performing configurations table figure
    plt.figure(figsize=(12, 6))
    top_configs = results_df.nlargest(10, 'accuracy')
    plt.axis('off')
    table_data = top_configs[['accuracy', 'n_components', 'gamma', 'learning_rate', 'batch_size', 'max_iter']]
    if 'sampling' in top_configs.columns:
        table_data['sampling'] = top_configs['sampling']
    
    plt.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    plt.title(f'Top 10 {method_name} Configurations')
    plt.tight_layout()
    plt.savefig(f'results/figures/{method_name.lower()}_top_configs.png', dpi=300)
    
    logger.info(f"Saved {method_name} optimization analysis plots to results/figures/")

def optimize_rff_parameters(X_train, y_train, X_val, y_val, n_iter=50, use_cv=True, n_splits=3):
    """
    Optimize Random Fourier Features parameters using adaptive random search.
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    X_val : array-like
        Validation data
    y_val : array-like
        Validation labels
    n_iter : int
        Number of parameter settings sampled in random search
    use_cv : bool
        Whether to use cross-validation for evaluation
    n_splits : int
        Number of cross-validation splits
    
    Returns:
    --------
    best_params : dict
        Best parameters found
    best_score : float
        Validation accuracy with best parameters
    """
    logger.info("Starting RFF parameter optimization...")
    
    # Expanded parameter space to search - more component values and wider gamma range
    n_components_list = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
    gamma_list = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    bs_list = [32, 64, 128, 256, 512]
    max_iter_list = [100, 200, 300, 500, 1000]
    
    # Reshape data if needed
    X_train = flatten_data(X_train)
    X_val = flatten_data(X_val)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Reduce dataset size for faster optimization if needed
    max_samples = 10000
    if X_train_scaled.shape[0] > max_samples:
        logger.info(f"Training data is large ({X_train_scaled.shape[0]} samples). Using a {max_samples} sample subset for optimization.")
        indices = np.random.choice(X_train_scaled.shape[0], max_samples, replace=False)
        X_train_subset = X_train_scaled[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_scaled
        y_train_subset = y_train
        
    # Set up cross-validation if requested
    if use_cv:
        logger.info(f"Using {n_splits}-fold cross-validation for more reliable evaluation")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initial random search to cover wide parameter space
    best_score = 0
    best_params = None
    results = []
    
    # Two-stage search: first broad search, then focused search
    logger.info("Stage 1: Broad parameter search")
    stage1_n_iter = n_iter // 2
    
    for i in tqdm(range(stage1_n_iter), desc="RFF Optimization Stage 1"):
        # Sample random parameters
        n_components = np.random.choice(n_components_list)
        gamma = np.random.choice(gamma_list)
        learning_rate = np.random.choice(lr_list)
        batch_size = np.random.choice(bs_list)
        max_iter = np.random.choice(max_iter_list)
        
        approx_params = {'n_components': n_components, 'gamma': gamma}
        optimizer_params = {'learning_rate': learning_rate, 'batch_size': batch_size, 'max_iter': max_iter}
        
        # Create classifier with these parameters
        params = {
            'approximation': 'rff',
            'approx_params': approx_params,
            'optimizer_params': optimizer_params,
            'random_state': 42
        }
        
        clf = KernelApproximationClassifier(**params)
        
        # Evaluate performance
        try:
            if use_cv:
                # Cross-validation evaluation
                cv_scores = []
                cv_times = []
                
                for train_idx, val_idx in cv.split(X_train_subset, y_train_subset):
                    X_cv_train, X_cv_val = X_train_subset[train_idx], X_train_subset[val_idx]
                    y_cv_train, y_cv_val = y_train_subset[train_idx], y_train_subset[val_idx]
                    
                    start_time = time.time()
                    clf.fit(X_cv_train, y_cv_train)
                    train_time = time.time() - start_time
                    
                    y_pred = clf.predict(X_cv_val)
                    score = accuracy_score(y_cv_val, y_pred)
                    
                    cv_scores.append(score)
                    cv_times.append(train_time)
                
                score = np.mean(cv_scores)
                train_time = np.mean(cv_times)
            else:
                # Single validation set evaluation
                start_time = time.time()
                clf.fit(X_train_subset, y_train_subset)
                train_time = time.time() - start_time
                
                y_pred = clf.predict(X_val_scaled)
                score = accuracy_score(y_val, y_pred)
            
            # Record results
            result = {
                'iteration': i,
                'accuracy': score,
                'training_time': train_time,
                'n_components': n_components,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_iter': max_iter
            }
            results.append(result)
            
            # Update best parameters if needed
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} with params: n_components={n_components}, gamma={gamma}, lr={learning_rate}")
                
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            continue
    
    # Save first stage results
    df_stage1 = pd.DataFrame(results)
    df_stage1.to_csv('results/rff_optimization_stage1_results.csv', index=False)
    
    # Analyze stage 1 results to find promising regions in parameter space
    logger.info("Stage 2: Focused parameter search based on promising regions")
    
    # Identify promising parameter regions from stage 1
    top_configs = df_stage1.nlargest(5, 'accuracy')
    
    # Define focused parameter ranges based on promising regions
    promising_n_components = top_configs['n_components'].unique().tolist()
    promising_gammas = top_configs['gamma'].unique().tolist()
    promising_lrs = top_configs['learning_rate'].unique().tolist()
    
    # Add nearby values for fine-tuning
    focused_n_components = []
    for n in promising_n_components:
        focused_n_components.extend([max(100, int(n * 0.8)), n, int(n * 1.2)])
    
    focused_gammas = []
    for g in promising_gammas:
        focused_gammas.extend([g * 0.5, g, g * 2])
    
    focused_lrs = []
    for lr in promising_lrs:
        focused_lrs.extend([lr * 0.5, lr, lr * 2])
    
    # Remove duplicates and sort
    focused_n_components = sorted(list(set(focused_n_components)))
    focused_gammas = sorted(list(set(focused_gammas)))
    focused_lrs = sorted(list(set(focused_lrs)))
    
    logger.info(f"Focused search on components: {focused_n_components}")
    logger.info(f"Focused search on gammas: {focused_gammas}")
    logger.info(f"Focused search on learning rates: {focused_lrs}")
    
    # Stage 2: Focused search
    stage2_n_iter = n_iter - stage1_n_iter
    
    for i in tqdm(range(stage2_n_iter), desc="RFF Optimization Stage 2"):
        # Sample from promising parameter regions
        n_components = np.random.choice(focused_n_components)
        gamma = np.random.choice(focused_gammas)
        learning_rate = np.random.choice(focused_lrs)
        batch_size = np.random.choice(bs_list)
        max_iter = np.random.choice(max_iter_list)
        
        approx_params = {'n_components': n_components, 'gamma': gamma}
        optimizer_params = {'learning_rate': learning_rate, 'batch_size': batch_size, 'max_iter': max_iter}
        
        # Create classifier with these parameters
        params = {
            'approximation': 'rff',
            'approx_params': approx_params,
            'optimizer_params': optimizer_params,
            'random_state': 42
        }
        
        clf = KernelApproximationClassifier(**params)
        
        # Evaluate performance
        try:
            if use_cv:
                # Cross-validation evaluation
                cv_scores = []
                cv_times = []
                
                for train_idx, val_idx in cv.split(X_train_subset, y_train_subset):
                    X_cv_train, X_cv_val = X_train_subset[train_idx], X_train_subset[val_idx]
                    y_cv_train, y_cv_val = y_train_subset[train_idx], y_train_subset[val_idx]
                    
                    start_time = time.time()
                    clf.fit(X_cv_train, y_cv_train)
                    train_time = time.time() - start_time
                    
                    y_pred = clf.predict(X_cv_val)
                    score = accuracy_score(y_cv_val, y_pred)
                    
                    cv_scores.append(score)
                    cv_times.append(train_time)
                
                score = np.mean(cv_scores)
                train_time = np.mean(cv_times)
            else:
                # Single validation set evaluation
                start_time = time.time()
                clf.fit(X_train_subset, y_train_subset)
                train_time = time.time() - start_time
                
                y_pred = clf.predict(X_val_scaled)
                score = accuracy_score(y_val, y_pred)
            
            # Record results
            result = {
                'iteration': i + stage1_n_iter,  # Continue numbering from stage 1
                'accuracy': score,
                'training_time': train_time,
                'n_components': n_components,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_iter': max_iter
            }
            results.append(result)
            
            # Update best parameters if needed
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} with params: n_components={n_components}, gamma={gamma}, lr={learning_rate}")
                
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            continue
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/rff_optimization_results.csv', index=False)
    
    # Save best parameters
    with open('results/rff_best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    # Plot optimization results
    plot_optimization_results(results_df, "RFF")
    
    logger.info(f"RFF optimization complete. Best accuracy: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Final evaluation of best model on the full validation set
    try:
        clf = KernelApproximationClassifier(**best_params)
        clf.fit(X_train_subset, y_train_subset)
        y_pred = clf.predict(X_val_scaled)
        final_score = accuracy_score(y_val, y_pred)
        logger.info(f"Final validation accuracy with best parameters: {final_score:.4f}")
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")
    
    return best_params, best_score

def optimize_nystrom_parameters(X_train, y_train, X_val, y_val, n_iter=50, use_cv=True, n_splits=3):
    """
    Optimize Nyström method parameters using adaptive random search.
    
    Parameters:
    -----------
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    X_val : array-like
        Validation data
    y_val : array-like
        Validation labels
    n_iter : int
        Number of parameter settings sampled in random search
    use_cv : bool
        Whether to use cross-validation for evaluation
    n_splits : int
        Number of cross-validation splits
    
    Returns:
    --------
    best_params : dict
        Best parameters found
    best_score : float
        Validation accuracy with best parameters
    """
    logger.info("Starting Nyström parameter optimization...")
    
    # MODIFIED: Smaller parameter space with safer gamma values
    n_components_list = [100, 250, 500, 750, 1000, 1500]
    gamma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # Removed very large values
    sampling_methods = ['uniform', 'kmeans']
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    bs_list = [32, 64, 128, 256]
    max_iter_list = [100, 200, 300, 500]
    
    # Reshape data if needed
    X_train = flatten_data(X_train)
    X_val = flatten_data(X_val)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # MODIFIED: Use a smaller subset for Nyström to prevent memory issues
    max_samples = 5000  # Reduced from 10000
    if X_train_scaled.shape[0] > max_samples:
        logger.info(f"Training data is large ({X_train_scaled.shape[0]} samples). Using a {max_samples} sample subset for optimization.")
        indices = np.random.choice(X_train_scaled.shape[0], max_samples, replace=False)
        X_train_subset = X_train_scaled[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_scaled
        y_train_subset = y_train
    
    # Set up cross-validation if requested
    if use_cv:
        logger.info(f"Using {n_splits}-fold cross-validation for more reliable evaluation")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initial random search to cover wide parameter space
    best_score = 0
    best_params = None
    results = []
    
    # MODIFIED: Function to check if a parameter combination is likely to cause memory issues
    def check_memory_risk(n_components, gamma, n_samples, n_features):
        # Estimate memory footprint in GB
        component_memory = n_components * n_features * 8 / (1024**3)  # 8 bytes per float64
        kmeans_memory = n_components * n_features * 8 / (1024**3)
        kernel_memory = n_components * n_components * 8 / (1024**3)
        sample_transform_memory = n_samples * n_components * 8 / (1024**3)
        
        total_memory = component_memory + kmeans_memory + kernel_memory + sample_transform_memory
        
        # Check risky combinations
        if total_memory > 4.0:  # More than 4GB is risky
            return True
        if gamma > 0.5 and n_components > 500:  # High gamma with many components
            return True
        if n_components > 1000 and n_samples > 2000:  # Large matrices
            return True
        return False
    
    # Two-stage search: first broad search, then focused search
    logger.info("Stage 1: Broad parameter search")
    stage1_n_iter = n_iter // 2
    
    for i in tqdm(range(stage1_n_iter), desc="Nyström Optimization Stage 1"):
        try:
            # Sample random parameters
            n_components = np.random.choice(n_components_list)
            gamma = np.random.choice(gamma_list)
            sampling = np.random.choice(sampling_methods)
            learning_rate = np.random.choice(lr_list)
            batch_size = np.random.choice(bs_list)
            max_iter = np.random.choice(max_iter_list)
            
            # Skip parameter combinations likely to cause memory issues
            if check_memory_risk(n_components, gamma, len(X_train_subset), X_train_subset.shape[1]):
                logger.warning(f"Skipping potentially problematic parameter combination: n_components={n_components}, gamma={gamma}")
                continue
            
            # Skip large component numbers with small gamma values (known to cause instability)
            if n_components > 1000 and gamma < 0.001:
                gamma = 0.001  # Use a safer gamma value for large component numbers
            
            approx_params = {'n_components': n_components, 'gamma': gamma, 'sampling': sampling}
            optimizer_params = {'learning_rate': learning_rate, 'batch_size': batch_size, 'max_iter': max_iter}
            
            # Create classifier with these parameters
            params = {
                'approximation': 'nystrom',
                'approx_params': approx_params,
                'optimizer_params': optimizer_params,
                'random_state': 42
            }
            
            clf = KernelApproximationClassifier(**params)
            
            # MODIFIED: Add timeout protection
            max_time_per_fit = 60  # Maximum time allowed for a single fit (seconds)
            
            # Evaluate performance
            if use_cv:
                # Cross-validation evaluation
                cv_scores = []
                cv_times = []
                
                for train_idx, val_idx in cv.split(X_train_subset, y_train_subset):
                    X_cv_train, X_cv_val = X_train_subset[train_idx], X_train_subset[val_idx]
                    y_cv_train, y_cv_val = y_train_subset[train_idx], y_train_subset[val_idx]
                    
                    start_time = time.time()
                    # Add timeout protection
                    try:
                        clf.fit(X_cv_train, y_cv_train)
                        train_time = time.time() - start_time
                        if train_time > max_time_per_fit:
                            logger.warning(f"Training took too long ({train_time:.2f}s), skipping configuration")
                            raise TimeoutError("Training timeout")
                        
                        y_pred = clf.predict(X_cv_val)
                        score = accuracy_score(y_cv_val, y_pred)
                        
                        cv_scores.append(score)
                        cv_times.append(train_time)
                    except Exception as e:
                        logger.warning(f"Error in CV fold: {str(e)}")
                        raise
                
                if len(cv_scores) == 0:
                    logger.warning(f"No valid CV scores for configuration")
                    continue
                    
                score = np.mean(cv_scores)
                train_time = np.mean(cv_times)
            else:
                # Single validation set evaluation
                start_time = time.time()
                clf.fit(X_train_subset, y_train_subset)
                train_time = time.time() - start_time
                
                # Skip if training took too long
                if train_time > max_time_per_fit:
                    logger.warning(f"Training took too long ({train_time:.2f}s), skipping configuration")
                    continue
                
                y_pred = clf.predict(X_val_scaled)
                score = accuracy_score(y_val, y_pred)
            
            # Record results
            result = {
                'iteration': i,
                'accuracy': score,
                'training_time': train_time,
                'n_components': n_components,
                'gamma': gamma,
                'sampling': sampling,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_iter': max_iter
            }
            results.append(result)
            
            # Update best parameters if needed
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} with params: n_components={n_components}, gamma={gamma}, sampling={sampling}, lr={learning_rate}")
                
        except Exception as e:
            logger.error(f"Error during optimization iteration {i}: {str(e)}")
            continue
    
    # Save first stage results
    if not results:
        logger.error("No successful configurations found in Stage 1. Cannot proceed to Stage 2.")
        return None, 0
    
    df_stage1 = pd.DataFrame(results)
    df_stage1.to_csv('results/nystrom_optimization_stage1_results.csv', index=False)
    
    # Analyze stage 1 results to find promising regions in parameter space
    logger.info("Stage 2: Focused parameter search based on promising regions")
    
    # Identify promising parameter regions from stage 1
    top_configs = df_stage1.nlargest(min(5, len(df_stage1)), 'accuracy')
    
    # Define focused parameter ranges based on promising regions
    promising_n_components = top_configs['n_components'].unique().tolist()
    promising_gammas = top_configs['gamma'].unique().tolist()
    promising_sampling = top_configs['sampling'].unique().tolist()
    promising_lrs = top_configs['learning_rate'].unique().tolist()
    
    # Add nearby values for fine-tuning
    focused_n_components = []
    for n in promising_n_components:
        focused_n_components.extend([max(100, int(n * 0.8)), n, int(n * 1.2)])
    
    focused_gammas = []
    for g in promising_gammas:
        focused_gammas.extend([g * 0.5, g, g * 2])
    
    focused_lrs = []
    for lr in promising_lrs:
        focused_lrs.extend([lr * 0.5, lr, lr * 2])
    
    # Remove duplicates and sort
    focused_n_components = sorted(list(set(focused_n_components)))
    focused_gammas = sorted(list(set(focused_gammas)))
    focused_lrs = sorted(list(set(focused_lrs)))
    
    logger.info(f"Focused search on components: {focused_n_components}")
    logger.info(f"Focused search on gammas: {focused_gammas}")
    logger.info(f"Focused search on sampling: {promising_sampling}")
    logger.info(f"Focused search on learning rates: {focused_lrs}")
    
    # Stage 2: Focused search
    stage2_n_iter = n_iter - stage1_n_iter
    
    for i in tqdm(range(stage2_n_iter), desc="Nyström Optimization Stage 2"):
        try:
            # Sample from promising parameter regions
            n_components = np.random.choice(focused_n_components)
            gamma = np.random.choice(focused_gammas)
            sampling = np.random.choice(promising_sampling)
            learning_rate = np.random.choice(focused_lrs)
            batch_size = np.random.choice(bs_list)
            max_iter = np.random.choice(max_iter_list)
            
            # Skip parameter combinations likely to cause memory issues
            if check_memory_risk(n_components, gamma, len(X_train_subset), X_train_subset.shape[1]):
                logger.warning(f"Skipping potentially problematic parameter combination: n_components={n_components}, gamma={gamma}")
                continue
            
            # Skip large component numbers with small gamma values (known to cause instability)
            if n_components > 1000 and gamma < 0.001:
                gamma = 0.001  # Use a safer gamma value for large component numbers
            
            approx_params = {'n_components': n_components, 'gamma': gamma, 'sampling': sampling}
            optimizer_params = {'learning_rate': learning_rate, 'batch_size': batch_size, 'max_iter': max_iter}
            
            # Create classifier with these parameters
            params = {
                'approximation': 'nystrom',
                'approx_params': approx_params,
                'optimizer_params': optimizer_params,
                'random_state': 42
            }
            
            clf = KernelApproximationClassifier(**params)
            
            # Evaluate performance with timeout protection
            if use_cv:
                # Cross-validation evaluation
                cv_scores = []
                cv_times = []
                
                for train_idx, val_idx in cv.split(X_train_subset, y_train_subset):
                    X_cv_train, X_cv_val = X_train_subset[train_idx], X_train_subset[val_idx]
                    y_cv_train, y_cv_val = y_train_subset[train_idx], y_train_subset[val_idx]
                    
                    start_time = time.time()
                    try:
                        clf.fit(X_cv_train, y_cv_train)
                        train_time = time.time() - start_time
                        
                        if train_time > max_time_per_fit:
                            logger.warning(f"Training took too long ({train_time:.2f}s), skipping configuration")
                            raise TimeoutError("Training timeout")
                        
                        y_pred = clf.predict(X_cv_val)
                        score = accuracy_score(y_cv_val, y_pred)
                        
                        cv_scores.append(score)
                        cv_times.append(train_time)
                    except Exception as e:
                        logger.warning(f"Error in CV fold: {str(e)}")
                        raise
                
                if len(cv_scores) == 0:
                    logger.warning(f"No valid CV scores for configuration")
                    continue
                    
                score = np.mean(cv_scores)
                train_time = np.mean(cv_times)
            else:
                # Single validation set evaluation
                start_time = time.time()
                clf.fit(X_train_subset, y_train_subset)
                train_time = time.time() - start_time
                
                # Skip if training took too long
                if train_time > max_time_per_fit:
                    logger.warning(f"Training took too long ({train_time:.2f}s), skipping configuration")
                    continue
                
                y_pred = clf.predict(X_val_scaled)
                score = accuracy_score(y_val, y_pred)
            
            # Record results
            result = {
                'iteration': i + stage1_n_iter,  # Continue numbering from stage 1
                'accuracy': score,
                'training_time': train_time,
                'n_components': n_components,
                'gamma': gamma,
                'sampling': sampling,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_iter': max_iter
            }
            results.append(result)
            
            # Update best parameters if needed
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f} with params: n_components={n_components}, gamma={gamma}, sampling={sampling}, lr={learning_rate}")
                
        except Exception as e:
            logger.error(f"Error during optimization iteration {i + stage1_n_iter}: {str(e)}")
            continue
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/nystrom_optimization_results.csv', index=False)
    
    # Check if we found any good parameters
    if best_params is None:
        logger.error("No successful configurations found. Unable to determine best parameters.")
        # Return a safe default configuration
        default_params = {
            'approximation': 'nystrom',
            'approx_params': {'n_components': 500, 'gamma': 0.01, 'sampling': 'kmeans'},
            'optimizer_params': {'learning_rate': 0.01, 'batch_size': 128, 'max_iter': 300},
            'random_state': 42
        }
        return default_params, 0
    
    # Save best parameters
    with open('results/nystrom_best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    # Plot optimization results
    try:
        plot_optimization_results(results_df, "Nyström")
    except Exception as e:
        logger.error(f"Error plotting optimization results: {str(e)}")
    
    logger.info(f"Nyström optimization complete. Best accuracy: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Final evaluation of best model on the full validation set
    try:
        clf = KernelApproximationClassifier(**best_params)
        clf.fit(X_train_subset, y_train_subset)
        y_pred = clf.predict(X_val_scaled)
        final_score = accuracy_score(y_val, y_pred)
        logger.info(f"Final validation accuracy with best parameters: {final_score:.4f}")
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")
    
    return best_params, best_score

def main():
    """Run hyperparameter optimization with enhanced search and analysis."""
    # Create results directory
    os.makedirs('results/figures', exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_reshaped_data()
    
    # Experiment settings
    n_iter = 50  # Increased from 20 to 50 for better coverage
    use_cv = True  # Use cross-validation for more reliable results
    n_splits = 3   # Number of cross-validation splits
    
    # Run RFF optimization first (more reliable)
    logger.info("\n" + "="*50)
    logger.info("STARTING RFF OPTIMIZATION")
    logger.info("="*50)
    try:
        rff_best_params, rff_best_score = optimize_rff_parameters(
            X_train, y_train, X_val, y_val, n_iter=n_iter, use_cv=use_cv, n_splits=n_splits)
        
        # Immediately save these results
        logger.info(f"RFF Best Accuracy: {rff_best_score:.4f}")
        logger.info(f"RFF Best Parameters: {rff_best_params}")
    except Exception as e:
        logger.error(f"RFF optimization failed with error: {str(e)}")
        rff_best_params = None
        rff_best_score = 0
    
    # Run Nyström optimization with more conservative settings
    logger.info("\n" + "="*50)
    logger.info("STARTING NYSTRÖM OPTIMIZATION")
    logger.info("="*50)
    try:
        nystrom_best_params, nystrom_best_score = optimize_nystrom_parameters(
            X_train, y_train, X_val, y_val, n_iter=n_iter, use_cv=use_cv, n_splits=n_splits)
        
        logger.info(f"Nyström Best Accuracy: {nystrom_best_score:.4f}")
        logger.info(f"Nyström Best Parameters: {nystrom_best_params}")
    except Exception as e:
        logger.error(f"Nyström optimization failed with error: {str(e)}")
        nystrom_best_params = {
            'approximation': 'nystrom',
            'approx_params': {'n_components': 500, 'gamma': 0.01, 'sampling': 'kmeans'},
            'optimizer_params': {'learning_rate': 0.01, 'batch_size': 128, 'max_iter': 300},
            'random_state': 42
        }
        nystrom_best_score = 0
        logger.warning("Using default Nyström parameters due to optimization failure")
    
    # Print summary
    logger.info("\n=== Optimization Results ===")
    if rff_best_params:
        logger.info(f"RFF Best Accuracy: {rff_best_score:.4f}")
        logger.info(f"RFF Best Parameters: {rff_best_params}")
    else:
        logger.warning("RFF optimization failed to find parameters")
        
    if nystrom_best_score > 0:
        logger.info(f"Nyström Best Accuracy: {nystrom_best_score:.4f}")
        logger.info(f"Nyström Best Parameters: {nystrom_best_params}")
    else:
        logger.warning("Using default Nyström parameters")
    
    # Evaluate best models on test set only if optimization was successful
    try:
        logger.info("\nEvaluating best models on test set:")
        
        # Scale test data
        X_test_reshaped = flatten_data(X_test)
        scaler = StandardScaler()
        X_train_flat = flatten_data(X_train)
        X_test_scaled = scaler.fit_transform(X_test_reshaped)
        
        # Evaluate RFF if available
        if rff_best_params:
            logger.info("Evaluating best RFF model:")
            rff_clf = KernelApproximationClassifier(**rff_best_params)
            rff_clf.fit(X_train_flat, y_train)
            rff_pred = rff_clf.predict(X_test_scaled)
            rff_test_acc = accuracy_score(y_test, rff_pred)
            logger.info(f"RFF Test Accuracy: {rff_test_acc:.4f}")
        
        # Evaluate Nyström
        if nystrom_best_score > 0:
            logger.info("Evaluating best Nyström model:")
            nystrom_clf = KernelApproximationClassifier(**nystrom_best_params)
            nystrom_clf.fit(X_train_flat, y_train)
            nystrom_pred = nystrom_clf.predict(X_test_scaled)
            nystrom_test_acc = accuracy_score(y_test, nystrom_pred)
            logger.info(f"Nyström Test Accuracy: {nystrom_test_acc:.4f}")
        
    except Exception as e:
        logger.error(f"Error in test evaluation: {str(e)}")
        
    logger.info("\nHyperparameter optimization completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    main() 