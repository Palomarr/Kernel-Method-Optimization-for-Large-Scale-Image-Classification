import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import logging

# Use relative imports
from models.kernel_approximation_classifier import KernelApproximationClassifier
from evaluation.benchmark import run_benchmark, plot_benchmark_results, plot_confusion_matrix
from utils.data_loader import load_reshaped_data, load_processed_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comparative_analysis(use_optimized_params=True):
    """
    Run comparative analysis of kernel approximation methods.
    
    Parameters:
    -----------
    use_optimized_params : bool, default=True
        Whether to use parameters optimized by hyperparameter_tuning.py.
        If True, tries to load saved parameters. If not found, uses default parameters.
    """
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_reshaped_data()
    
    # Preprocess data
    print("Preprocessing data...")
    # Print original shapes
    print(f"Original shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    # Reshape the data if it's 3D
    if X_train.ndim > 2:
        n_samples = X_train.shape[0]
        X_train = X_train.reshape(n_samples, -1)
        
    if X_val is not None and X_val.ndim > 2:
        X_val = X_val.reshape(X_val.shape[0], -1)
        
    # Check if X_test is 1D and reshape properly
    if X_test is not None:
        if X_test.ndim == 1:
            print(f"X_test is 1D with shape {X_test.shape}. Reshaping...")
            # Try to determine number of features from X_train
            if X_train.ndim > 1:
                n_features = X_train.shape[1]
                n_samples_test = X_test.shape[0] // n_features
                if n_samples_test > 0 and X_test.size % n_features == 0:
                    X_test = X_test.reshape(n_samples_test, n_features)
                    print(f"Reshaped X_test to: {X_test.shape}")
                else:
                    # If the shapes don't align, load the flattened data instead
                    print("Shapes don't align. Loading processed (flattened) data...")
                    _, _, _, _, X_test, _ = load_processed_data()
                    print(f"Loaded X_test with shape: {X_test.shape}")
            else:
                # If X_train is also 1D, we have a more serious problem
                print("Both X_train and X_test are 1D. Cannot determine proper reshape.")
        elif X_test.ndim > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Ensure all datasets have same number of features
    if X_test.ndim > 1 and X_train.ndim > 1:
        if X_test.shape[1] != X_train.shape[1]:
            print(f"Dimension mismatch! X_train: {X_train.shape}, X_test: {X_test.shape}")
            # Try to reshape X_test to match X_train
            try:
                # If test data has only 1 feature, it might be a completely wrong shape
                # Try to reshape it assuming it's flattened or incorrectly shaped
                total_elements = X_test.size
                target_features = X_train.shape[1]
                if total_elements % target_features == 0:
                    n_samples_test = total_elements // target_features
                    X_test = X_test.reshape(n_samples_test, target_features)
                    print(f"Reshaped X_test to: {X_test.shape}")
                else:
                    print(f"Cannot reshape X_test to match X_train - incompatible dimensions")
            except ValueError as e:
                print(f"Could not reshape: {e}")
    
    print(f"After reshape - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define default parameters
    rff_params = {
        'n_components': 500,
        'gamma': 0.001
    }
    nystrom_params = {
        'n_components': 500,
        'gamma': 0.001,
        'sampling': 'kmeans'
    }
    optimizer_params = {
        'learning_rate': 0.01,
        'batch_size': 64,
        'max_iter': 200
    }
    
    # Try to load optimized parameters if requested
    if use_optimized_params:
        try:
            # Check if optimized parameters exist
            if os.path.exists('results/rff_best_params.pkl'):
                with open('results/rff_best_params.pkl', 'rb') as f:
                    rff_best_params = pickle.load(f)
                rff_params = rff_best_params['approx_params']
                rff_optimizer_params = rff_best_params['optimizer_params']
                print(f"Loaded optimized RFF parameters: {rff_params}")
                print(f"Loaded optimized RFF optimizer parameters: {rff_optimizer_params}")
            else:
                print("No optimized RFF parameters found. Using defaults.")
                rff_optimizer_params = optimizer_params
                
            if os.path.exists('results/nystrom_best_params.pkl'):
                with open('results/nystrom_best_params.pkl', 'rb') as f:
                    nystrom_best_params = pickle.load(f)
                nystrom_params = nystrom_best_params['approx_params']
                nystrom_optimizer_params = nystrom_best_params['optimizer_params']
                print(f"Loaded optimized Nyström parameters: {nystrom_params}")
                print(f"Loaded optimized Nyström optimizer parameters: {nystrom_optimizer_params}")
            else:
                print("No optimized Nyström parameters found. Using defaults.")
                nystrom_optimizer_params = optimizer_params
        except Exception as e:
            print(f"Error loading optimized parameters: {e}")
            print("Using default parameters instead.")
            rff_optimizer_params = optimizer_params
            nystrom_optimizer_params = optimizer_params
    else:
        print("Using default parameters (no optimization).")
        rff_optimizer_params = optimizer_params
        nystrom_optimizer_params = optimizer_params
    
    # Define sample sizes for benchmarking - use realistic sizes based on available data
    total_samples = X_train.shape[0]
    print(f"Total available training samples: {total_samples}")
    
    # Define more sensible sample sizes
    sample_sizes = [1000, 5000, 10000]
    if total_samples > 10000:
        sample_sizes.append(min(20000, total_samples))
    
    # Add full dataset only if it's different from the largest explicitly defined size
    if total_samples > sample_sizes[-1]:
        sample_sizes.append(total_samples)
    
    print(f"Using sample sizes: {sample_sizes}")
    
    # Function to create classifiers dynamically based on sample size
    def create_classifier_configs():
        # Define base classifiers for all methods
        configs = {}
        
        # Add RFF classifiers - these don't depend on sample size
        configs['rff'] = {
            f'RFF ({rff_params["n_components"]} components)': {
                'approx': 'rff',
                'params': rff_params.copy(),
                'optimizer': rff_optimizer_params.copy()
            },
            f'RFF (2000 components)': {
                'approx': 'rff',
                'params': {'n_components': 2000, 'gamma': rff_params['gamma']},
                'optimizer': rff_optimizer_params.copy()
            }
        }
        
        # Add Nyström classifiers - component count depends on sample size
        configs['nystrom'] = {}
        for size in sample_sizes:
            # For Nyström, ensure component count doesn't exceed sample size
            opt_components = min(nystrom_params['n_components'], size - 100)  # Leave margin for safety
            large_components = min(2000, size - 100)  # For the larger version
            
            # Only add if component count is valid
            if opt_components > 50:  # Minimum meaningful component count
                nys_config = {
                    'approx': 'nystrom',
                    'params': {
                        'n_components': opt_components,
                        'gamma': nystrom_params['gamma'],
                        'sampling': nystrom_params.get('sampling', 'kmeans')
                    },
                    'optimizer': nystrom_optimizer_params.copy(),
                    'sample_size': size
                }
                configs['nystrom'][f'Nyström ({opt_components} components)'] = nys_config
            
            # Add larger component version if different and valid
            if large_components > opt_components and large_components > 50:
                nys_large_config = {
                    'approx': 'nystrom',
                    'params': {
                        'n_components': large_components,
                        'gamma': nystrom_params['gamma'],
                        'sampling': nystrom_params.get('sampling', 'kmeans')
                    },
                    'optimizer': nystrom_optimizer_params.copy(),
                    'sample_size': size
                }
                configs['nystrom'][f'Nyström ({large_components} components)'] = nys_large_config
                
        return configs
    
    # Get classifier configurations
    classifier_configs = create_classifier_configs()
    
    # Run benchmarks separately for each algorithm type and sample size
    all_results = {'accuracy': {}, 'training_time': {}, 'prediction_time': {}, 
                   'sample_sizes': [], 'confusion_matrices': {}}
    
    # First run RFF benchmarks (these are simpler and work on all sample sizes)
    print("\nRunning RFF benchmarks...")
    rff_classifiers = {}
    for name, config in classifier_configs['rff'].items():
        rff_classifiers[name] = KernelApproximationClassifier(
            approximation=config['approx'],
            approx_params=config['params'],
            optimizer_params=config['optimizer']
        )
    
    rff_results = run_benchmark(
        rff_classifiers,
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        sample_sizes=sample_sizes,
        n_runs=1
    )
    
    # Merge RFF results into all_results
    for key in ['accuracy', 'training_time', 'prediction_time']:
        all_results[key].update(rff_results[key])
    all_results['sample_sizes'] = rff_results['sample_sizes']
    all_results['confusion_matrices'].update(rff_results['confusion_matrices'])
    
    # Run Nyström benchmarks - handle each sample size separately
    print("\nRunning Nyström benchmarks...")
    for size in sample_sizes:
        nystrom_classifiers = {}
        
        # Filter configurations that work with this sample size
        for name, config in classifier_configs['nystrom'].items():
            if config['sample_size'] == size:
                nystrom_classifiers[name] = KernelApproximationClassifier(
                    approximation=config['approx'],
                    approx_params=config['params'],
                    optimizer_params=config['optimizer']
                )
        
        if nystrom_classifiers:
            # Run benchmarks only for this size to prevent errors
            try:
                print(f"\nBenchmarking Nyström with {size} samples:")
                nys_results = run_benchmark(
                    nystrom_classifiers,
                    X_train_scaled, y_train,
                    X_test_scaled, y_test,
                    sample_sizes=[size],
                    n_runs=1
                )
                
                # Merge Nyström results into all_results
                for name in nystrom_classifiers:
                    if name not in all_results['accuracy']:
                        all_results['accuracy'][name] = []
                        all_results['training_time'][name] = []
                        all_results['prediction_time'][name] = []
                        all_results['confusion_matrices'][name] = {}
                    
                    # Add results for this sample size
                    idx = all_results['sample_sizes'].index(size)
                    while len(all_results['accuracy'][name]) <= idx:
                        all_results['accuracy'][name].append(float('nan'))
                        all_results['training_time'][name].append(float('nan'))
                        all_results['prediction_time'][name].append(float('nan'))
                    
                    all_results['accuracy'][name][idx] = nys_results['accuracy'][name][0]
                    all_results['training_time'][name][idx] = nys_results['training_time'][name][0]
                    all_results['prediction_time'][name][idx] = nys_results['prediction_time'][name][0]
                    all_results['confusion_matrices'][name][size] = nys_results['confusion_matrices'][name][size]
            except Exception as e:
                print(f"Error running Nyström benchmark with {size} samples: {e}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Plot results
    fig = plot_benchmark_results(all_results, title='Comparison of Kernel Approximation Methods')
    fig.savefig('results/benchmark_comparison.png', dpi=300)
    
    # Collect all classifiers that were evaluated
    all_classifiers = {}
    all_classifiers.update(rff_classifiers)
    
    # For Nyström, use the classifiers from the largest sample size that worked
    largest_nys_size = 0
    for size in sorted(sample_sizes, reverse=True):
        nystrom_classifiers = {}
        for name, config in classifier_configs['nystrom'].items():
            if config['sample_size'] == size and name in all_results['accuracy']:
                # Check if this classifier has valid results for this size
                idx = all_results['sample_sizes'].index(size)
                if idx < len(all_results['accuracy'][name]) and not np.isnan(all_results['accuracy'][name][idx]):
                    nystrom_classifiers[name] = KernelApproximationClassifier(
                        approximation=config['approx'],
                        approx_params=config['params'],
                        optimizer_params=config['optimizer']
                    )
        
        if nystrom_classifiers:
            all_classifiers.update(nystrom_classifiers)
            largest_nys_size = size
            break
    
    # Detailed evaluation of models on the largest feasible dataset
    print("\nDetailed evaluation of models on the full dataset:")
    
    # Find the largest sample size that can be used for evaluation
    eval_size = min(total_samples, largest_nys_size if largest_nys_size > 0 else total_samples)
    print(f"Using {eval_size} samples for final evaluation")
    
    for name, clf in all_classifiers.items():
        print(f"\n{name}:")
        try:
            # Fit on evaluation dataset
            start_time = time.time()
            clf.fit(X_train_scaled[:eval_size], y_train[:eval_size])
            train_time = time.time() - start_time
            print(f"Training time: {train_time:.4f} s")
            
            # Evaluate on test set
            y_pred = clf.predict(X_test_scaled)
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    print("\nBenchmark completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    # Add command-line argument for using optimized parameters
    use_optimized = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'no-optimize':
        use_optimized = False
    
    run_comparative_analysis(use_optimized_params=use_optimized)