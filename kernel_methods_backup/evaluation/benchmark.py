import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def run_benchmark(classifiers, X_train, y_train, X_test, y_test, 
                  sample_sizes=None, n_runs=1, indices_dict=None):
    """
    Run benchmark comparison of multiple classifiers.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary mapping classifier names to classifier objects.
    X_train : array-like of shape (n_samples, n_features)
        Training data.
    y_train : array-like of shape (n_samples,)
        Training labels.
    X_test : array-like of shape (n_test_samples, n_features)
        Test data.
    y_test : array-like of shape (n_test_samples,)
        Test labels.
    sample_sizes : list, default=None
        List of training set sizes to evaluate. If None, only the full dataset is used.
    n_runs : int, default=1
        Number of runs to average results over.
    indices_dict : dict, default=None
        Dictionary mapping sample sizes to arrays of indices to use for training.
        If provided, these exact indices will be used instead of random sampling.
        
    Returns:
    --------
    results : dict
        Dictionary containing benchmark results.
    """
    if sample_sizes is None:
        sample_sizes = [X_train.shape[0]]
    
    # Make sure sample sizes don't exceed the available data
    available_samples = X_train.shape[0]
    valid_sample_sizes = []
    for size in sample_sizes:
        if size <= available_samples:
            valid_sample_sizes.append(size)
        else:
            print(f"Warning: Requested sample size {size} exceeds available data size {available_samples}.")
            print(f"Using full dataset size {available_samples} instead.")
            valid_sample_sizes.append(available_samples)
    
    sample_sizes = valid_sample_sizes
    
    results = {
        'accuracy': {name: [] for name in classifiers.keys()},
        'training_time': {name: [] for name in classifiers.keys()},
        'prediction_time': {name: [] for name in classifiers.keys()},
        'sample_sizes': sample_sizes,
        'confusion_matrices': {name: {} for name in classifiers.keys()}
    }
    
    for size in sample_sizes:
        print(f"\nEvaluating with {size} training samples:")
        
        # For each classifier
        for name, clf in classifiers.items():
            print(f"\n{name}:")
            
            accuracies = []
            train_times = []
            pred_times = []
            conf_matrices = []
            
            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}")
                
                # Sample training data
                if indices_dict is not None and size in indices_dict:
                    # Use the pre-defined indices for this size
                    indices = indices_dict[size]
                    X_train_sample = X_train[indices]
                    y_train_sample = y_train[indices]
                else:
                    # Sample training data - only sample if size is less than total available
                    if size < X_train.shape[0]:
                        indices = np.random.choice(X_train.shape[0], size=size, replace=False)
                        X_train_sample = X_train[indices]
                        y_train_sample = y_train[indices]
                    else:
                        # Use the full dataset
                        X_train_sample = X_train
                        y_train_sample = y_train
                
                # Train classifier
                start_time = time.time()
                clf.fit(X_train_sample, y_train_sample)
                train_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                y_pred = clf.predict(X_test)
                pred_time = time.time() - start_time
                
                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Store results
                accuracies.append(accuracy)
                train_times.append(train_time)
                pred_times.append(pred_time)
                conf_matrices.append(conf_matrix)
                
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Training time: {train_time:.4f} s")
                print(f"    Prediction time: {pred_time:.4f} s")
            
            # Average results over runs
            results['accuracy'][name].append(np.mean(accuracies))
            results['training_time'][name].append(np.mean(train_times))
            results['prediction_time'][name].append(np.mean(pred_times))
            results['confusion_matrices'][name][size] = np.mean(conf_matrices, axis=0)
    
    return results

def plot_benchmark_results(results, title=None, figsize=(15, 10)):
    """
    Plot benchmark results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing benchmark results from run_benchmark.
    title : str, default=None
        Title for the figure.
    figsize : tuple, default=(15, 10)
        Figure size.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    classifiers = list(results['accuracy'].keys())
    sample_sizes = results['sample_sizes']
    
    # Plot accuracy
    for name in classifiers:
        # Skip if all values are NaN
        if all(np.isnan(acc) for acc in results['accuracy'][name]):
            continue
            
        # Get non-NaN values and their corresponding sample sizes
        valid_indices = [i for i, acc in enumerate(results['accuracy'][name]) if not np.isnan(acc)]
        valid_sizes = [sample_sizes[i] for i in valid_indices]
        valid_accuracies = [results['accuracy'][name][i] for i in valid_indices]
        
        # Only plot if we have valid data
        if valid_accuracies:
            axes[0].plot(valid_sizes, valid_accuracies, marker='o', label=name)
    
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Training Set Size')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot training time
    for name in classifiers:
        # Skip if all values are NaN
        if all(np.isnan(time) for time in results['training_time'][name]):
            continue
            
        # Get non-NaN values and their corresponding sample sizes
        valid_indices = [i for i, time in enumerate(results['training_time'][name]) if not np.isnan(time)]
        valid_sizes = [sample_sizes[i] for i in valid_indices]
        valid_times = [results['training_time'][name][i] for i in valid_indices]
        
        # Only plot if we have valid data
        if valid_times:
            axes[1].plot(valid_sizes, valid_times, marker='o', label=name)
    
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('Training Time vs Training Set Size')
    axes[1].grid(True)
    
    # Plot prediction time
    for name in classifiers:
        # Skip if all values are NaN
        if all(np.isnan(time) for time in results['prediction_time'][name]):
            continue
            
        # Get non-NaN values and their corresponding sample sizes
        valid_indices = [i for i, time in enumerate(results['prediction_time'][name]) if not np.isnan(time)]
        valid_sizes = [sample_sizes[i] for i in valid_indices]
        valid_times = [results['prediction_time'][name][i] for i in valid_indices]
        
        # Only plot if we have valid data
        if valid_times:
            axes[2].plot(valid_sizes, valid_times, marker='o', label=name)
    
    axes[2].set_xlabel('Training Set Size')
    axes[2].set_ylabel('Prediction Time (s)')
    axes[2].set_title('Prediction Time vs Training Set Size')
    axes[2].grid(True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, classes, title=None, cmap='Reds', figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix.
    classes : list
        List of class labels.
    title : str, default=None
        Title for the figure.
    cmap : matplotlib colormap, default=plt.cm.Blues
        Colormap for the plot.
    figsize : tuple, default=(8, 6)
        Figure size.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                xticklabels=classes, yticklabels=classes)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    
    return fig


if __name__ == "__main__":
    import sys
    import os
    import time
    # Add the project root to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from kernel_methods.utils.data_loader import load_reshaped_data
    from kernel_methods.models.kernel_approximation_classifier import KernelApproximationClassifier
    from sklearn.svm import SVC
    
    # Create some sample data for testing if needed
    X_train, y_train, X_val, y_val, X_test, y_test = load_reshaped_data()
    
    # For simplicity in testing, just use a smaller subset
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Flatten data if needed
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define a maximum sample size for the benchmark that won't be too slow
    max_sample_size = min(10000, X_train.shape[0])  # Using up to 10k samples
    
    # Improved kernel approximation classifiers with better parameters
    classifiers = {
        "Exact RBF Kernel (γ=0.001)": SVC(
            kernel='rbf', 
            gamma=0.001,
            C=10.0,
            cache_size=2000  # Use 2GB of cache for kernel computations
        ),
        "Exact RBF Kernel (γ=0.01)": SVC(
            kernel='rbf', 
            gamma=0.01,
            C=10.0,
            cache_size=2000
        ),
        "RFF (1000 components, γ=0.001)": KernelApproximationClassifier(
            approximation='rff',
            approx_params={'n_components': 1000, 'gamma': 0.001},
            optimizer_params={'learning_rate': 0.05, 'batch_size': 128, 'max_iter': 300}
        ),
        "RFF (1000 components, γ=0.01)": KernelApproximationClassifier(
            approximation='rff',
            approx_params={'n_components': 1000, 'gamma': 0.01},
            optimizer_params={'learning_rate': 0.05, 'batch_size': 128, 'max_iter': 300}
        ),
        "Nyström (1000 components, γ=0.001)": KernelApproximationClassifier(
            approximation='nystrom',
            approx_params={'n_components': 1000, 'gamma': 0.001, 'sampling': 'kmeans'},
            optimizer_params={'learning_rate': 0.05, 'batch_size': 128, 'max_iter': 300}
        ),
        "Nyström (1000 components, γ=0.01)": KernelApproximationClassifier(
            approximation='nystrom',
            approx_params={'n_components': 1000, 'gamma': 0.01, 'sampling': 'kmeans'},
            optimizer_params={'learning_rate': 0.05, 'batch_size': 128, 'max_iter': 300}
        )
    }
    
    # Use more training data for better results
    indices = np.random.choice(X_train.shape[0], max_sample_size, replace=False)
    
    print(f"Running benchmark with exact kernels and approximations on {max_sample_size} total samples")
    print("Note: Exact kernel methods may be significantly slower on larger sample sizes")
    
    # For exact kernel methods, we'll need to use smaller sample sizes 
    # since they scale poorly with the number of samples
    exact_sample_sizes = [500, 1000, 2000]
    approx_sample_sizes = [2000, 5000, max_sample_size]
    
    # Create a fixed set of training data indices for all tests
    # This ensures we use the same subsets of data for small and large sample sizes
    # First, shuffle the indices
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    shuffled_indices = rng.permutation(indices)
    
    # Use these indices to create nested subsets for different sample sizes
    training_indices = {}
    for size in sorted(set(exact_sample_sizes + approx_sample_sizes)):
        # Use the first 'size' samples from the shuffled indices
        training_indices[size] = shuffled_indices[:size]
    
    # Define a function to estimate memory requirements for different methods
    def estimate_memory_requirement(method_name, n_samples, n_features):
        """Estimate memory requirements in GB for a method with given parameters"""
        # Base memory for data storage (X, y)
        base_memory_gb = (n_samples * n_features * 8) / (1024**3)  # 8 bytes per float64
        
        if "Exact" in method_name:
            # Exact kernel methods store the full kernel matrix (n_samples × n_samples)
            kernel_matrix_gb = (n_samples * n_samples * 8) / (1024**3)
            return base_memory_gb + kernel_matrix_gb
        elif "RFF" in method_name:
            # Extract component count
            components = int(method_name.split("components")[0].strip().split("(")[1].strip())
            # RFF stores random weights (n_features × n_components) and transformed data (n_samples × n_components)
            rff_memory_gb = ((n_features * components + n_samples * components) * 8) / (1024**3)
            return base_memory_gb + rff_memory_gb
        elif "Nyström" in method_name:
            # Extract component count
            components = int(method_name.split("components")[0].strip().split("(")[1].strip())
            # Nyström stores component data, kernel matrices (n_components × n_components and n_samples × n_components)
            # K-means clustering also uses significant memory
            nystrom_memory_gb = ((components * n_features + components * components + n_samples * components) * 8) / (1024**3)
            # K-means uses additional memory for centroids and distance calculations
            kmeans_memory_gb = ((components * n_features + n_samples * components) * 8) / (1024**3)
            return base_memory_gb + nystrom_memory_gb + kmeans_memory_gb
        else:
            return base_memory_gb
    
    # Define memory threshold in GB (adjust based on available system memory)
    MEMORY_THRESHOLD_GB = 8.0  # Skip tests that would use more than 8GB
    
    # Run a custom benchmark with different sample sizes for exact vs. approximate methods
    all_results = {}
    all_methods = set()
    
    # Start with the exact kernel methods (which are slow for large samples)
    exact_classifiers = {k: v for k, v in classifiers.items() if "Exact" in k}
    if exact_classifiers:
        print("\nBenchmarking exact kernel methods (smaller sample sizes due to computational cost):")
        
        # Check memory requirements and filter out methods that would exceed the threshold
        safe_exact_classifiers = {}
        for name, clf in exact_classifiers.items():
            for size in exact_sample_sizes:
                mem_req = estimate_memory_requirement(name, size, X_train.shape[1])
                if mem_req > MEMORY_THRESHOLD_GB:
                    print(f"  Skipping {name} with {size} samples - estimated memory requirement: {mem_req:.2f} GB")
                else:
                    if name not in safe_exact_classifiers:
                        safe_exact_classifiers[name] = clf
                        all_methods.add(name)
        
        if safe_exact_classifiers:
            exact_results = run_benchmark(
                safe_exact_classifiers,
                X_train, y_train, 
                X_test, y_test,
                sample_sizes=exact_sample_sizes,
                n_runs=1,
                indices_dict=training_indices  # Pass the pre-generated indices
            )
            # Store these results
            all_results.update({
                'accuracy': {k: exact_results['accuracy'][k] for k in safe_exact_classifiers.keys()},
                'training_time': {k: exact_results['training_time'][k] for k in safe_exact_classifiers.keys()},
                'prediction_time': {k: exact_results['prediction_time'][k] for k in safe_exact_classifiers.keys()},
                'confusion_matrices': {k: exact_results['confusion_matrices'][k] for k in safe_exact_classifiers.keys()}
            })
    
    # Then run the approximation methods for each sample size
    for size in approx_sample_sizes:
        approx_classifiers = {k: v for k, v in classifiers.items() if "Exact" not in k}
        
        # Check memory requirements and filter out methods that would exceed the threshold
        safe_approx_classifiers = {}
        for name, clf in approx_classifiers.items():
            # Explicitly skip Nyström with 1000 components for large sample sizes
            if "Nyström" in name and "1000 components" in name and size >= 5000:
                print(f"\nSkipping {name} with {size} samples - known to exceed memory limits")
                continue
                
            mem_req = estimate_memory_requirement(name, size, X_train.shape[1])
            if mem_req > MEMORY_THRESHOLD_GB:
                print(f"\nSkipping {name} with {size} samples - estimated memory requirement: {mem_req:.2f} GB")
            else:
                safe_approx_classifiers[name] = clf
                all_methods.add(name)
        
        if safe_approx_classifiers:
            print(f"\nBenchmarking approximation methods with {size} samples:")
            approx_results = run_benchmark(
                safe_approx_classifiers,
                X_train, y_train, 
                X_test, y_test,
                sample_sizes=[size],
                n_runs=1,
                indices_dict=training_indices  # Pass the pre-generated indices
            )
            
            # Merge the results
            if not all_results:
                all_results = approx_results
            else:
                for name in approx_results['accuracy']:
                    if name not in all_results['accuracy']:
                        all_results['accuracy'][name] = []
                        all_results['training_time'][name] = []
                        all_results['prediction_time'][name] = []
                        all_results['confusion_matrices'][name] = {}
                    
                    all_results['accuracy'][name].extend(approx_results['accuracy'][name])
                    all_results['training_time'][name].extend(approx_results['training_time'][name])
                    all_results['prediction_time'][name].extend(approx_results['prediction_time'][name])
                    all_results['confusion_matrices'][name].update(approx_results['confusion_matrices'][name])
    
    # Combine sample sizes for the plots, with duplicates removed
    all_sample_sizes = sorted(list(set(exact_sample_sizes + approx_sample_sizes)))
    all_results['sample_sizes'] = all_sample_sizes
    
    # For plotting, we need to ensure all methods have values for all sample sizes
    # For missing data points, fill with NaN
    for method in all_methods:
        # Initialize if not already present
        if method not in all_results['accuracy']:
            all_results['accuracy'][method] = []
            all_results['training_time'][method] = []
            all_results['prediction_time'][method] = []
            all_results['confusion_matrices'][method] = {}
            
        # Find which sample sizes this method was evaluated on
        evaluated_sizes = []
        for size in all_sample_sizes:
            # Skip Nyström with 1000 components for large sample sizes (regardless of memory estimate)
            if "Nyström" in method and "1000 components" in method and size >= 5000:
                continue
                
            mem_req = estimate_memory_requirement(method, size, X_train.shape[1])
            if mem_req <= MEMORY_THRESHOLD_GB:
                # This method should have been evaluated at this size
                if "Exact" in method and size in exact_sample_sizes:
                    evaluated_sizes.append(size)
                elif "Exact" not in method and size in approx_sample_sizes:
                    evaluated_sizes.append(size)
        
        # Create new lists with NaN for missing sizes
        new_accuracy = []
        new_train_time = []
        new_pred_time = []
        
        for size in all_sample_sizes:
            if size in evaluated_sizes and method in all_results['accuracy']:
                # Find the index in the existing results
                existing_sizes = [s for s in all_results['sample_sizes'] 
                                  if s <= size and 
                                  (s in exact_sample_sizes if "Exact" in method else s in approx_sample_sizes)]
                
                if existing_sizes:
                    # Find the closest size that was actually evaluated
                    closest_size = max(existing_sizes)
                    idx = all_results['sample_sizes'].index(closest_size)
                    
                    if idx < len(all_results['accuracy'][method]):
                        new_accuracy.append(all_results['accuracy'][method][idx])
                        new_train_time.append(all_results['training_time'][method][idx])
                        new_pred_time.append(all_results['prediction_time'][method][idx])
                    else:
                        new_accuracy.append(float('nan'))
                        new_train_time.append(float('nan'))
                        new_pred_time.append(float('nan'))
                else:
                    new_accuracy.append(float('nan'))
                    new_train_time.append(float('nan'))
                    new_pred_time.append(float('nan'))
            else:
                new_accuracy.append(float('nan'))
                new_train_time.append(float('nan'))
                new_pred_time.append(float('nan'))
        
        # Update the results
        all_results['accuracy'][method] = new_accuracy
        all_results['training_time'][method] = new_train_time
        all_results['prediction_time'][method] = new_pred_time
    
    # Plot the results
    fig = plot_benchmark_results(all_results, title="Exact vs. Approximate Kernel Methods Comparison")
    fig.savefig("benchmark_exact_vs_approx.png", dpi=300, bbox_inches='tight')
    print(f"Saved comparison benchmark results to benchmark_exact_vs_approx.png")
    
    # Print a summary of the best configurations
    print("\nBest Performance Summary (by method):")
    print("-----------------------------------")
    
    # Organize methods by type
    method_types = {
        "Exact Kernel": [],
        "RFF": [],
        "Nyström": []
    }
    
    # Group methods by their type
    for method in all_methods:
        if "Exact" in method:
            method_types["Exact Kernel"].append(method)
        elif "RFF" in method:
            method_types["RFF"].append(method)
        elif "Nyström" in method:
            method_types["Nyström"].append(method)
    
    for method_type, method_keys in method_types.items():
        print(f"\n{method_type} Methods:")
        best_method_info = []
        
        for name in method_keys:
            # Get non-NaN results for this method
            valid_results = []
            for i, (acc, size) in enumerate(zip(all_results['accuracy'][name], all_results['sample_sizes'])):
                if not np.isnan(acc):
                    train_time = all_results['training_time'][name][i]
                    if not np.isnan(train_time):
                        valid_results.append((acc, size, train_time))
            
            if valid_results:
                # Find best accuracy
                best_result = max(valid_results, key=lambda x: x[0])
                best_acc, best_size, best_time = best_result
                
                # Format method name for display
                method_str = f"{name} (sample size: {best_size})"
                best_method_info.append((method_str, best_acc, best_time))
        
        # Sort by accuracy
        for method_str, acc, time in sorted(best_method_info, key=lambda x: x[1], reverse=True):
            print(f"  {method_str}: {acc:.4f} accuracy (training time: {time:.2f}s)")
            
    print("\nNote: Some configurations were skipped due to estimated memory requirements exceeding the threshold.")
    print(f"Memory threshold: {MEMORY_THRESHOLD_GB:.1f} GB")
    print("\nNyström with 1000 components was explicitly excluded for sample sizes ≥ 5000 due to known memory issues.")
    