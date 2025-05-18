# Evaluation Instructions

This document explains how to run the comprehensive evaluations for comparing kernel approximation methods with deep learning baselines on the Sign Language MNIST dataset.

## Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Evaluations

### 1. Comprehensive Kernel Methods Evaluation

To run the comprehensive evaluation on the full Sign Language MNIST dataset comparing RFF and Nyström methods:

```bash
python run_evaluations.py
```

This script will:

- Run the comparative analysis using optimized parameters (if available)
- Generate confusion matrices for each method
- Create performance trade-off visualizations
- Generate a summary CSV file with results

### 2. Comparison with CNN Baseline

To compare kernel methods against the CNN baseline:

```bash
python compare_with_cnn.py
```

This script will:

- Evaluate Random Fourier Features (RFF) with optimized parameters
- Evaluate Nyström method with optimized parameters
- Train and evaluate a CNN baseline model
- Compare performance in terms of:
  - Accuracy
  - Training time
  - Prediction time
  - Memory usage
- Generate visualizations of all comparisons

## Output Files

All output files will be saved to the `results` directory, including:

- `benchmark_results.pkl`: Pickle file with all benchmark results
- `method_comparison_summary.csv`: CSV file summarizing method comparisons
- `kernel_vs_cnn_comparison.csv`: Comparison between kernel methods and CNN
- Various PNG files with visualizations:
  - Confusion matrices for each method
  - Accuracy comparisons
  - Training/prediction time comparisons
  - Memory usage comparisons
  - Accuracy vs. time trade-off plots

## Notes

- The scripts automatically use optimized parameters if they exist (from hyperparameter tuning)
- For large datasets, the evaluation uses a maximum of 10,000 samples to ensure reasonable runtime
- All evaluations include memory usage tracking for proper resource comparison
- The CNN baseline uses early stopping to prevent overfitting
