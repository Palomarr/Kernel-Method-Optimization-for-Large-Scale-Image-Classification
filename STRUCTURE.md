# Project Structure

This document outlines the organization of the codebase for the Kernel Method Optimization project.

## Directory Structure

```
Kernel-Method-Optimization-for-Large-Scale-Image-Classification/
├── run.py                    # Main runner script for the project
├── requirements.txt          # Project dependencies
├── EVALUATION.md             # Instructions for running evaluations
├── README.md                 # Project overview
├── STRUCTURE.md              # This file
│
├── models/                   # All model implementations
│   ├── kernel/               # Kernel methods implementations
│   │   ├── approximations/   # Kernel approximation techniques
│   │   │   ├── nystrom.py    # Nyström method implementation
│   │   │   └── random_fourier_features.py # RFF implementation
│   │   ├── optimization/     # Optimization algorithms for kernel methods
│   │   │   └── stochastic_optimizer.py # SGD implementation
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── kernel_approximation_classifier.py # Main classifier
│   │   └── kernel_base.py    # Base class for kernel methods
│   │
│   └── neural/               # Deep learning implementations
│       └── cnn_baseline.py   # CNN baseline for comparison
│
├── evaluation/               # Evaluation and benchmarking code
│   ├── benchmark.py          # Benchmarking utilities
│   └── experiments.py        # Experiment runner
│
├── data_utils/               # Data processing utilities
│   └── data_loader.py        # Loading and preprocessing data
│
├── scripts/                  # Execution scripts
│   ├── run_optimization.py   # Script to run hyperparameter optimization
│   ├── run_experiments.py    # Script to run experiments
│   ├── run_evaluations.py    # Script for comprehensive evaluation
│   └── compare_with_cnn.py   # Script to compare with CNN baseline
│
└── results/                  # Directory for storing results (created at runtime)
```

## Running the Code

The main entry point for running all functionality is `run.py`, which provides a unified interface:

```bash
# Run hyperparameter optimization
python run.py optimize

# Run benchmarks
python run.py benchmark

# Run full evaluation
python run.py evaluate

# Compare with CNN baseline
python run.py compare-cnn

# Run all steps
python run.py all
```

Use the `--no-optimize` flag to skip optimization when running benchmarks or evaluations:

```bash
python run.py benchmark --no-optimize
```

## Code Organization

- **Models**: Contains all model implementations, separated into kernel methods and neural networks
- **Evaluation**: Contains code for benchmarking and running experiments
- **Data Utils**: Contains utilities for loading and processing data
- **Scripts**: Contains execution scripts for different tasks
- **Results**: Directory where all output files are stored

This structure is designed to keep the code organized and modular, making it easier to understand and maintain.
