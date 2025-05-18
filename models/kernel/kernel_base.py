# kernel_base.py
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for kernel-based classification methods.
    This is a simple implementation that uses exact kernel computation.
    """
    
    def __init__(self, kernel='rbf', gamma=1.0, C=1.0):
        """
        Initialize the kernel classifier.
        
        Args:
            kernel (str or callable): Kernel function to use. If string, must be one of
                'rbf', 'linear', or 'polynomial'. If callable, must take two arrays and
                return a kernel matrix.
            gamma (float): Parameter for RBF kernel.
            C (float): Regularization parameter.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.X_train = None
        self.y_train = None
        self.alpha = None
        
        # For performance monitoring
        self.train_time = None
        self.predict_time = None
        self.memory_usage = None
    
    def _get_kernel(self, X, Y=None):
        """
        Compute the kernel matrix between X and Y.
        
        Args:
            X (array-like): First set of samples.
            Y (array-like, optional): Second set of samples. If None, X is used.
            
        Returns:
            K (array): Kernel matrix.
        """
        if Y is None:
            Y = X
            
        if callable(self.kernel):
            return self.kernel(X, Y)
        
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        
        elif self.kernel == 'rbf':
            # Compute squared Euclidean distances using NumPy broadcasting
            # This approach is more efficient and avoids torch dependency issues
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            Y_norm = np.sum(Y**2, axis=1)
            distances_squared = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            
            # Apply RBF kernel formula: K(x,y) = exp(-gamma * ||x-y||^2)
            return np.exp(-self.gamma * distances_squared)
        
        elif self.kernel == 'polynomial':
            # K(x, y) = (gamma * x^T y + 1)^3
            return (self.gamma * np.dot(X, Y.T) + 1) ** 3
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        
        Args:
            X (array-like): Training vectors.
            y (array-like): Target values.
            
        Returns:
            self
        """
        start_time = time.time()
        
        # Store training data
        self.X_train = X
        self.y_train = np.unique(y, return_inverse=True)[1]  # Convert to integer indices
        self.classes_ = np.unique(y)
        
        # Compute kernel matrix
        K = self._get_kernel(X)
        
        # Add small regularization term to the diagonal to improve numerical stability
        K = K + np.eye(K.shape[0]) * self.C
        
        # For binary classification, we can use a simplified approach
        if len(self.classes_) == 2:
            # Convert to {-1, 1} labels
            y_binary = 2 * self.y_train - 1
            
            # Solve dual problem (simplified)
            self.alpha = np.linalg.solve(K, y_binary)
        else:
            # For multiclass, we use one-vs-rest approach
            n_classes = len(self.classes_)
            n_samples = X.shape[0]
            self.alpha = np.zeros((n_classes, n_samples))
            
            for i in range(n_classes):
                # Create binary labels (1 for current class, -1 for other classes)
                y_binary = 2 * (self.y_train == i).astype(int) - 1
                
                # Solve dual problem for each class
                self.alpha[i] = np.linalg.solve(K, y_binary)
        
        self.train_time = time.time() - start_time
        
        # Estimate memory usage (very rough approximation)
        self.memory_usage = (
            X.nbytes +                  # Training samples
            y.nbytes +                  # Labels
            K.nbytes +                  # Kernel matrix
            self.alpha.nbytes           # Model parameters
        )
        
        logger.info(f"Training completed in {self.train_time:.2f} seconds.")
        logger.info(f"Estimated memory usage: {self.memory_usage / (1024**2):.2f} MB")
        
        return self
    
    def decision_function(self, X):
        """
        Compute the decision function for samples in X.
        
        Args:
            X (array-like): Test vectors.
            
        Returns:
            array: Decision function values.
        """
        # Compute kernel matrix between test and training samples
        K = self._get_kernel(X, self.X_train)
        
        if len(self.classes_) == 2:
            return K @ self.alpha
        else:
            # For multiclass, return the decision function for each class
            return K @ self.alpha.T
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X (array-like): Test vectors.
            
        Returns:
            array: Predicted class labels.
        """
        start_time = time.time()
        
        # Compute decision function
        decision = self.decision_function(X)
        
        # Get predictions
        if len(self.classes_) == 2:
            y_pred = (decision > 0).astype(int)
        else:
            y_pred = np.argmax(decision, axis=1)
        
        self.predict_time = time.time() - start_time
        
        # Map back to original class labels
        return self.classes_[y_pred]
    
    def score(self, X, y):
        """
        Returns the accuracy score on the given test data and labels.
        
        Args:
            X (array-like): Test vectors.
            y (array-like): True labels.
            
        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_performance_metrics(self):
        """
        Return performance metrics for the model.
        
        Returns:
            dict: Dictionary containing performance metrics.
        """
        return {
            'train_time': self.train_time,
            'predict_time': self.predict_time,
            'memory_usage_mb': self.memory_usage / (1024**2) if self.memory_usage else None
        }