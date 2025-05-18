# random_fourier_features.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """
    Implementation of Random Fourier Features for kernel approximation.
    
    This class implements the Random Fourier Features method proposed by Rahimi and Recht.
    It approximates the RBF kernel by explicitly mapping the data to a randomized low-dimensional
    feature space where the inner product approximates the kernel.
    
    Reference:
    Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines.
    Advances in neural information processing systems, 20.
    """
    
    def __init__(self, n_components=100, gamma=1.0, random_state=None):
        """
        Initialize the transformer.
        
        Args:
            n_components (int): Number of Monte Carlo samples per original feature.
                Equals the dimensionality of the computed feature space.
            gamma (float): Parameter of the RBF kernel to be approximated.
            random_state (int or RandomState): Controls the random sampling.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        
        # Initialize these in the fit method
        self.random_weights_ = None
        self.random_offset_ = None
        
        # For performance monitoring
        self.transform_time = None
        self.memory_usage = None
    
    def fit(self, X, y=None):
        """
        Generate the random weights for the Fourier features.
        
        Args:
            X (array-like): Training data, shape (n_samples, n_features)
            y (array-like): Target values (ignored)
            
        Returns:
            self
        """
        X = check_array(X)
        n_features = X.shape[1]
        
        # Set random state
        rng = np.random.default_rng(self.random_state)
        
        # Generate random weights and offsets
        # For RBF kernel, weights are sampled from a normal distribution
        self.random_weights_ = rng.normal(
            scale=np.sqrt(2 * self.gamma),
            size=(n_features, self.n_components)
        )
        
        # Random offsets are sampled from uniform distribution [0, 2*pi]
        self.random_offset_ = rng.uniform(
            0, 2 * np.pi, size=self.n_components
        )
        
        # Estimate memory usage
        self.memory_usage = (
            self.random_weights_.nbytes +  # Random weights
            self.random_offset_.nbytes      # Random offsets
        )
        
        logger.info(f"Fitted Random Fourier Features with {self.n_components} components")
        logger.info(f"Estimated memory usage: {self.memory_usage / (1024**2):.2f} MB")
        
        return self
    
    def transform(self, X):
        """
        Apply the approximated kernel map to X.
        
        Args:
            X (array-like): New data, shape (n_samples, n_features)
            
        Returns:
            X_new (array): Transformed data
        """
        start_time = time.time()
        
        X = check_array(X)
        
        # Check if fit has been called
        if self.random_weights_ is None:
            raise ValueError("RandomFourierFeatures has not been fitted. Call 'fit' first.")
        
        # Project the data: X_proj = X * random_weights
        X_proj = np.dot(X, self.random_weights_)
        
        # Apply the random offset and cosine transformation
        # The cos transformation ensures that <z(x), z(y)> approximates K(x, y)
        X_proj += self.random_offset_
        X_proj = np.cos(X_proj)
        
        # Apply normalization factor
        X_new = X_proj * np.sqrt(2. / self.n_components)
        
        self.transform_time = time.time() - start_time
        
        return X_new
    
    def fit_transform(self, X, y=None):
        """
        Fit and apply the approximated kernel map to X.
        
        Args:
            X (array-like): Training data, shape (n_samples, n_features)
            y (array-like): Target values (ignored)
            
        Returns:
            X_new (array): Transformed data
        """
        return self.fit(X).transform(X)
    
    def get_performance_metrics(self):
        """
        Return performance metrics for the transformation.
        
        Returns:
            dict: Dictionary containing performance metrics.
        """
        return {
            'transform_time': self.transform_time,
            'memory_usage_mb': self.memory_usage / (1024**2) if self.memory_usage else None,
            'n_components': self.n_components
        }