# nystrom.py
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

class NystromApproximation:
    """
    Implements the Nystrom method for approximating kernel matrices.
    
    The Nyström method approximates a kernel matrix by sampling a subset of
    data points and using those to construct a low-rank approximation.
    """
    
    def __init__(self, n_components=100, kernel='rbf', gamma=None, sampling='kmeans', random_state=None):
        """
        Initialize the Nyström approximation.
        
        Parameters:
        -----------
        n_components : int, default=100
            Number of samples to use for the approximation.
        kernel : str or callable, default='rbf'
            Kernel function to use. If a string, must be one of 'rbf'.
            If a callable, it should take two arrays and return a kernel matrix.
        gamma : float, default=None
            Parameter for the RBF kernel. If None, defaults to 1/n_features.
        sampling : str, default='kmeans'
            Method to use for sampling the landmarks.
            Options: 'uniform', 'kmeans'
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.sampling = sampling
        self.random_state = random_state
        self.components_ = None
        self.component_indices_ = None
        self.normalization_ = None
        
    def _get_kernel_function(self):
        """Returns the kernel function based on the kernel parameter."""
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == 'rbf':
            return lambda X, Y: rbf_kernel(X, Y, gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _uniform_sampling(self, X, n_samples):
        """Uniformly sample indices from X."""
        rng = np.random.default_rng(self.random_state)
        n_samples = min(n_samples, X.shape[0])
        indices = rng.choice(X.shape[0], size=n_samples, replace=False)
        return indices
    
    def _kmeans_sampling(self, X, n_samples):
        """Sample indices using K-means clustering."""
        kmeans = KMeans(n_clusters=n_samples, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        
        # Find the nearest point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :], axis=2)
        indices = np.argmin(distances, axis=0)
        
        return indices
    
    def fit(self, X, sampling=None):
        """
        Fit the Nystrom approximation on the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        sampling : str, default=None
            Method to use for sampling the landmarks. If None, uses the sampling method specified in the constructor.
            Options: 'uniform', 'kmeans'
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Use the sampling method from the constructor if not specified
        if sampling is None:
            sampling = self.sampling
            
        # Sample points to use as components
        if sampling == 'uniform':
            self.component_indices_ = self._uniform_sampling(X, self.n_components)
        elif sampling == 'kmeans':
            self.component_indices_ = self._kmeans_sampling(X, self.n_components)
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        
        self.components_ = X[self.component_indices_]
        
        # Compute kernel between components
        kernel_fn = self._get_kernel_function()
        K_mm = kernel_fn(self.components_, self.components_)
        
        # Add small regularization to diagonal for numerical stability
        K_mm = K_mm + np.eye(K_mm.shape[0]) * 1e-10
        
        # Compute SVD of kernel matrix
        U, S, _ = np.linalg.svd(K_mm)
        
        # Improved handling of small eigenvalues
        # Set a minimum threshold for eigenvalues to avoid numerical issues
        min_eigenvalue = 1e-10
        S = np.maximum(S, min_eigenvalue)
        
        # Compute inverse square root of eigenvalues
        S_inv = 1.0 / np.sqrt(S)
        
        # Store normalization matrix for transform
        self.normalization_ = U @ np.diag(S_inv)
        
        return self
    
    def transform(self, X):
        """
        Apply feature map to X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns:
        --------
        X_transformed : array of shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError("The model has not been fitted yet.")
        
        kernel_fn = self._get_kernel_function()
        K_nm = kernel_fn(X, self.components_)
        
        # Apply normalization to get the features
        X_transformed = K_nm @ self.normalization_
        
        return X_transformed