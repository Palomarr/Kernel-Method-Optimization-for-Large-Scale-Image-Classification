import numpy as np
import os, sys
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approximations.random_fourier_features import RandomFourierFeatures
from approximations.nystrom import NystromApproximation
from optimization.stochastic_optimizer import SGDOptimizer

class KernelApproximationClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier using kernel approximation techniques.
    
    This classifier combines kernel approximation methods with
    stochastic optimization for scalable kernel-based classification.
    """
    
    def __init__(self, approximation='rff', approx_params=None, 
                 optimizer_params=None, random_state=None):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        approximation : str, default='rff'
            Kernel approximation method. Options: 'rff', 'nystrom'.
        approx_params : dict, default=None
            Parameters for the kernel approximation method.
        optimizer_params : dict, default=None
            Parameters for the optimizer.
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.approximation = approximation
        self.approx_params = approx_params or {}
        self.optimizer_params = optimizer_params or {}
        self.random_state = random_state
        
    def _create_approximator(self):
        """Create the kernel approximation object."""
        if self.approximation == 'rff':
            return RandomFourierFeatures(
                random_state=self.random_state,
                **self.approx_params
            )
        elif self.approximation == 'nystrom':
            # Update default params to include kmeans sampling
            nystrom_params = dict(self.approx_params)
            if 'sampling' not in nystrom_params:
                nystrom_params['sampling'] = 'kmeans'  # Use k-means sampling instead of uniform
            return NystromApproximation(
                random_state=self.random_state,
                **nystrom_params
            )
        else:
            raise ValueError(f"Unknown approximation method: {self.approximation}")
    
    def _create_optimizer(self):
        """Create the optimizer object."""
        return SGDOptimizer(
            random_state=self.random_state,
            **self.optimizer_params
        )
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the classifier to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        X_val : array-like of shape (n_val_samples, n_features), default=None
            Validation data for early stopping.
        y_val : array-like of shape (n_val_samples,), default=None
            Validation target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Check data
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Remap labels to [0, n_classes-1]
        self.label_binarizer_ = LabelBinarizer()
        self.label_binarizer_.fit(y)
        y_bin = np.argmax(self.label_binarizer_.transform(y), axis=1)
        
        # Create and fit kernel approximator
        self.approximator_ = self._create_approximator()
        print(f"Fitting {self.approximation} approximation...")
        self.approximator_.fit(X)
        
        # Transform data using the kernel approximation
        X_transformed = self.approximator_.transform(X)
        if X_val is not None:
            X_val_transformed = self.approximator_.transform(X_val)
        else:
            X_val_transformed = None
        
        # Create optimizer
        self.optimizer_ = self._create_optimizer()
        
        # Fit model using the optimizer
        print("Training classifier with stochastic optimization...")
        self.weights_ = self.optimizer_.fit(
            X_transformed, y_bin, 
            X_val_transformed, None if y_val is None else np.argmax(self.label_binarizer_.transform(y_val), axis=1)
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Transform data
        X_transformed = self.approximator_.transform(X)
        
        # Make predictions
        pred_indices = self.optimizer_.predict(X_transformed)
        return self.label_binarizer_.inverse_transform(
            np.eye(self.n_classes_)[pred_indices]
        )
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Probability of each class.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Transform data
        X_transformed = self.approximator_.transform(X)
        
        # Compute probabilities
        return self.optimizer_.predict_proba(X_transformed)
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns:
        --------
        score : float
            Accuracy of the classifier.
        """
        return np.mean(self.predict(X) == y)