# stochastic_optimizer.py
import numpy as np
from sklearn.utils import shuffle

class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer for kernel-based models.
    
    This optimizer implements SGD with momentum and learning rate scheduling
    for training kernel-based models.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, regularization=0.01, 
                 batch_size=32, max_iter=1000, tol=1e-4, random_state=None):
        """
        Initialize the SGD optimizer.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Initial learning rate.
        momentum : float, default=0.9
            Momentum parameter.
        regularization : float, default=0.01
            L2 regularization parameter.
        batch_size : int, default=32
            Size of minibatches for stochastic optimization.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for stopping criterion.
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.loss_history_ = []
        
    def _init_weights(self, n_features, n_classes):
        """Initialize model weights."""
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, (n_features, n_classes))
        self.velocity_ = np.zeros_like(self.weights_)
        
    def _compute_gradient(self, X_batch, y_batch, weights):
        """Compute gradient on a batch of data."""
        n_samples = X_batch.shape[0]
        
        # Compute linear scores
        scores = X_batch @ weights
        
        # Compute softmax probabilities
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        
        # One-hot encode labels
        n_classes = weights.shape[1]
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_batch] = 1
        
        # Compute gradient
        grad = (1 / n_samples) * X_batch.T @ (probs - y_one_hot)
        
        # Add regularization
        grad += self.regularization * weights
        
        return grad
        
    def _compute_loss(self, X, y, weights):
        """Compute cross-entropy loss."""
        n_samples = X.shape[0]
        
        # Compute linear scores
        scores = X @ weights
        
        # Compute softmax probabilities and loss
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        loss = -np.sum(np.log(probs[np.arange(n_samples), y])) / n_samples
        
        # Add regularization term
        loss += 0.5 * self.regularization * np.sum(weights**2)
        
        return loss
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model to the data using stochastic gradient descent.
        
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
        weights : array of shape (n_features, n_classes)
            Optimized weights.
        """
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        
        # Initialize weights
        self._init_weights(n_features, n_classes)
        
        # Set up early stopping
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_weights = self.weights_.copy()
        
        # Training loop
        for iteration in range(self.max_iter):
            # Shuffle data at each epoch
            X_shuffled, y_shuffled = shuffle(X, y, random_state=self.random_state)
            
            # Process mini-batches
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradient
                grad = self._compute_gradient(X_batch, y_batch, self.weights_)
                
                # Update velocity and weights with momentum
                self.velocity_ = self.momentum * self.velocity_ - self.learning_rate * grad
                self.weights_ += self.velocity_
            
            # Compute loss on training data
            train_loss = self._compute_loss(X, y, self.weights_)
            self.loss_history_.append(train_loss)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {train_loss:.4f}")
            
            # Check for early stopping with validation data
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val, self.weights_)
                
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    best_weights = self.weights_.copy()
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= 5:  # Early stopping
                    print(f"Early stopping at iteration {iteration}")
                    self.weights_ = best_weights
                    break
                    
            # Decay learning rate
            self.learning_rate *= 0.999
        
        return self.weights_
    
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
        scores = X @ self.weights_
        return np.argmax(scores, axis=1)
    
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
        scores = X @ self.weights_
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        return probs