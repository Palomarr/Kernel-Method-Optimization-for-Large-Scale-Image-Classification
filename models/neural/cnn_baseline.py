"""
PyTorch CNN baseline model for comparison with kernel methods.
This implements a simple convolutional neural network for image classification.
"""
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

class CNNModel(nn.Module):
    """
    CNN model
    """
    def __init__(self, input_shape=(28, 28, 1), n_classes=10):
        super(CNNModel, self).__init__()
        self.input_shape = input_shape
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        
        # Calculate size after convolutions
        # For 28x28 input with same padding and 2 MaxPool2d layers: 28/2/2 = 7
        conv_output_size = input_shape[0] // 4 * input_shape[1] // 4 * 64
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """
    CNN classifier using PyTorch.
    """
    
    def __init__(self, input_shape=(28, 28, 1), n_classes=10, 
                 epochs=20, batch_size=128, learning_rate=0.001, 
                 verbose=0, early_stopping=True):
        """
        Initialize the CNN classifier.
        
        Parameters:
        -----------
        input_shape : tuple, default=(28, 28, 1)
            Shape of input images (height, width, channels).
        n_classes : int, default=10
            Number of classes.
        epochs : int, default=20
            Number of epochs to train for.
        batch_size : int, default=128
            Batch size for training.
        learning_rate : float, default=0.001
            Learning rate for optimizer.
        verbose : int, default=0
            Verbosity level (0, 1, or 2).
        early_stopping : bool, default=True
            Whether to use early stopping.
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.early_stopping = early_stopping
        
        # model, loss, and optimizer will be initialized in fit
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self):
        """Build the CNN model."""
        self.model = CNNModel(input_shape=self.input_shape, n_classes=self.n_classes)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the CNN to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. If 2D, will be reshaped to (n_samples, channels, height, width).
        y : array-like of shape (n_samples,)
            Target values.
        X_val : array-like of shape (n_val_samples, n_features), default=None
            Validation data.
        y_val : array-like of shape (n_val_samples,), default=None
            Validation target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Keep track of classes
        self.classes_ = np.unique(y)
        self._build_model()
        
        # Preprocess data
        X_tensor = self._preprocess_X(X)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Create DataLoader for batches
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Set up validation if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = self._preprocess_X(X_val)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Number of epochs with no improvement to wait before early stopping
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Validation phase
            if val_loader is not None:
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                
                # Early stopping check
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if self.verbose > 0:
                                print(f'Early stopping at epoch {epoch+1}')
                            # Restore best model
                            self.model.load_state_dict(best_model_state)
                            break
                
                self.model.train()
                
                if self.verbose > 0 and (epoch+1) % max(1, self.epochs//10) == 0:
                    print(f'Epoch {epoch+1}/{self.epochs} | '
                          f'Loss: {epoch_loss:.4f} | '
                          f'Val Loss: {val_loss:.4f}')
            elif self.verbose > 0 and (epoch+1) % max(1, self.epochs//10) == 0:
                print(f'Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss:.4f}')
                
        # Switch to evaluation mode
        self.model.eval()
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
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
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
        # Ensure model is built and in evaluation mode
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        self.model.eval()
        
        # Preprocess data
        X_tensor = self._preprocess_X(X)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = torch.softmax(logits, dim=1)
        
        return probas.cpu().numpy()
    
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
    
    def _preprocess_X(self, X):
        """
        Preprocess the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or (n_samples, height, width)
            Input data.
            
        Returns:
        --------
        X_tensor : torch.Tensor of shape (n_samples, channels, height, width)
            Reshaped data suitable for CNN input.
        """

        if torch.is_tensor(X):
            # If already a tensor with right shape
            if len(X.shape) == 4 and X.shape[1] == self.input_shape[2]:
                return X.to(self.device)
            # If tensor but needs reshaping
            X = X.numpy()
        
        # If already in shape (N, C, H, W), just convert to tensor
        if len(X.shape) == 4 and X.shape[1] == self.input_shape[2]:
            return torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Reshape flat data to image format
        if len(X.shape) == 2:
            # Calculate height and width assuming square images
            height = width = int(np.sqrt(X.shape[1]))
            # Reshape to (N, H, W, C) format first
            X_reshaped = X.reshape(X.shape[0], height, width, self.input_shape[2])
            X_permuted = np.transpose(X_reshaped, (0, 3, 1, 2))
            return torch.tensor(X_permuted, dtype=torch.float32, device=self.device)
        
        # If shape is (N, H, W), add a channel dimension and permute
        elif len(X.shape) == 3:
            # Add channel dimension to get (N, H, W, 1)
            X_expanded = np.expand_dims(X, axis=3)
            # Permute to (N, C, H, W) format
            X_permuted = np.transpose(X_expanded, (0, 3, 1, 2))
            return torch.tensor(X_permuted, dtype=torch.float32, device=self.device)
            
        elif len(X.shape) == 4:
            X_permuted = np.transpose(X, (0, 3, 1, 2))
            return torch.tensor(X_permuted, dtype=torch.float32, device=self.device)
        
        # Otherwise, raise an error
        else:
            raise ValueError(f"Cannot preprocess X with shape {X.shape}") 