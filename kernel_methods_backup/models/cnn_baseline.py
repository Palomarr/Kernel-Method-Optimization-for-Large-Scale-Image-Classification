"""
CNN baseline model for comparison with kernel methods.
This implements a simple convolutional neural network for image classification.
"""
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin

class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    CNN classifier for image classification.
    
    This is a simple convolutional neural network that follows the sklearn
    estimator API for easy comparison with kernel methods.
    """
    
    def __init__(self, input_shape=(28, 28, 1), n_classes=10, 
                 epochs=20, batch_size=128, learning_rate=0.001, 
                 verbose=0, early_stopping=True):
        """
        Initialize the CNN classifier.
        
        Parameters:
        -----------
        input_shape : tuple, default=(28, 28, 1)
            Shape of input images.
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
        self.model = None
        self._build_model()
        
    def _build_model(self):
        """Build the CNN model."""
        self.model = Sequential([
            # First convolutional block
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                  input_shape=self.input_shape),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the CNN to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. If 2D, will be reshaped to (n_samples, height, width, channels).
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
        
        # Preprocess the data
        X_reshaped = self._preprocess_X(X)
        y_categorical = to_categorical(y, num_classes=self.n_classes)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_reshaped = self._preprocess_X(X_val)
            y_val_categorical = to_categorical(y_val, num_classes=self.n_classes)
            validation_data = (X_val_reshaped, y_val_categorical)
        
        # Set up callbacks
        callbacks = []
        if self.early_stopping and validation_data is not None:
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            ))
        
        # Train the model
        self.model.fit(
            X_reshaped, y_categorical,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=validation_data,
            callbacks=callbacks
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
        X_reshaped = self._preprocess_X(X)
        y_pred_proba = self.model.predict(X_reshaped, verbose=0)
        return np.argmax(y_pred_proba, axis=1)
    
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
        X_reshaped = self._preprocess_X(X)
        return self.model.predict(X_reshaped, verbose=0)
    
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
        X_reshaped : array of shape (n_samples, height, width, channels)
            Reshaped data suitable for CNN input.
        """
        # If X is already in the right shape, just return it
        if len(X.shape) == 4 and X.shape[1:] == self.input_shape:
            return X
        
        # Reshape flat data to image format
        if len(X.shape) == 2:
            # Calculate height and width assuming square images
            height = width = int(np.sqrt(X.shape[1]))
            X_reshaped = X.reshape(X.shape[0], height, width, 1)
            return X_reshaped
        
        # If X is 3D (samples, height, width), add channel dimension
        elif len(X.shape) == 3:
            return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Otherwise, raise an error
        else:
            raise ValueError(f"Cannot preprocess X with shape {X.shape}") 