# data_loader.py
import os
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# Fix the path to use the current directory structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

logger.info(f"Using data directory: {DATA_DIR}")
logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")

# Updated URLs for Sign Language MNIST dataset
# These URLs are from a fork of the original repository
TRAIN_URL = "https://github.com/adilgupta/sign-language-mnist/raw/master/sign_mnist_train.csv"
TEST_URL = "https://github.com/adilgupta/sign-language-mnist/raw/master/sign_mnist_test.csv"

def download_data():
    """
    Download the Sign Language MNIST dataset if it doesn't exist
    """
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Download training data
    train_path = os.path.join(RAW_DATA_DIR, 'sign_mnist_train.csv')
    if not os.path.exists(train_path):
        logger.info(f"Downloading training data to {train_path}...")
        try:
            urlretrieve(TRAIN_URL, train_path)
            logger.info("Training data downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading training data: {e}")
            logger.info("Trying to use cached data if available...")
            # If data files already exist in processed directory, we can skip the download
            if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')):
                logger.info("Found cached processed data. Using that instead.")
                return True
            return False
    else:
        logger.info(f"Training data already exists at {train_path}.")
    
    # Download test data
    test_path = os.path.join(RAW_DATA_DIR, 'sign_mnist_test.csv')
    if not os.path.exists(test_path):
        logger.info(f"Downloading test data to {test_path}...")
        try:
            urlretrieve(TEST_URL, test_path)
            logger.info("Test data downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading test data: {e}")
            logger.info("Trying to use cached data if available...")
            # If data files already exist in processed directory, we can skip the download
            if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy')):
                logger.info("Found cached processed data. Using that instead.")
                return True
            return False
    else:
        logger.info(f"Test data already exists at {test_path}.")
    
    return True

def load_and_preprocess_data(validation_split=0.2, random_state=42):
    """
    Load and preprocess the Sign Language MNIST dataset
    
    Args:
        validation_split: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Check if processed data already exists
    if (os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')) and
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))):
        logger.info("Found processed data. Loading it directly.")
        return load_processed_data()
    
    # Try to download data
    if not download_data():
        raise RuntimeError("Failed to download data and no cached data available.")
    
    # Load data
    train_path = os.path.join(RAW_DATA_DIR, 'sign_mnist_train.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'sign_mnist_test.csv')
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Data files not found. Try running the compare-cnn script first.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split features and labels
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=random_state
    )
    
    # Reshape images to be 28x28
    X_train_reshaped = X_train.reshape(-1, 28, 28)
    X_val_reshaped = X_val.reshape(-1, 28, 28)
    X_test_reshaped = X_test.reshape(-1, 28, 28)
    
    # Save processed data
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save reshaped data as well (useful for visualization)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train_reshaped.npy'), X_train_reshaped)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val_reshaped.npy'), X_val_reshaped)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test_reshaped.npy'), X_test_reshaped)
    
    logger.info("Data preprocessing completed successfully.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_processed_data():
    """
    Load the preprocessed data from disk
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Check if processed data exists
    if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy')):
        logger.info("Processed data not found. Processing data...")
        return load_and_preprocess_data()
    
    # Load processed data
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    logger.info("Processed data loaded successfully.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_reshaped_data():
    """
    Load the reshaped data (images in 28x28 format) from disk
    
    Returns:
        X_train_reshaped, X_val_reshaped, X_test_reshaped, y_train, y_val, y_test
    """
    # Check if processed data exists
    if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train_reshaped.npy')):
        logger.info("Reshaped data not found. Processing data...")
        load_and_preprocess_data()
    
    # Load reshaped data
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_reshaped.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val_reshaped.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_reshaped.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    # Verify data shapes are correct
    if X_train.ndim != X_test.ndim:
        logger.warning(f"Dimension mismatch: X_train has {X_train.ndim}D but X_test has {X_test.ndim}D")
        
        # If one is 1D but should be 3D, try to load the flattened version and reshape
        if X_test.ndim == 1:
            # Try to load the original flattened data
            X_test_flat = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
            if X_test_flat.ndim == 2:
                logger.info(f"Loaded flattened X_test with shape {X_test_flat.shape}")
                # Reshape to match X_train dimensions
                if X_train.ndim == 3:
                    img_size = int(np.sqrt(X_test_flat.shape[1]))
                    if img_size * img_size == X_test_flat.shape[1]:
                        X_test = X_test_flat.reshape(-1, img_size, img_size)
                        logger.info(f"Reshaped X_test to {X_test.shape}")
                    else:
                        logger.warning("Cannot reshape X_test to match X_train dimensions")
            else:
                logger.warning("Flattened X_test has unexpected dimensions")
        
    logger.info("Reshaped data loaded successfully.")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    

def get_sample_dataset(n_samples=1000, random_state=42):
    """
    Get a small sample of the dataset for quick experimentation
    
    Args:
        n_samples: Number of samples to include
        random_state: Random seed for reproducibility
        
    Returns:
        X_train_sample, X_val_sample, y_train_sample, y_val_sample
    """
    X_train, X_val, _, y_train, y_val, _ = load_processed_data()
    
    # Create a smaller random sample
    np.random.seed(random_state)
    train_indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
    val_indices = np.random.choice(len(X_val), min(n_samples // 5, len(X_val)), replace=False)
    
    X_train_sample = X_train[train_indices]
    y_train_sample = y_train[train_indices]
    X_val_sample = X_val[val_indices]
    y_val_sample = y_val[val_indices]
    
    logger.info(f"Created sample dataset with {len(X_train_sample)} training and {len(X_val_sample)} validation samples.")
    
    return X_train_sample, X_val_sample, y_train_sample, y_val_sample

if __name__ == "__main__":
    # If run as script, download and preprocess the data
    load_and_preprocess_data()