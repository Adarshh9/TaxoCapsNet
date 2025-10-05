"""
Data Preprocessing Module
========================

Functions for loading, preprocessing, and transforming microbiome data.
Includes CLR transformation, standardization, and train-test splitting.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import gmean
import logging
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def clr_transform(data: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """
    Apply Center Log-Ratio (CLR) transformation for compositional data.
    
    CLR transformation addresses the compositional nature of microbiome data
    by transforming relative abundances to overcome the constraint that 
    compositional data sums to a constant.
    
    Args:
        data: Input data array (samples x features)
        pseudocount: Small value to add to zero values to avoid log(0)
        
    Returns:
        CLR-transformed data
    """
    # Add pseudocount to handle zeros
    data_with_pseudo = data + pseudocount
    
    # Calculate geometric mean for each sample
    geom_means = gmean(data_with_pseudo, axis=1)
    
    # Apply CLR transformation: log(x_i / geometric_mean)
    clr_data = np.log(data_with_pseudo / geom_means[:, np.newaxis])
    
    logger.info(f"Applied CLR transformation. Shape: {clr_data.shape}")
    logger.info(f"Data range after CLR: {clr_data.min():.3f} to {clr_data.max():.3f}")
    
    return clr_data


def remove_low_prevalence_features(X: np.ndarray, y: np.ndarray, 
                                 min_prevalence: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """
    Remove features present in fewer than min_prevalence fraction of samples.
    
    Args:
        X: Feature matrix
        y: Target labels  
        min_prevalence: Minimum prevalence threshold (fraction of samples)
        
    Returns:
        Filtered feature matrix and indices of kept features
    """
    # Calculate prevalence for each feature
    prevalence = (X > 1e-10).mean(axis=0)
    
    # Find features above threshold
    keep_features = prevalence >= min_prevalence
    
    logger.info(f"Removing {np.sum(~keep_features)} features with prevalence < {min_prevalence}")
    logger.info(f"Keeping {np.sum(keep_features)} features")
    
    return X[:, keep_features], np.where(keep_features)[0]


def remove_zero_variance_features(X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Remove features with zero variance.
    
    Args:
        X: Feature matrix
        
    Returns:
        Filtered feature matrix and indices of kept features
    """
    # Calculate variance for each feature
    variances = np.var(X, axis=0)
    
    # Find features with non-zero variance
    keep_features = variances > 1e-10
    
    logger.info(f"Removing {np.sum(~keep_features)} zero-variance features")
    
    return X[:, keep_features], np.where(keep_features)[0]


def load_and_preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess microbiome data with comprehensive preprocessing pipeline.
    
    Args:
        df: Input DataFrame with OTU columns and label column
        config: Configuration dictionary
        
    Returns:
        Processed feature matrix, labels, and feature names
    """
    logger.info("Starting data preprocessing pipeline")
    
    # Separate features and labels
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'label' column")
    
    # Get OTU columns (assume all columns except 'label' are OTUs)
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    logger.info(f"Data range: {X.min():.6f} to {X.max():.6f}")
    logger.info(f"Mean abundance: {X.mean():.6f}")
    
    # Remove zero-variance features
    if config.get('data', {}).get('preprocessing', {}).get('remove_zero_variance', True):
        X, kept_indices = remove_zero_variance_features(X)
        feature_cols = [feature_cols[i] for i in kept_indices]
    
    # Remove low-prevalence features
    min_prevalence = config.get('data', {}).get('preprocessing', {}).get('min_prevalence', 0.05)
    if min_prevalence > 0:
        X, kept_indices = remove_low_prevalence_features(X, y, min_prevalence)
        feature_cols = [feature_cols[i] for i in kept_indices]
    
    # Apply CLR transformation
    if config.get('data', {}).get('preprocessing', {}).get('clr_transformation', True):
        X = clr_transform(X)
    
    # Apply standardization
    if config.get('data', {}).get('preprocessing', {}).get('standardization', True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("Applied standardization")
    
    logger.info(f"Final processed data shape: {X.shape}")
    logger.info(f"Final data range: {X.min():.3f} to {X.max():.3f}")
    
    return X, y, feature_cols


def prepare_taxonomy_data(df: pd.DataFrame, taxonomy_map: Dict[str, List[str]]) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Group OTUs by taxonomic classification and prepare multi-input data.
    
    Args:
        df: Input DataFrame with preprocessed data
        taxonomy_map: Dictionary mapping phylum names to OTU lists
        
    Returns:
        List of grouped feature matrices, labels, and phylum names used
    """
    logger.info("Preparing taxonomy-grouped data")
    
    # Load and preprocess data first
    from src.data.preprocessing import load_and_preprocess_data
    X, y, feature_names = load_and_preprocess_data(df, {
        'data': {
            'preprocessing': {
                'clr_transformation': True,
                'remove_zero_variance': True,
                'min_prevalence': 0.05,
                'standardization': False  # Don't standardize before grouping
            }
        }
    })
    
    # Group features by phylum
    X_grouped = []
    used_phyla = []
    
    print("\\nGrouping OTUs by phylum:")
    for phylum, otu_list in taxonomy_map.items():
        # Find indices of OTUs in this phylum that exist in our processed data
        phylum_indices = []
        for otu in otu_list:
            if otu in feature_names:
                phylum_indices.append(feature_names.index(otu))
        
        if len(phylum_indices) >= 3:  # Only include phyla with at least 3 OTUs
            phylum_data = X[:, phylum_indices]
            X_grouped.append(phylum_data)
            used_phyla.append(phylum)
            print(f"  {phylum}: {len(phylum_indices)} OTUs")
        else:
            print(f"  {phylum}: {len(phylum_indices)} OTUs (excluded - too few)")
    
    # Handle unmapped OTUs
    mapped_otu_indices = set()
    for phylum, otu_list in taxonomy_map.items():
        for otu in otu_list:
            if otu in feature_names:
                mapped_otu_indices.add(feature_names.index(otu))
    
    unmapped_indices = [i for i in range(len(feature_names)) if i not in mapped_otu_indices]
    
    if len(unmapped_indices) >= 3:
        unmapped_data = X[:, unmapped_indices]
        X_grouped.append(unmapped_data)
        used_phyla.append("Unknown_Phylum")
        print(f"\\nFound {len(unmapped_indices)} unmapped OTUs")
        print(f"  Unknown_Phylum: {len(unmapped_indices)} OTUs")
    
    print(f"\\nFinal grouping summary:")
    print(f"  Total phylum groups: {len(X_grouped)}")
    print(f"  Total mapped OTUs: {sum(arr.shape[1] for arr in X_grouped)}")
    print(f"  Coverage: {sum(arr.shape[1] for arr in X_grouped)}/{len(feature_names)} ({100*sum(arr.shape[1] for arr in X_grouped)/len(feature_names):.1f}%)")
    
    return X_grouped, y, used_phyla


def create_train_val_test_splits(X: np.ndarray, y: np.ndarray, 
                                test_size: float = 0.2, 
                                val_size: float = 0.1,
                                random_state: int = 42,
                                stratify: bool = True) -> Tuple:
    """
    Create stratified train-validation-test splits.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed
        stratify: Whether to stratify splits
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    stratify_param = y if stratify else None
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Second split: train vs val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        stratify_param = y_train_val if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_param
        )
    else:
        X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None
    
    logger.info(f"Data splits created:")
    logger.info(f"  Training: {X_train.shape[0]} samples")
    if X_val is not None:
        logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_phylum_splits(X_grouped: List[np.ndarray], y: np.ndarray, 
                         test_size: float = 0.2, seed: int = 42) -> Tuple:
    """
    Split each phylum group independently to maintain taxonomic structure.
    
    This function is used for ablation studies to ensure proper data splitting
    per phylum while maintaining biological relationships.
    
    Args:
        X_grouped: List of grouped feature matrices by phylum
        y: Target labels
        test_size: Proportion for test set
        seed: Random seed
        
    Returns:
        X_train_list, X_test_list, y_train, y_test
    """
    X_train_list = []
    X_test_list = []
    
    # Split each phylum separately (same approach as your original code)
    for i, phylum_data in enumerate(X_grouped):
        X_train_phylum, X_test_phylum, y_train, y_test = train_test_split(
            phylum_data, y, test_size=test_size, random_state=seed, stratify=y
        )
        X_train_list.append(X_train_phylum)
        X_test_list.append(X_test_phylum)
    
    return X_train_list, X_test_list, y_train, y_test


def get_data_statistics(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive data statistics.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Dictionary with data statistics
    """
    stats = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': {
            f'class_{i}': count for i, count in enumerate(np.bincount(y))
        },
        'feature_stats': {
            'mean': X.mean(),
            'std': X.std(), 
            'min': X.min(),
            'max': X.max(),
            'sparsity': (X == 0).mean()
        }
    }
    
    return stats
