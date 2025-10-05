"""
Data Augmentation Functions
===========================

Implementation of data augmentation techniques for microbiome data.
Includes compositional and Dirichlet augmentation methods.

"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def compositional_augmentation(X: np.ndarray, y: np.ndarray, n_augment: int = 50, 
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment microbiome data preserving compositional constraints.
    
    This function generates synthetic samples by applying log-normal noise
    to original samples while preserving the compositional nature of the data.
    
    Args:
        X: Original feature matrix (samples x features)
        y: Original labels
        n_augment: Number of augmented samples to generate per original sample
        noise_level: Standard deviation of log-normal noise
        
    Returns:
        Tuple of (augmented_X, augmented_y) including original data
    """
    logger.info(f"Starting compositional augmentation: {X.shape} -> target {n_augment} samples per original")
    
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        for _ in range(n_augment):
            # Generate log-normal noise
            noise = np.random.lognormal(0, noise_level, X.shape[1])
            
            # Apply noise to original sample
            augmented_sample = X[i] * noise
            
            # Preserve total abundance (compositional constraint)
            if augmented_sample.sum() > 0:
                augmented_sample = augmented_sample / augmented_sample.sum() * X[i].sum()
                X_augmented.append(augmented_sample)
                y_augmented.append(y[i])
    
    # Combine original and augmented data
    X_final = np.vstack([X, np.array(X_augmented)])
    y_final = np.hstack([y, np.array(y_augmented)])
    
    logger.info(f"Generated {len(X_augmented)} synthetic samples")
    logger.info(f"Final dataset: {X_final.shape}")
    
    return X_final, y_final


def dirichlet_augmentation(X: np.ndarray, y: np.ndarray, n_augment: int = 50, 
                          concentration: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced Dirichlet augmentation with better biological constraints.
    
    This function uses Dirichlet distribution to generate synthetic samples
    that respect the compositional nature of microbiome data while preserving
    biological constraints such as sparsity patterns.
    
    Args:
        X: Original feature matrix (samples x features)
        y: Original labels
        n_augment: Total number of synthetic samples to generate
        concentration: Concentration parameter for Dirichlet distribution
        
    Returns:
        Tuple of (augmented_X, augmented_y) including original data
    """
    logger.info(f"Starting Dirichlet augmentation: {X.shape} -> target {n_augment} samples")
    
    if len(X) == 0 or X.shape[1] == 0:
        logger.warning("Empty input data, returning original data")
        return X, y
    
    X_augmented = []
    y_augmented = []
    
    # Calculate biological bounds per feature
    feature_mins = np.percentile(X, 5, axis=0)
    feature_maxs = np.percentile(X, 95, axis=0)
    feature_stds = np.std(X, axis=0)
    
    samples_per_original = max(1, n_augment // len(X))
    
    for i in range(len(X)):
        # Identify non-zero features for sparsity preservation
        non_zero_mask = X[i] > 1e-10
        
        if np.sum(non_zero_mask) < 2:
            # Conservative perturbation for sparse samples
            for _ in range(samples_per_original):
                noise_scale = feature_stds * 0.01
                noise = np.random.normal(0, noise_scale)
                augmented = X[i] + noise
                
                # Apply biological bounds
                augmented = np.clip(augmented, feature_mins, feature_maxs)
                augmented = np.maximum(augmented, 0)
                
                X_augmented.append(augmented)
                y_augmented.append(y[i])
            continue
        
        # Enhanced Dirichlet sampling
        for attempt in range(samples_per_original):
            try:
                augmented_sample = X[i].copy()
                
                # Work only with non-zero features
                non_zero_values = X[i][non_zero_mask]
                total_abundance = np.sum(non_zero_values)
                
                if total_abundance > 0:
                    # Higher concentration for more conservative sampling
                    rel_abundances = non_zero_values / total_abundance
                    alpha = rel_abundances * concentration + 1e-6
                    alpha = np.maximum(alpha, 1e-6)
                    
                    # Sample from Dirichlet with retry mechanism
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            new_rel_abundances = np.random.dirichlet(alpha)
                            
                            # Scale back with slight variation (±5%)
                            abundance_variation = np.random.uniform(0.95, 1.05)
                            new_total = total_abundance * abundance_variation
                            
                            augmented_sample[non_zero_mask] = new_rel_abundances * new_total
                            break
                        except:
                            if retry == max_retries - 1:
                                # Final fallback: small perturbation
                                perturbation = np.random.normal(1, 0.02, np.sum(non_zero_mask))
                                perturbation = np.clip(perturbation, 0.98, 1.02)
                                augmented_sample[non_zero_mask] = non_zero_values * perturbation
                    
                    # Apply biological constraints
                    augmented_sample = np.clip(augmented_sample, feature_mins, feature_maxs)
                    augmented_sample = np.maximum(augmented_sample, 0)
                    
                    # Ensure realistic distance from original
                    distance = np.linalg.norm(augmented_sample - X[i])
                    original_norm = np.linalg.norm(X[i])
                    if distance > 0.2 * original_norm:
                        # Scale back toward original
                        direction = augmented_sample - X[i]
                        max_distance = 0.2 * original_norm
                        augmented_sample = X[i] + direction * (max_distance / distance)
                    
                    X_augmented.append(augmented_sample)
                    y_augmented.append(y[i])
                
            except Exception as e:
                logger.warning(f"Dirichlet sampling failed for sample {i}: {e}")
                # Conservative fallback
                noise_factor = np.random.normal(1, 0.01, X.shape[1])
                noise_factor = np.clip(noise_factor, 0.99, 1.01)
                augmented = X[i] * noise_factor
                augmented = np.clip(augmented, feature_mins, feature_maxs)
                X_augmented.append(augmented)
                y_augmented.append(y[i])
    
    if len(X_augmented) == 0:
        logger.warning("No synthetic samples generated, returning original data")
        return X, y
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    # Combine original and augmented data
    X_final = np.vstack([X, X_augmented])
    y_final = np.hstack([y, y_augmented])
    
    logger.info(f"Successfully generated {len(X_augmented)} synthetic samples")
    logger.info(f"Final dataset: {X_final.shape}")
    logger.info(f"Final class distribution: {np.bincount(y_final)}")
    
    return X_final, y_final


def gaussian_noise_augmentation(X: np.ndarray, y: np.ndarray, n_augment: int = 50,
                               noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple Gaussian noise augmentation.
    
    Adds Gaussian noise to the original samples. Less sophisticated than
    compositional methods but sometimes effective as a baseline.
    
    Args:
        X: Original feature matrix
        y: Original labels
        n_augment: Number of augmented samples per original sample
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Tuple of (augmented_X, augmented_y) including original data
    """
    logger.info(f"Starting Gaussian noise augmentation: {X.shape}")
    
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        for _ in range(n_augment):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, X.shape[1])
            augmented_sample = X[i] + noise
            
            # Ensure non-negative values (important for microbiome data)
            augmented_sample = np.maximum(augmented_sample, 0)
            
            X_augmented.append(augmented_sample)
            y_augmented.append(y[i])
    
    # Combine original and augmented data
    X_final = np.vstack([X, np.array(X_augmented)])
    y_final = np.hstack([y, np.array(y_augmented)])
    
    logger.info(f"Generated {len(X_augmented)} synthetic samples with Gaussian noise")
    return X_final, y_final


def mixup_augmentation(X: np.ndarray, y: np.ndarray, n_augment: int = 50,
                      alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixup augmentation for microbiome data.
    
    Creates synthetic samples by mixing pairs of samples from the same class.
    Adapted for microbiome data by preserving compositional constraints.
    
    Args:
        X: Original feature matrix
        y: Original labels
        n_augment: Total number of synthetic samples to generate
        alpha: Beta distribution parameter for mixing ratio
        
    Returns:
        Tuple of (augmented_X, augmented_y) including original data
    """
    logger.info(f"Starting Mixup augmentation: {X.shape}")
    
    X_augmented = []
    y_augmented = []
    
    # Group samples by class
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
    
    samples_per_class = n_augment // len(unique_classes)
    
    for cls in unique_classes:
        indices = class_indices[cls]
        if len(indices) < 2:
            continue
        
        for _ in range(samples_per_class):
            # Randomly select two samples from the same class
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            
            # Sample mixing ratio from Beta distribution
            lam = np.random.beta(alpha, alpha)
            
            # Create mixed sample
            mixed_sample = lam * X[idx1] + (1 - lam) * X[idx2]
            
            # Preserve compositional nature if needed
            if mixed_sample.sum() > 0:
                original_sum = (X[idx1].sum() + X[idx2].sum()) / 2
                mixed_sample = mixed_sample / mixed_sample.sum() * original_sum
            
            X_augmented.append(mixed_sample)
            y_augmented.append(cls)
    
    # Combine original and augmented data
    X_final = np.vstack([X, np.array(X_augmented)])
    y_final = np.hstack([y, np.array(y_augmented)])
    
    logger.info(f"Generated {len(X_augmented)} synthetic samples with Mixup")
    return X_final, y_final


def validate_augmented_data(X_original: np.ndarray, X_augmented: np.ndarray, 
                           y_original: np.ndarray, y_augmented: np.ndarray) -> dict:
    """
    Validate augmented data quality.
    
    Checks various properties of augmented data to ensure it maintains
    realistic characteristics compared to original data.
    
    Args:
        X_original: Original feature matrix
        X_augmented: Augmented feature matrix (including original)
        y_original: Original labels
        y_augmented: Augmented labels (including original)
        
    Returns:
        Dictionary with validation metrics
    """
    validation_results = {
        'original_samples': len(X_original),
        'augmented_samples': len(X_augmented),
        'augmentation_ratio': len(X_augmented) / len(X_original),
        'class_balance_preserved': True,
        'feature_statistics': {}
    }
    
    # Check class balance preservation
    original_class_dist = np.bincount(y_original) / len(y_original)
    augmented_class_dist = np.bincount(y_augmented) / len(y_augmented)
    
    max_class_diff = np.max(np.abs(original_class_dist - augmented_class_dist))
    validation_results['max_class_distribution_change'] = max_class_diff
    validation_results['class_balance_preserved'] = max_class_diff < 0.1
    
    # Feature statistics comparison
    for i, stat_name in enumerate(['mean', 'std', 'min', 'max']):
        original_stat = getattr(X_original, stat_name)(axis=0)
        augmented_stat = getattr(X_augmented, stat_name)(axis=0)
        
        # Calculate relative change
        rel_change = np.abs(augmented_stat - original_stat) / (original_stat + 1e-8)
        validation_results['feature_statistics'][f'{stat_name}_max_rel_change'] = np.max(rel_change)
        validation_results['feature_statistics'][f'{stat_name}_mean_rel_change'] = np.mean(rel_change)
    
    # Sparsity preservation
    original_sparsity = (X_original == 0).mean()
    augmented_sparsity = (X_augmented == 0).mean()
    validation_results['sparsity_change'] = abs(augmented_sparsity - original_sparsity)
    
    return validation_results


def apply_augmentation(X: np.ndarray, y: np.ndarray, method: str = 'dirichlet',
                      config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply specified augmentation method with configuration.
    
    Args:
        X: Original feature matrix
        y: Original labels
        method: Augmentation method ('dirichlet', 'compositional', 'gaussian', 'mixup')
        config: Configuration dictionary for augmentation parameters
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if config is None:
        config = {}
    
    if method == 'dirichlet':
        return dirichlet_augmentation(
            X, y,
            n_augment=config.get('n_augment', 50),
            concentration=config.get('concentration', 100.0)
        )
    elif method == 'compositional':
        return compositional_augmentation(
            X, y,
            n_augment=config.get('n_augment', 50),
            noise_level=config.get('noise_level', 0.1)
        )
    elif method == 'gaussian':
        return gaussian_noise_augmentation(
            X, y,
            n_augment=config.get('n_augment', 50),
            noise_std=config.get('noise_std', 0.01)
        )
    elif method == 'mixup':
        return mixup_augmentation(
            X, y,
            n_augment=config.get('n_augment', 50),
            alpha=config.get('alpha', 0.2)
        )
    else:
        raise ValueError(f"Unknown augmentation method: {method}")


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing data augmentation functions...")
    
    # Generate sample microbiome data
    np.random.seed(42)
    n_samples, n_features = 50, 100
    
    # Create sparse, compositional-like data
    X = np.random.exponential(0.1, (n_samples, n_features))
    X = X / X.sum(axis=1, keepdims=True)  # Make compositional
    
    # Create binary labels
    y = np.random.randint(0, 2, n_samples)
    
    print(f"Original data: {X.shape}, Class distribution: {np.bincount(y)}")
    
    # Test each augmentation method
    methods = ['dirichlet', 'compositional', 'gaussian', 'mixup']
    
    for method in methods:
        print(f"\\nTesting {method} augmentation...")
        try:
            X_aug, y_aug = apply_augmentation(X, y, method=method, config={'n_augment': 20})
            validation = validate_augmented_data(X, X_aug, y, y_aug)
            
            print(f"  Augmented data: {X_aug.shape}")
            print(f"  Augmentation ratio: {validation['augmentation_ratio']:.1f}x")
            print(f"  Class balance preserved: {validation['class_balance_preserved']}")
            print(f"  Max class distribution change: {validation['max_class_distribution_change']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\\nAugmentation testing complete!")
