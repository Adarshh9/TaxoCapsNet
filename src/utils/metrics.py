"""
Custom Metrics and Statistical Functions
========================================

Implementation of custom metrics with confidence intervals for microbiome classification.
Includes Wilson score CI, bootstrap CI, and Information Coefficient calculation.

"""

import numpy as np
import scipy.stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mutual_info_score
)
from sklearn.utils import resample
from typing import Tuple, Callable, Any
import logging

logger = logging.getLogger(__name__)


def wilson_confidence_interval(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for proportions.
    
    More accurate than normal approximation for small samples or extreme proportions.
    Used for sensitivity and specificity confidence intervals.
    
    Args:
        successes: Number of positive outcomes
        trials: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)
    
    p = successes / trials
    z = scipy.stats.norm.ppf(1 - alpha/2)
    
    denominator = 1 + z**2 / trials
    centre_adjusted = p + z**2 / (2 * trials)
    adjusted_std = np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)
    
    lower_bound = (centre_adjusted - z * adjusted_std) / denominator
    upper_bound = (centre_adjusted + z * adjusted_std) / denominator
    
    return (max(0, lower_bound), min(1, upper_bound))


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, 
                                metric_func: Callable, n_bootstrap: int = 1000, 
                                alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for any metric.
    
    Uses bootstrap resampling to estimate the sampling distribution
    of a metric and calculate confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        metric_func: Function to calculate metric (y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_scores = []
    n_samples = len(y_true)
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric for bootstrap sample
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            if not np.isnan(score):
                bootstrap_scores.append(score)
        except:
            continue
    
    if len(bootstrap_scores) == 0:
        return (np.nan, np.nan)
    
    # Calculate confidence interval
    bootstrap_scores = np.array(bootstrap_scores)
    lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    
    return (lower, upper)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (True Negative Rate) for binary classification.
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return np.nan


def calculate_ic_with_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                        n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Information Coefficient (IC) with confidence interval.
    
    Information Coefficient measures the mutual information between
    true labels and predictions, normalized by the entropy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples for CI
        alpha: Significance level
        
    Returns:
        Tuple of (IC_score, (lower_bound, upper_bound))
    """
    try:
        # Primary IC calculation
        ic_score = mutual_info_score(y_true, y_pred)
        
        # Bootstrap confidence interval
        ic_ci = bootstrap_confidence_interval(
            y_true, y_pred,
            lambda yt, yp: mutual_info_score(yt, yp),
            n_bootstrap, alpha
        )
        
        return ic_score, ic_ci
    
    except Exception as e:
        logger.warning(f"IC calculation failed: {e}")
        return np.nan, (np.nan, np.nan)


def calculate_comprehensive_metrics_with_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                                          y_proba: np.ndarray = None, 
                                          is_binary: bool = True) -> dict:
    """
    Calculate comprehensive metrics with proper confidence intervals.
    
    This is the main function used by the training pipeline to calculate
    all metrics including accuracy, sensitivity, specificity, precision, F1, AUC, and IC
    with appropriate confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for AUC calculation)
        is_binary: Whether this is binary classification
        
    Returns:
        Dictionary with all metrics and their confidence intervals
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if is_binary:
        # Binary classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Same as recall for binary
        metrics['specificity'] = specificity_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Calculate confidence intervals for sensitivity and specificity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity CI (True Positive Rate)
            sensitivity_ci = wilson_confidence_interval(tp, tp + fn)
            metrics['sensitivity_ci'] = sensitivity_ci
            
            # Specificity CI (True Negative Rate) 
            specificity_ci = wilson_confidence_interval(tn, tn + fp)
            metrics['specificity_ci'] = specificity_ci
        else:
            metrics['sensitivity_ci'] = (np.nan, np.nan)
            metrics['specificity_ci'] = (np.nan, np.nan)
        
        # AUC calculation
        try:
            if y_proba is not None:
                # Use probabilities if available
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    # Multi-class probabilities - use positive class
                    y_proba_binary = y_proba[:, 1]
                else:
                    # Already binary probabilities
                    y_proba_binary = y_proba.ravel() if len(y_proba.shape) > 1 else y_proba
                
                metrics['auc'] = roc_auc_score(y_true, y_proba_binary)
            else:
                # Fall back to predictions
                metrics['auc'] = roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"AUC calculation failed: {e}")
            metrics['auc'] = 0.0
    
    else:
        # Multi-class metrics (weighted average)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['sensitivity'] = metrics['recall']
        metrics['specificity'] = np.nan  # Not well-defined for multi-class
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['auc'] = 0.0  # AUC calculation for multi-class is more complex
        
        # For multi-class, CI calculation is more complex - set to NaN
        metrics['sensitivity_ci'] = (np.nan, np.nan)
        metrics['specificity_ci'] = (np.nan, np.nan)
    
    # Information Coefficient (IC) with confidence interval
    try:
        ic_score, ic_ci = calculate_ic_with_ci(y_true, y_pred)
        metrics['ic'] = ic_score
        metrics['ic_ci'] = ic_ci
    except Exception as e:
        logger.warning(f"IC calculation error: {e}")
        metrics['ic'] = np.nan
        metrics['ic_ci'] = (np.nan, np.nan)
    
    return metrics


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two classifiers.
    
    Tests whether two classifiers have significantly different error rates
    on the same dataset. Useful for comparing model performance.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from first classifier
        y_pred2: Predictions from second classifier
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    # Create contingency table
    # Rows: model 1 correct/incorrect
    # Cols: model 2 correct/incorrect
    
    model1_correct = (y_true == y_pred1)
    model2_correct = (y_true == y_pred2)
    
    # Count cases
    both_correct = np.sum(model1_correct & model2_correct)
    both_incorrect = np.sum(~model1_correct & ~model2_correct)
    model1_correct_model2_incorrect = np.sum(model1_correct & ~model2_correct)
    model1_incorrect_model2_correct = np.sum(~model1_correct & model2_correct)
    
    # McNemar's test statistic
    # Uses the cases where models disagree
    b = model1_correct_model2_incorrect
    c = model1_incorrect_model2_correct
    
    if b + c == 0:
        # No disagreements - models perform identically
        return 0.0, 1.0
    
    # Chi-square test statistic with continuity correction
    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    
    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - scipy.stats.chi2.cdf(chi_square, df=1)
    
    return chi_square, p_value


def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test for comparing model performances.
    
    Args:
        scores1: Performance scores from first model
        scores2: Performance scores from second model
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have same length")
    
    # Paired t-test
    t_stat, p_val = scipy.stats.ttest_rel(scores1, scores2)
    
    return t_stat, p_val


def calculate_effect_size(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for the difference between two groups.
    
    Effect size interpretation:
    - Small: 0.2
    - Medium: 0.5  
    - Large: 0.8
    
    Args:
        scores1: Scores from first group
        scores2: Scores from second group
        
    Returns:
        Cohen's d effect size
    """
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
    n1, n2 = len(scores1), len(scores2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d


def format_ci_string(value: float, ci_tuple: Tuple[float, float], decimals: int = 3) -> str:
    """
    Format a value with confidence interval as string.
    
    Args:
        value: Point estimate
        ci_tuple: (lower_bound, upper_bound)
        decimals: Number of decimal places
        
    Returns:
        Formatted string like "0.942 (0.901-0.967)"
    """
    if np.isnan(value) or np.isnan(ci_tuple[0]) or np.isnan(ci_tuple[1]):
        return f"{value:.{decimals}f} (NA)"
    
    return f"{value:.{decimals}f} ({ci_tuple[0]:.{decimals}f}-{ci_tuple[1]:.{decimals}f})"


def statistical_significance_test(model1_scores: dict, model2_scores: dict, 
                                metric: str = 'accuracy') -> dict:
    """
    Perform statistical significance tests between two models.
    
    Args:
        model1_scores: Dictionary with scores from model 1
        model2_scores: Dictionary with scores from model 2  
        metric: Metric to compare
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Extract scores for the specified metric
    if f'{metric}_values' in model1_scores and f'{metric}_values' in model2_scores:
        scores1 = np.array(model1_scores[f'{metric}_values'])
        scores2 = np.array(model2_scores[f'{metric}_values'])
        
        # Paired t-test
        t_stat, p_val = paired_t_test(scores1, scores2)
        effect_size = calculate_effect_size(scores1, scores2)
        
        results.update({
            'metric': metric,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'mean_difference': np.mean(scores1) - np.mean(scores2),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'significant': p_val < 0.05
        })
    
    return results


if __name__ == "__main__":
    # Test the metrics functions
    print("Testing custom metrics...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    
    # Test comprehensive metrics
    metrics = calculate_comprehensive_metrics_with_ci(y_true, y_pred, y_proba)
    
    print("Sample metrics with CI:")
    for key, value in metrics.items():
        if '_ci' in key:
            print(f"  {key}: {value}")
        elif key not in ['sensitivity_ci', 'specificity_ci', 'ic_ci']:
            ci_key = f"{key}_ci"
            if ci_key in metrics:
                print(f"  {key}: {format_ci_string(value, metrics[ci_key])}")
            else:
                print(f"  {key}: {value:.3f}")
    
    print("\\nTesting complete!")
