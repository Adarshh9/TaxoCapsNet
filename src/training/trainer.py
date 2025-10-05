"""
Training Pipeline for TaxoCapsNet
=================================

Main training pipeline that handles TaxoCapsNet and baseline model training,
evaluation, and comparison with comprehensive metrics and confidence intervals.

"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Local imports
from src.models.baseline_models import BaselineModelFactory
from src.utils.metrics import calculate_comprehensive_metrics_with_ci
from src.data.augmentation import apply_augmentation
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TaxoCapsNetTrainer:
    """
    Trainer for TaxoCapsNet model with comprehensive evaluation.
    
    Handles training, validation, and evaluation of the main TaxoCapsNet model
    with proper confidence intervals and statistical analysis.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize TaxoCapsNet trainer.
        
        Args:
            config: Configuration dictionary
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.results = {}
        self.training_history = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info("TaxoCapsNet trainer initialized")
    
    def prepare_train_val_split(self, X_grouped: List[np.ndarray], y: np.ndarray, 
                               config: Dict[str, Any]) -> Tuple:
        """
        Prepare train-validation split for multi-input data.
        
        Args:
            X_grouped: List of grouped feature matrices by phylum
            y: Target labels
            config: Configuration dictionary
            
        Returns:
            Tuple of (X_train_grouped, y_train, X_val_grouped, y_val)
        """
        test_size = config.get('data', {}).get('validation_size', 0.2)
        stratify = config.get('data', {}).get('stratify', True)
        
        # Create indices for splitting
        train_indices, val_indices = train_test_split(
            range(len(y)), 
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
        
        # Split each phylum group
        X_train_grouped = []
        X_val_grouped = []
        
        for phylum_data in X_grouped:
            X_train_grouped.append(phylum_data[train_indices])
            X_val_grouped.append(phylum_data[val_indices])
        
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        logger.info(f"Data split: Train={len(y_train)}, Val={len(y_val)}")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")
        logger.info(f"Val class distribution: {np.bincount(y_val)}")
        
        return X_train_grouped, y_train, X_val_grouped, y_val
    
    def train_model(self, model, X_train_grouped: List[np.ndarray], y_train: np.ndarray,
                   X_val_grouped: List[np.ndarray], y_val: np.ndarray) -> Dict[str, List]:
        """
        Train TaxoCapsNet model with callbacks and monitoring.
        
        Args:
            model: TaxoCapsNet model instance
            X_train_grouped: Training data grouped by phylum
            y_train: Training labels
            X_val_grouped: Validation data grouped by phylum  
            y_val: Validation labels
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting TaxoCapsNet training...")
        
        # Training configuration
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        batch_size = training_config.get('batch_size', 32)
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        if training_config.get('early_stopping', {}).get('enabled', True):
            early_stopping = EarlyStopping(
                monitor=training_config.get('early_stopping', {}).get('monitor', 'val_accuracy'),
                patience=training_config.get('early_stopping', {}).get('patience', 15),
                restore_best_weights=training_config.get('early_stopping', {}).get('restore_best_weights', True),
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning rate scheduling
        if training_config.get('lr_schedule', {}).get('enabled', True):
            lr_reducer = ReduceLROnPlateau(
                monitor='val_loss',
                factor=training_config.get('lr_schedule', {}).get('factor', 0.5),
                patience=training_config.get('lr_schedule', {}).get('patience', 8),
                min_lr=training_config.get('lr_schedule', {}).get('min_lr', 1e-6),
                verbose=1
            )
            callbacks.append(lr_reducer)
        
        # Model checkpointing
        if training_config.get('checkpointing', {}).get('enabled', True):
            checkpoint = ModelCheckpoint(
                'results/models/taxocapsnet_checkpoint.h5',
                monitor=training_config.get('checkpointing', {}).get('monitor', 'val_accuracy'),
                save_best_only=training_config.get('checkpointing', {}).get('save_best_only', True),
                save_weights_only=training_config.get('checkpointing', {}).get('save_weights_only', False),
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train model
        start_time = time.time()
        
        history = model.fit(
            X_train_grouped, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_grouped, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        self.training_history['training_time'] = training_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history.history
    
    def evaluate_model(self, model, X_grouped: List[np.ndarray], y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model with comprehensive metrics and confidence intervals.
        
        Args:
            model: Trained model
            X_grouped: Input data grouped by phylum
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating TaxoCapsNet model...")
        
        # Make predictions
        y_proba_raw = model.predict(X_grouped, verbose=0)
        y_proba = y_proba_raw.ravel() if len(y_proba_raw.shape) > 1 else y_proba_raw
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics with CI
        metrics = calculate_comprehensive_metrics_with_ci(y, y_pred, y_proba, is_binary=True)
        
        # Log results
        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f} (CI: {metrics['sensitivity_ci'][0]:.4f}-{metrics['sensitivity_ci'][1]:.4f})")
        logger.info(f"  Specificity: {metrics['specificity']:.4f} (CI: {metrics['specificity_ci'][0]:.4f}-{metrics['specificity_ci'][1]:.4f})")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  IC: {metrics['ic']:.4f} (CI: {metrics['ic_ci'][0]:.4f}-{metrics['ic_ci'][1]:.4f})")
        
        return metrics


class BaselineModelTrainer:
    """
    Enhanced training pipeline for baseline models with proper CI calculation.
    
    This class is extracted and adapted from your original notebook code
    to handle training of all baseline models with comprehensive metrics.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize baseline model trainer.
        
        Args:
            config: Configuration dictionary
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.results = {}
        self.training_histories = {}
        self.models = {}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info("Enhanced baseline model trainer initialized")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray):
        """
        Prepare and store data for all models.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Handle different input formats
        if isinstance(X, list):  # Multi-input format from TaxoCapsNet
            self.input_shapes = [arr.shape[1:] for arr in X]
            self.total_features = sum(arr.shape[1] for arr in X)
            # Create flattened version for traditional ML
            self.X_flat = np.concatenate(X, axis=1)
        else:
            self.input_shapes = X.shape[1:]
            self.total_features = X.shape[1]
            self.X_flat = X.reshape(X.shape[0], -1)
        
        self.y = y
        self.num_classes = len(np.unique(y))
        self.is_binary = self.num_classes == 2
        
        logger.info(f"Data prepared:")
        logger.info(f"  Samples: {len(y)}")
        logger.info(f"  Total features: {self.total_features}")
        logger.info(f"  Classes: {self.num_classes} ({'Binary' if self.is_binary else 'Multi-class'})")
        logger.info(f"  Class distribution: {np.bincount(y)}")
    
    def prepare_train_val_split(self) -> Tuple:
        """
        Create train-validation split.
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        test_size = self.config.get('data', {}).get('validation_size', 0.2)
        stratify = self.config.get('data', {}).get('stratify', True)
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_flat, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y if stratify else None
        )
        
        return X_train, X_val, y_train, y_val
    
    def train_sklearn_model(self, model, model_name: str, X_train: np.ndarray, 
                           y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train sklearn-based models with enhanced CI metrics.
        
        This method is adapted from your notebook's EnhancedModelTrainingPipeline.
        """
        logger.info(f"🌳 Training {model_name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Probabilities for AUC
        train_proba = None
        val_proba = None
        if hasattr(model, 'predict_proba'):
            train_proba_full = model.predict_proba(X_train)
            val_proba_full = model.predict_proba(X_val)
            if self.is_binary:
                train_proba = train_proba_full[:, 1]
                val_proba = val_proba_full[:, 1]
            else:
                train_proba = train_proba_full
                val_proba = val_proba_full
        
        training_time = time.time() - start_time
        
        # Calculate comprehensive metrics with proper CI
        train_metrics = calculate_comprehensive_metrics_with_ci(y_train, train_pred, train_proba, self.is_binary)
        val_metrics = calculate_comprehensive_metrics_with_ci(y_val, val_pred, val_proba, self.is_binary)
        
        # Store results
        metrics = {}
        for key in train_metrics:
            metrics[f'train_{key}'] = train_metrics[key]
            metrics[f'val_{key}'] = val_metrics[key]
        metrics['training_time'] = training_time
        
        self.models[model_name] = model
        self.results[model_name] = metrics
        
        logger.info(f"✅ {model_name} completed in {training_time:.2f}s")
        logger.info(f"   Val Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"   Val Sensitivity: {metrics['val_sensitivity']:.4f} (CI: {metrics['val_sensitivity_ci'][0]:.4f}-{metrics['val_sensitivity_ci'][1]:.4f})")
        logger.info(f"   Val Specificity: {metrics['val_specificity']:.4f} (CI: {metrics['val_specificity_ci'][0]:.4f}-{metrics['val_specificity_ci'][1]:.4f})")
        logger.info(f"   Val AUC: {metrics['val_auc']:.4f}")
        logger.info(f"   Val IC: {metrics['val_ic']:.4f} (CI: {metrics['val_ic_ci'][0]:.4f}-{metrics['val_ic_ci'][1]:.4f})")
    
    def train_keras_model(self, model, model_name: str, X_train: np.ndarray,
                         y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 100, batch_size: int = 32):
        """
        Train Keras-based models with enhanced CI metrics.
        
        This method is adapted from your notebook's training pipeline.
        """
        logger.info(f"🧠 Training {model_name}...")
        start_time = time.time()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        train_proba_raw = model.predict(X_train, verbose=0)
        val_proba_raw = model.predict(X_val, verbose=0)
        
        if self.is_binary:
            train_proba = train_proba_raw.ravel()
            val_proba = val_proba_raw.ravel()
            train_pred = (train_proba > 0.5).astype(int)
            val_pred = (val_proba > 0.5).astype(int)
        else:
            train_pred = np.argmax(train_proba_raw, axis=1)
            val_pred = np.argmax(val_proba_raw, axis=1)
            train_proba = train_proba_raw
            val_proba = val_proba_raw
        
        # Calculate comprehensive metrics with proper CI
        train_metrics = calculate_comprehensive_metrics_with_ci(y_train, train_pred, train_proba, self.is_binary)
        val_metrics = calculate_comprehensive_metrics_with_ci(y_val, val_pred, val_proba, self.is_binary)
        
        # Store results
        metrics = {}
        for key in train_metrics:
            metrics[f'train_{key}'] = train_metrics[key]
            metrics[f'val_{key}'] = val_metrics[key]
        metrics['training_time'] = training_time
        
        self.models[model_name] = model
        self.results[model_name] = metrics
        self.training_histories[model_name] = history.history
        
        logger.info(f"✅ {model_name} completed in {training_time:.2f}s")
        logger.info(f"   Val Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"   Val Sensitivity: {metrics['val_sensitivity']:.4f} (CI: {metrics['val_sensitivity_ci'][0]:.4f}-{metrics['val_sensitivity_ci'][1]:.4f})")
        logger.info(f"   Val Specificity: {metrics['val_specificity']:.4f} (CI: {metrics['val_specificity_ci'][0]:.4f}-{metrics['val_specificity_ci'][1]:.4f})")
        logger.info(f"   Val AUC: {metrics['val_auc']:.4f}")
        logger.info(f"   Val IC: {metrics['val_ic']:.4f} (CI: {metrics['val_ic_ci'][0]:.4f}-{metrics['val_ic_ci'][1]:.4f})")
    
    def train_all_models(self):
        """
        Train all enabled baseline models with strong regularization.
        
        This method implements the training loop from your original notebook.
        """
        logger.info("🚀 Training Models with Strong Regularization and Enhanced CI Metrics")
        logger.info("="*70)
        
        # Prepare data split
        X_train, X_val, y_train, y_val = self.prepare_train_val_split()
        
        # Initialize model factory
        factory = BaselineModelFactory(self.config, self.random_state)
        
        # Get available models
        available_models = factory.get_available_models()
        
        for model_name, model_type in available_models.items():
            try:
                if model_type == 'sklearn':
                    # Create sklearn model
                    model = factory.create_model(model_name, self.total_features, self.num_classes)
                    self.train_sklearn_model(model, model_name, X_train, y_train, X_val, y_val)
                    
                elif model_type == 'keras':
                    # Create keras model
                    model = factory.create_model(model_name, self.total_features, self.num_classes)
                    # Use smaller epochs and batch size for regularization
                    epochs = self.config.get('baseline_models', {}).get(model_name.lower(), {}).get('epochs', 50)
                    batch_size = self.config.get('baseline_models', {}).get(model_name.lower(), {}).get('batch_size', 8)
                    self.train_keras_model(model, model_name, X_train, y_train, X_val, y_val, epochs, batch_size)
                    
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {str(e)}")
                continue
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return pd.DataFrame()
        
        model_names = list(self.results.keys())
        
        # Prepare data for DataFrame
        metrics_data = {
            'Model': model_names,
            'Val_Accuracy': [self.results[name]['val_accuracy'] for name in model_names],
            'Val_Sensitivity': [self.results[name]['val_sensitivity'] for name in model_names],
            'Val_Specificity': [self.results[name]['val_specificity'] for name in model_names],
            'Val_Precision': [self.results[name]['val_precision'] for name in model_names],
            'Val_F1': [self.results[name]['val_f1'] for name in model_names],
            'Val_AUC': [self.results[name]['val_auc'] for name in model_names],
            'Val_IC': [self.results[name]['val_ic'] for name in model_names],
            'IC_CI_Lower': [self.results[name].get('val_ic_ci', (np.nan, np.nan))[0] for name in model_names],
            'IC_CI_Upper': [self.results[name].get('val_ic_ci', (np.nan, np.nan))[1] for name in model_names],
            'Training_Time': [self.results[name]['training_time'] for name in model_names]
        }
        
        return pd.DataFrame(metrics_data)
    
    def print_results_table(self):
        """
        Print detailed results table with CI formatting.
        
        This method recreates the table printing from your original notebook.
        """
        if not self.results:
            logger.warning("No results to display. Train models first.")
            return
        
        logger.info("\\n" + "="*160)
        logger.info("📊 COMPREHENSIVE METRICS COMPARISON WITH CONFIDENCE INTERVALS")
        logger.info("="*160)
        
        # Header with CI columns
        header = (f"{'Model':<15} {'Accuracy':<9} {'Sensitivity':<15} {'Specificity':<15} "
                 f"{'Precision':<10} {'F1 Score':<9} {'AUC':<7} {'CI':<15} {'Time (s)':<9}")
        logger.info(header)
        logger.info("-" * 160)
        
        # Sort models by validation accuracy
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['val_accuracy'], reverse=True)
        
        for model_name, metrics in sorted_models:
            # Format CI values properly
            sensitivity_ci = metrics.get('val_sensitivity_ci', (np.nan, np.nan))
            specificity_ci = metrics.get('val_specificity_ci', (np.nan, np.nan))
            ic_ci = metrics.get('val_ic_ci', (np.nan, np.nan))
            
            # Format CI display
            if not (np.isnan(ic_ci[0]) or np.isnan(ic_ci[1])):
                ci_display = f"({ic_ci[0]:.2f}, {ic_ci[1]:.2f})"
            else:
                ci_display = "(NA, NA)"
            
            logger.info(f"{model_name:<15} "
                       f"{metrics['val_accuracy']:<9.3f} "
                       f"{metrics['val_sensitivity']:<15.3f} "
                       f"{metrics['val_specificity']:<15.3f} "
                       f"{metrics['val_precision']:<10.3f} "
                       f"{metrics['val_f1']:<9.3f} "
                       f"{metrics['val_auc']:<7.3f} "
                       f"{ci_display:<15} "
                       f"{metrics['training_time']:<9.1f}")
        
        logger.info("="*160)


def run_baseline_comparison(config: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Tuple:
    """
    Run complete baseline model comparison pipeline.
    
    This function recreates the main comparison pipeline from your notebook.
    
    Args:
        config: Configuration dictionary
        X: Feature matrix  
        y: Target labels
        
    Returns:
        Tuple of (trainer, results_df)
    """
    # Initialize trainer
    trainer = BaselineModelTrainer(config, random_state=config.get('data', {}).get('random_state', 42))
    
    # Prepare data
    trainer.prepare_data(X, y)
    
    # Train all models
    trainer.train_all_models()
    
    # Print results
    trainer.print_results_table()
    
    # Generate results DataFrame
    results_df = trainer.get_results_dataframe()
    
    return trainer, results_df


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing training pipeline...")
    
    # Sample configuration
    config = {
        'data': {'validation_size': 0.2, 'random_state': 42},
        'training': {'epochs': 50, 'batch_size': 16},
        'baseline_models': {
            'random_forest': {'enabled': True},
            'simple_nn': {'enabled': True}
        }
    }
    
    # Sample data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 2, 100)
    
    # Run comparison
    trainer, results_df = run_baseline_comparison(config, X, y)
    
    print("Training pipeline test completed!")
    print(f"Results shape: {results_df.shape}")
