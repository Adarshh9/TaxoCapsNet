"""
Baseline Model Implementations
==============================

Collection of baseline models for comparison with TaxoCapsNet.
Includes traditional ML models and deep learning architectures.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, 
    LSTM, Bidirectional, MultiHeadAttention, LayerNormalization,
    Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BaselineModelFactory:
    """Factory class for creating baseline models."""
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize baseline model factory.
        
        Args:
            config: Configuration dictionary
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.baseline_config = config.get('baseline_models', {})
        
    def create_regularized_random_forest(self) -> RandomForestClassifier:
        """
        Create Random Forest with strong regularization for small datasets.
        
        Returns:
            Configured RandomForestClassifier
        """
        rf_config = self.baseline_config.get('random_forest', {})
        
        if not rf_config.get('enabled', True):
            raise ValueError("Random Forest model is disabled in config")
        
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 50),
            max_depth=rf_config.get('max_depth', 3),
            min_samples_split=rf_config.get('min_samples_split', 2),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            max_features=rf_config.get('max_features', 'sqrt'),
            random_state=self.random_state,
            class_weight=rf_config.get('class_weight', 'balanced')
        )
        
        logger.info("Created Random Forest with regularization")
        return model
    
    def create_logistic_regression(self) -> LogisticRegression:
        """
        Create Logistic Regression with L1/L2 regularization.
        
        Returns:
            Configured LogisticRegression
        """
        lr_config = self.baseline_config.get('logistic_regression', {})
        
        if not lr_config.get('enabled', True):
            raise ValueError("Logistic Regression model is disabled in config")
        
        model = LogisticRegression(
            C=lr_config.get('C', 0.1),
            penalty=lr_config.get('penalty', 'elasticnet'),
            l1_ratio=lr_config.get('l1_ratio', 0.5),
            solver=lr_config.get('solver', 'saga'),
            max_iter=lr_config.get('max_iter', 1000),
            random_state=self.random_state,
            class_weight=lr_config.get('class_weight', 'balanced')
        )
        
        logger.info("Created Logistic Regression with elastic net regularization")
        return model
    
    def create_regularized_xgboost(self) -> xgb.XGBClassifier:
        """
        Create XGBoost with strong regularization.
        
        Returns:
            Configured XGBClassifier
        """
        xgb_config = self.baseline_config.get('xgboost', {})
        
        if not xgb_config.get('enabled', True):
            raise ValueError("XGBoost model is disabled in config")
        
        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 50),
            max_depth=xgb_config.get('max_depth', 2),
            learning_rate=xgb_config.get('learning_rate', 0.01),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.6),
            reg_alpha=xgb_config.get('reg_alpha', 1.0),
            reg_lambda=xgb_config.get('reg_lambda', 1.0),
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        logger.info("Created XGBoost with strong regularization")
        return model
    
    def create_simple_nn(self, input_dim: int, num_classes: int) -> Sequential:
        """
        Create simple neural network to prevent overfitting.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        nn_config = self.baseline_config.get('simple_nn', {})
        
        if not nn_config.get('enabled', True):
            raise ValueError("Simple NN model is disabled in config")
        
        hidden_sizes = nn_config.get('hidden_sizes', [32, 16])
        dropout_rate = nn_config.get('dropout_rate', 0.5)
        
        model = Sequential([
            Dense(hidden_sizes[0], activation='relu', input_shape=(input_dim,)),
            Dropout(dropout_rate),
            Dense(hidden_sizes[1], activation='relu'),
            Dropout(dropout_rate),
            Dense(1 if num_classes == 2 else num_classes, 
                  activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created simple NN: {hidden_sizes} -> {num_classes}")
        return model
    
    def create_cnn(self, input_dim: int, num_classes: int) -> Sequential:
        """
        Create 1D Convolutional Neural Network.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        cnn_config = self.baseline_config.get('cnn', {})
        
        if not cnn_config.get('enabled', True):
            raise ValueError("CNN model is disabled in config")
        
        filters = cnn_config.get('filters', [64, 128])
        kernel_size = cnn_config.get('kernel_size', 3)
        dropout_rate = cnn_config.get('dropout_rate', 0.3)
        
        model = Sequential([
            Reshape((input_dim, 1), input_shape=(input_dim,)),
            Conv1D(filters[0], kernel_size=kernel_size, activation='relu', padding='same'),
            Conv1D(filters[1], kernel_size=kernel_size, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1 if num_classes == 2 else num_classes,
                  activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created CNN: filters={filters}, kernel_size={kernel_size}")
        return model
    
    def create_bilstm(self, input_dim: int, num_classes: int) -> Sequential:
        """
        Create Bidirectional LSTM model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        bilstm_config = self.baseline_config.get('bilstm', {})
        
        if not bilstm_config.get('enabled', True):
            raise ValueError("BiLSTM model is disabled in config")
        
        lstm_units = bilstm_config.get('lstm_units', [256, 128])
        dropout_rate = bilstm_config.get('dropout_rate', 0.2)
        recurrent_dropout = bilstm_config.get('recurrent_dropout', 0.2)
        
        # Reshape input for LSTM (sequence_length, features)
        seq_length = 2
        feature_dim = input_dim // seq_length
        
        model = Sequential([
            Reshape((seq_length, feature_dim), input_shape=(input_dim,)),
            Bidirectional(LSTM(lstm_units[0], return_sequences=True, 
                              dropout=dropout_rate, recurrent_dropout=recurrent_dropout)),
            Bidirectional(LSTM(lstm_units[1], return_sequences=False,
                              dropout=dropout_rate, recurrent_dropout=recurrent_dropout)),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(1 if num_classes == 2 else num_classes,
                  activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created BiLSTM: units={lstm_units}, seq_length={seq_length}")
        return model
    
    def create_transformer(self, input_dim: int, num_classes: int) -> Model:
        """
        Create Transformer-based model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        transformer_config = self.baseline_config.get('transformer', {})
        
        if not transformer_config.get('enabled', True):
            raise ValueError("Transformer model is disabled in config")
        
        num_heads = transformer_config.get('num_heads', 4)
        head_size = transformer_config.get('head_size', 64)
        ff_dim = transformer_config.get('ff_dim', 512)
        num_layers = transformer_config.get('num_layers', 2)
        dropout_rate = transformer_config.get('dropout_rate', 0.1)
        
        # Reshape for transformer
        seq_length = 2
        d_model = input_dim // seq_length
        
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
            # Multi-head attention
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
            x = Dropout(dropout)(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            res = x + inputs
            
            # Feed forward network
            x = Dense(ff_dim, activation="relu")(res)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            return x + res
        
        # Build transformer model
        inputs = Input(shape=(input_dim,))
        x = Reshape((seq_length, d_model))(inputs)
        
        # Stack transformer encoder layers
        for _ in range(num_layers):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
        
        # Global pooling and dense layers
        x = GlobalMaxPooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1 if num_classes == 2 else num_classes,
                       activation='sigmoid' if num_classes == 2 else 'softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created Transformer: {num_layers} layers, {num_heads} heads")
        return model
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get list of available baseline models.
        
        Returns:
            Dictionary mapping model names to their types
        """
        models = {}
        
        if self.baseline_config.get('random_forest', {}).get('enabled', True):
            models['Random_Forest'] = 'sklearn'
        if self.baseline_config.get('logistic_regression', {}).get('enabled', True):
            models['Logistic_Regression'] = 'sklearn'
        if self.baseline_config.get('xgboost', {}).get('enabled', True):
            models['XGBoost'] = 'sklearn'
        if self.baseline_config.get('simple_nn', {}).get('enabled', True):
            models['Simple_NN'] = 'keras'
        if self.baseline_config.get('cnn', {}).get('enabled', True):
            models['CNN'] = 'keras'
        if self.baseline_config.get('bilstm', {}).get('enabled', True):
            models['BiLSTM'] = 'keras'
        if self.baseline_config.get('transformer', {}).get('enabled', True):
            models['Transformer'] = 'keras'
        
        return models
    
    def create_model(self, model_name: str, input_dim: int, num_classes: int):
        """
        Create a model by name.
        
        Args:
            model_name: Name of the model to create
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Created model instance
        """
        model_creators = {
            'Random_Forest': self.create_regularized_random_forest,
            'Logistic_Regression': self.create_logistic_regression,
            'XGBoost': self.create_regularized_xgboost,
            'Simple_NN': lambda: self.create_simple_nn(input_dim, num_classes),
            'CNN': lambda: self.create_cnn(input_dim, num_classes),
            'BiLSTM': lambda: self.create_bilstm(input_dim, num_classes),
            'Transformer': lambda: self.create_transformer(input_dim, num_classes)
        }
        
        if model_name not in model_creators:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_creators[model_name]()


def build_baseline_models(config: Dict[str, Any], input_dim: int, num_classes: int) -> Dict[str, Any]:
    """
    Build all enabled baseline models.
    
    Args:
        config: Configuration dictionary
        input_dim: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Dictionary mapping model names to model instances
    """
    factory = BaselineModelFactory(config)
    available_models = factory.get_available_models()
    
    models = {}
    for model_name, model_type in available_models.items():
        try:
            model = factory.create_model(model_name, input_dim, num_classes)
            models[model_name] = {
                'model': model,
                'type': model_type
            }
            logger.info(f"Created {model_name} ({model_type})")
        except Exception as e:
            logger.error(f"Failed to create {model_name}: {e}")
    
    return models


if __name__ == "__main__":
    # Test baseline model creation
    print("Testing baseline model creation...")
    
    # Sample config
    config = {
        'baseline_models': {
            'random_forest': {'enabled': True},
            'logistic_regression': {'enabled': True},
            'xgboost': {'enabled': True},
            'simple_nn': {'enabled': True},
            'cnn': {'enabled': False},  # Disable for testing
            'bilstm': {'enabled': True},
            'transformer': {'enabled': True}
        }
    }
    
    # Create models
    models = build_baseline_models(config, input_dim=1000, num_classes=2)
    
    print(f"Created {len(models)} models:")
    for name, model_info in models.items():
        print(f"  - {name}: {model_info['type']}")
    
    print("Testing complete!")
