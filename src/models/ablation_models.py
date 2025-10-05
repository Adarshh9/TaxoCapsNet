"""
Ablation Study Model Architectures
==================================

Implementation of ablation models for testing different components of TaxoCapsNet:
- TaxoDense: Dense networks with taxonomy grouping
- RandomCapsNet: Capsules with random OTU grouping  
- FlatCapsNet: Capsules without hierarchical structure

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Lambda, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from typing import List, Dict, Any
import logging

# Import CapsuleLayer from main model
from src.models.taxocapsnet_model import CapsuleLayer, length_layer

logger = logging.getLogger(__name__)


class TaxoDenseModel:
    """
    TaxoDense: Dense networks with taxonomy grouping.
    
    This model tests whether the biological taxonomy grouping provides
    benefits even with traditional dense layers instead of capsules.
    """
    
    def __init__(self, input_shapes: List[int], taxonomy_map: Dict[str, List[str]], 
                 config: Dict[str, Any]):
        """
        Initialize TaxoDense model.
        
        Args:
            input_shapes: List of input dimensions for each phylum group
            taxonomy_map: Mapping of phylum names to OTU lists
            config: Model configuration dictionary
        """
        self.input_shapes = input_shapes
        self.taxonomy_map = taxonomy_map
        self.config = config
        self.model = None
        
        # Extract model parameters
        dense_config = config.get('model', {}).get('dense', {})
        self.hidden_layers = dense_config.get('hidden_layers', [64, 32, 16])
        self.dropout_rate = dense_config.get('dropout_rate', 0.3)
        
        logger.info(f"TaxoDense initialized with {len(input_shapes)} phylum groups")
        logger.info(f"Hidden layers: {self.hidden_layers}")
    
    def build_model(self) -> Model:
        """
        Build TaxoDense model with taxonomy-aware dense networks.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building TaxoDense model architecture...")
        
        # Create separate inputs for each phylum group
        inputs = []
        phylum_outputs = []
        
        phylum_names = list(self.taxonomy_map.keys())
        if len(phylum_names) < len(self.input_shapes):
            phylum_names.append("Unknown_Phylum")
        
        for i, (input_shape, phylum_name) in enumerate(zip(self.input_shapes, phylum_names)):
            # Input layer for this phylum
            phylum_input = Input(shape=(input_shape,), name=f'input_{phylum_name}')
            inputs.append(phylum_input)
            
            # Dense layers for each phylum (similar structure to TaxoCapsNet)
            x = phylum_input
            for j, hidden_size in enumerate(self.hidden_layers):
                x = Dense(hidden_size, activation='relu', 
                         name=f'dense_{j+1}_{phylum_name}')(x)
                x = Dropout(self.dropout_rate)(x)
            
            phylum_outputs.append(x)
        
        # Concatenate all phylum outputs
        if len(phylum_outputs) > 1:
            combined_features = Concatenate(name='combine_phyla')(phylum_outputs)
        else:
            combined_features = phylum_outputs[0]
        
        # Final classification layers
        x = Dense(32, activation='relu', name='final_dense_1')(combined_features)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(16, activation='relu', name='final_dense_2')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=predictions, name='TaxoDense')
        
        logger.info("TaxoDense model built successfully")
        self._print_model_summary()
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"TaxoDense model compiled with learning rate: {learning_rate}")
    
    def _print_model_summary(self):
        """Print model summary."""
        if self.model:
            total_params = self.model.count_params()
            logger.info(f"TaxoDense - Total parameters: {total_params:,}")
            logger.info(f"Architecture: Dense layers with taxonomy grouping")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model


class RandomCapsNetModel:
    """
    RandomCapsNet: Capsules with random OTU grouping.
    
    This model tests whether the biological taxonomy is important by
    using the same capsule architecture but with random OTU groupings.
    """
    
    def __init__(self, input_shapes: List[int], random_taxonomy: Dict[str, List[str]], 
                 config: Dict[str, Any]):
        """
        Initialize RandomCapsNet model.
        
        Args:
            input_shapes: List of input dimensions for each random group
            random_taxonomy: Random grouping of OTUs
            config: Model configuration dictionary
        """
        self.input_shapes = input_shapes
        self.random_taxonomy = random_taxonomy
        self.config = config
        self.model = None
        
        # Extract model parameters from config (same as TaxoCapsNet)
        model_config = config.get('model', {}).get('taxocapsnet', {})
        self.num_primary_capsules = model_config.get('num_primary_capsules', 8)
        self.primary_capsule_dim = model_config.get('primary_capsule_dim', 16)
        self.num_class_capsules = model_config.get('num_class_capsules', 2)
        self.class_capsule_dim = model_config.get('class_capsule_dim', 16)
        self.routing_iterations = model_config.get('routing_iterations', 3)
        
        dense_config = config.get('model', {}).get('dense', {})
        self.dropout_rate = dense_config.get('dropout_rate', 0.3)
        
        logger.info(f"RandomCapsNet initialized with {len(input_shapes)} random groups")
    
    def build_model(self) -> Model:
        """
        Build RandomCapsNet with identical architecture to TaxoCapsNet.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building RandomCapsNet model architecture...")
        
        # Create separate inputs for each random group
        inputs = []
        group_outputs = []
        
        group_names = list(self.random_taxonomy.keys())
        
        for i, (input_shape, group_name) in enumerate(zip(self.input_shapes, group_names)):
            # Input layer for this random group
            group_input = Input(shape=(input_shape,), name=f'input_{group_name}')
            inputs.append(group_input)
            
            # Dense layers for feature extraction (same as TaxoCapsNet)
            x = Dense(64, activation='relu', name=f'dense1_{group_name}')(group_input)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(32, activation='relu', name=f'dense2_{group_name}')(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Primary capsules (same structure as TaxoCapsNet)
            primary_caps_output = Dense(self.num_primary_capsules * self.primary_capsule_dim, 
                                      activation='relu', 
                                      name=f'primary_caps_{group_name}')(x)
            
            # Reshape to capsule format
            primary_caps_reshaped = Reshape((self.num_primary_capsules, self.primary_capsule_dim),
                                          name=f'primary_reshape_{group_name}')(primary_caps_output)
            
            group_outputs.append(primary_caps_reshaped)
        
        # Concatenate all group capsules
        if len(group_outputs) > 1:
            combined_capsules = layers.Concatenate(axis=1, name='combine_groups')(group_outputs)
        else:
            combined_capsules = group_outputs[0]
        
        # Class capsules with dynamic routing (identical to TaxoCapsNet)
        class_capsules = CapsuleLayer(
            num_capsules=self.num_class_capsules,
            dim_capsule=self.class_capsule_dim,
            routings=self.routing_iterations,
            name='class_capsules'
        )(combined_capsules)
        
        # Output layer - calculate capsule lengths for classification
        output_lengths = Lambda(length_layer, name='output_lengths')(class_capsules)
        
        # Take the length of the positive class capsule
        predictions = Lambda(lambda x: x[:, 1:2], name='predictions')(output_lengths)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=predictions, name='RandomCapsNet')
        
        logger.info("RandomCapsNet model built successfully")
        self._print_model_summary()
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"RandomCapsNet model compiled with learning rate: {learning_rate}")
    
    def _print_model_summary(self):
        """Print model summary."""
        if self.model:
            total_params = self.model.count_params()
            logger.info(f"RandomCapsNet - Total parameters: {total_params:,}")
            logger.info(f"Architecture: Capsules with random grouping")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model


class FlatCapsNetModel:
    """
    FlatCapsNet: Capsules without hierarchical structure.
    
    This model tests whether the hierarchical multi-input design is important
    by using capsules with all features as a single flat input.
    """
    
    def __init__(self, total_features: int, config: Dict[str, Any]):
        """
        Initialize FlatCapsNet model.
        
        Args:
            total_features: Total number of input features
            config: Model configuration dictionary
        """
        self.total_features = total_features
        self.config = config
        self.model = None
        
        # Extract model parameters from config
        model_config = config.get('model', {}).get('taxocapsnet', {})
        self.num_primary_capsules = model_config.get('num_primary_capsules', 8)
        self.primary_capsule_dim = model_config.get('primary_capsule_dim', 16)
        self.num_class_capsules = model_config.get('num_class_capsules', 2)
        self.class_capsule_dim = model_config.get('class_capsule_dim', 16)
        self.routing_iterations = model_config.get('routing_iterations', 3)
        
        dense_config = config.get('model', {}).get('dense', {})
        self.dropout_rate = dense_config.get('dropout_rate', 0.3)
        
        logger.info(f"FlatCapsNet initialized with {total_features} total features")
    
    def build_model(self) -> Model:
        """
        Build FlatCapsNet with single flat input.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building FlatCapsNet model architecture...")
        
        # Single flat input (no hierarchy)
        inputs = Input(shape=(self.total_features,), name='flat_input')
        
        # Dense layers for feature extraction
        x = Dense(128, activation='relu', name='dense1')(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu', name='dense2')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Create multiple primary capsule groups from flat input
        # Use same total number of capsules as hierarchical version
        total_primary_capsules = self.num_primary_capsules * 4  # Approximate number from hierarchy
        
        primary_caps_output = Dense(total_primary_capsules * self.primary_capsule_dim, 
                                  activation='relu', 
                                  name='primary_caps_flat')(x)
        
        # Reshape to capsule format
        primary_caps_reshaped = Reshape((total_primary_capsules, self.primary_capsule_dim),
                                      name='primary_reshape_flat')(primary_caps_output)
        
        # Class capsules with dynamic routing
        class_capsules = CapsuleLayer(
            num_capsules=self.num_class_capsules,
            dim_capsule=self.class_capsule_dim,
            routings=self.routing_iterations,
            name='class_capsules'
        )(primary_caps_reshaped)
        
        # Output layer - calculate capsule lengths for classification
        output_lengths = Lambda(length_layer, name='output_lengths')(class_capsules)
        
        # Take the length of the positive class capsule
        predictions = Lambda(lambda x: x[:, 1:2], name='predictions')(output_lengths)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=predictions, name='FlatCapsNet')
        
        logger.info("FlatCapsNet model built successfully")
        self._print_model_summary()
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"FlatCapsNet model compiled with learning rate: {learning_rate}")
    
    def _print_model_summary(self):
        """Print model summary."""
        if self.model:
            total_params = self.model.count_params()
            logger.info(f"FlatCapsNet - Total parameters: {total_params:,}")
            logger.info(f"Architecture: Capsules with flat input (no hierarchy)")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model


def build_ablation_models(input_shapes: List[int], taxonomy_map: Dict[str, List[str]], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build all ablation models for comparison.
    
    Args:
        input_shapes: List of input dimensions for each phylum group
        taxonomy_map: Mapping of phylum names to OTU lists
        config: Model configuration dictionary
        
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {}
    
    ablation_config = config.get('ablation', {}).get('models', {})
    
    # TaxoDense model
    if ablation_config.get('taxo_dense', {}).get('enabled', True):
        models['TaxoDense'] = TaxoDenseModel(input_shapes, taxonomy_map, config)
        logger.info("Created TaxoDense model")
    
    # RandomCapsNet model
    if ablation_config.get('random_capsnet', {}).get('enabled', True):
        from src.utils.taxonomy import create_random_taxonomy_map
        total_features = sum(input_shapes)
        random_taxonomy = create_random_taxonomy_map(total_features, seed=42)
        models['RandomCapsNet'] = RandomCapsNetModel(input_shapes, random_taxonomy, config)
        logger.info("Created RandomCapsNet model")
    
    # FlatCapsNet model
    if ablation_config.get('flat_capsnet', {}).get('enabled', True):
        total_features = sum(input_shapes)
        models['FlatCapsNet'] = FlatCapsNetModel(total_features, config)
        logger.info("Created FlatCapsNet model")
    
    logger.info(f"Built {len(models)} ablation models: {list(models.keys())}")
    
    return models


if __name__ == "__main__":
    # Test ablation model creation
    print("Testing ablation model creation...")
    
    # Sample configuration
    config = {
        'model': {
            'taxocapsnet': {
                'num_primary_capsules': 8,
                'primary_capsule_dim': 16,
                'num_class_capsules': 2,
                'class_capsule_dim': 16,
                'routing_iterations': 3
            },
            'dense': {
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3
            }
        },
        'ablation': {
            'models': {
                'taxo_dense': {'enabled': True},
                'random_capsnet': {'enabled': True},
                'flat_capsnet': {'enabled': True}
            }
        }
    }
    
    # Sample data
    input_shapes = [200, 150, 100]  # Three phylum groups
    taxonomy_map = {
        'Firmicutes': [f'OTU{i}' for i in range(200)],
        'Proteobacteria': [f'OTU{i}' for i in range(200, 350)],
        'Bacteroidetes': [f'OTU{i}' for i in range(350, 450)]
    }
    
    # Build models
    models = build_ablation_models(input_shapes, taxonomy_map, config)
    
    print(f"Created {len(models)} ablation models:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Test model building
    for name, model_instance in models.items():
        try:
            model = model_instance.build_model()
            print(f"  ✅ {name}: {model.count_params():,} parameters")
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    print("Ablation model testing completed!")
