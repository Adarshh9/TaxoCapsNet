"""
TaxoCapsNet Model Architecture
=============================

Implementation of the main TaxoCapsNet model with taxonomy-aware capsule networks.
Includes capsule layer implementation and multi-input architecture.

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, Lambda, Reshape
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CapsuleLayer(layers.Layer):
    """
    Capsule Layer with Dynamic Routing-by-Agreement.
    
    Implements the dynamic routing algorithm to determine how lower-level 
    capsules contribute to higher-level capsules based on agreement.
    """
    
    def __init__(self, num_capsules: int, dim_capsule: int, routings: int = 3,
                 kernel_initializer: str = 'glorot_uniform', **kwargs):
        """
        Initialize Capsule Layer.
        
        Args:
            num_capsules: Number of capsules in this layer
            dim_capsule: Dimension of each capsule
            routings: Number of routing iterations
            kernel_initializer: Weight initialization method
        """
        super(CapsuleLayer, self).__init__(**kwargs)
        
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        """Build the layer weights."""
        # Input shape: (batch_size, input_num_capsules, input_dim_capsule)
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        # Weight matrix for transformation: W_ij
        self.W = self.add_weight(
            name='W',
            shape=(self.input_num_capsules, self.num_capsules, 
                   self.input_dim_capsule, self.dim_capsule),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        super(CapsuleLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Dynamic routing algorithm.
        
        Args:
            inputs: Input tensor (batch_size, input_num_capsules, input_dim_capsule)
            
        Returns:
            Output capsules (batch_size, num_capsules, dim_capsule)
        """
        # Expand inputs to match weight dimensions
        inputs_expand = tf.expand_dims(inputs, 2)  # (batch_size, input_num_capsules, 1, input_dim_capsule)
        inputs_expand = tf.expand_dims(inputs_expand, 4)  # Add dimension for matrix multiplication
        
        # Transform inputs by weight matrix
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])
        u_hat = tf.reduce_sum(inputs_tiled * self.W, axis=3)  # (batch_size, input_num_capsules, num_capsules, dim_capsule)
        
        # Initialize coupling coefficients
        batch_size = tf.shape(inputs)[0]
        b_ij = tf.zeros((batch_size, self.input_num_capsules, self.num_capsules))
        
        # Dynamic routing iterations
        for i in range(self.routings):
            # Softmax over coupling coefficients
            c_ij = tf.nn.softmax(b_ij, axis=2)  # (batch_size, input_num_capsules, num_capsules)
            
            # Weighted sum of prediction vectors
            c_ij_expand = tf.expand_dims(c_ij, -1)  # (batch_size, input_num_capsules, num_capsules, 1)
            s_j = tf.reduce_sum(c_ij_expand * u_hat, axis=1)  # (batch_size, num_capsules, dim_capsule)
            
            # Squashing non-linearity
            v_j = self.squash(s_j)
            
            # Update coupling coefficients (except for last iteration)
            if i < self.routings - 1:
                # Agreement: a_ij = u_hat_j · v_j
                v_j_expand = tf.expand_dims(v_j, 1)  # (batch_size, 1, num_capsules, dim_capsule)
                agreement = tf.reduce_sum(u_hat * v_j_expand, axis=-1)  # (batch_size, input_num_capsules, num_capsules)
                b_ij = b_ij + agreement
        
        return v_j
    
    def squash(self, vectors):
        """
        Squashing function to ensure capsule output magnitude is between 0 and 1.
        
        Args:
            vectors: Input vectors to squash
            
        Returns:
            Squashed vectors
        """
        vec_squared_norm = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
        return scalar_factor * vectors
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return (input_shape[0], self.num_capsules, self.dim_capsule)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = {
            'num_capsules': self.num_capsules,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
            'kernel_initializer': self.kernel_initializer
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def length_layer(inputs):
    """
    Calculate the length of capsule vectors.
    Used for classification - the length represents the probability.
    """
    return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))


class TaxoCapsNet:
    """
    Main TaxoCapsNet model implementation with taxonomy-aware architecture.
    
    Combines multiple phylum-specific input streams with capsule networks
    for hierarchical microbiome classification.
    """
    
    def __init__(self, input_shapes: List[int], taxonomy_map: Dict[str, List[str]], 
                 config: Dict[str, Any]):
        """
        Initialize TaxoCapsNet model.
        
        Args:
            input_shapes: List of input dimensions for each phylum group
            taxonomy_map: Mapping of phylum names to OTU lists  
            config: Model configuration dictionary
        """
        self.input_shapes = input_shapes
        self.taxonomy_map = taxonomy_map
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
        
        logger.info(f"TaxoCapsNet initialized with {len(input_shapes)} phylum groups")
        logger.info(f"Primary capsules: {self.num_primary_capsules}, dim: {self.primary_capsule_dim}")
        logger.info(f"Class capsules: {self.num_class_capsules}, dim: {self.class_capsule_dim}")
        
    def build_model(self) -> Model:
        """
        Build the complete TaxoCapsNet model architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building TaxoCapsNet model architecture...")
        
        # Create separate inputs for each phylum group
        inputs = []
        phylum_outputs = []
        
        phylum_names = list(self.taxonomy_map.keys())
        if len(phylum_names) < len(self.input_shapes):
            # Add Unknown phylum if we have more shapes than named phyla
            phylum_names.append("Unknown_Phylum")
        
        for i, (input_shape, phylum_name) in enumerate(zip(self.input_shapes, phylum_names)):
            # Input layer for this phylum
            phylum_input = Input(shape=(input_shape,), name=f'input_{phylum_name}')
            inputs.append(phylum_input)
            
            # Dense layers for feature extraction
            x = Dense(64, activation='relu', name=f'dense1_{phylum_name}')(phylum_input)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(32, activation='relu', name=f'dense2_{phylum_name}')(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Reshape for primary capsules
            # Each phylum contributes primary capsules
            primary_caps_output = Dense(self.num_primary_capsules * self.primary_capsule_dim, 
                                      activation='relu', 
                                      name=f'primary_caps_{phylum_name}')(x)
            
            # Reshape to capsule format
            primary_caps_reshaped = Reshape((self.num_primary_capsules, self.primary_capsule_dim),
                                          name=f'primary_reshape_{phylum_name}')(primary_caps_output)
            
            phylum_outputs.append(primary_caps_reshaped)
        
        # Concatenate all phylum capsules
        if len(phylum_outputs) > 1:
            combined_capsules = layers.Concatenate(axis=1, name='combine_phyla')(phylum_outputs)
        else:
            combined_capsules = phylum_outputs[0]
        
        # Class capsules with dynamic routing
        class_capsules = CapsuleLayer(
            num_capsules=self.num_class_capsules,
            dim_capsule=self.class_capsule_dim,
            routings=self.routing_iterations,
            name='class_capsules'
        )(combined_capsules)
        
        # Output layer - calculate capsule lengths for classification
        output_lengths = Lambda(length_layer, name='output_lengths')(class_capsules)
        
        # For binary classification, use sigmoid activation on single output
        if self.num_class_capsules == 2:
            # Take the length of the positive class capsule
            predictions = Lambda(lambda x: x[:, 1:2], name='predictions')(output_lengths)
        else:
            # Multi-class: use softmax
            predictions = layers.Activation('softmax', name='predictions')(output_lengths)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=predictions, name='TaxoCapsNet')
        
        logger.info("TaxoCapsNet model built successfully")
        self._print_model_summary()
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.num_class_capsules == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy' 
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with learning rate: {learning_rate}")
    
    def _print_model_summary(self):
        """Print detailed model summary."""
        if self.model:
            logger.info("\\n" + "="*60)
            logger.info("TaxoCapsNet Model Summary")
            logger.info("="*60)
            
            # Count parameters
            total_params = self.model.count_params()
            logger.info(f"Total parameters: {total_params:,}")
            
            # Print input shapes
            logger.info(f"Input shapes: {self.input_shapes}")
            logger.info(f"Number of phylum groups: {len(self.input_shapes)}")
            
            # Print architecture details
            logger.info(f"Primary capsules per phylum: {self.num_primary_capsules}")
            logger.info(f"Primary capsule dimension: {self.primary_capsule_dim}")
            logger.info(f"Class capsules: {self.num_class_capsules}")
            logger.info(f"Class capsule dimension: {self.class_capsule_dim}")
            logger.info(f"Routing iterations: {self.routing_iterations}")
            
            logger.info("="*60)
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model
    
    def save(self, filepath: str):
        """Save the model to file."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        # Save the full model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        # Custom objects needed for loading
        custom_objects = {'CapsuleLayer': CapsuleLayer, 'length_layer': length_layer}
        
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"Model loaded from {filepath}")
        
        return self.model


def build_taxocapsnet_model(input_shapes: List[int], taxonomy_map: Dict[str, List[str]], 
                           config: Dict[str, Any]) -> Model:
    """
    Convenience function to build TaxoCapsNet model.
    
    Args:
        input_shapes: List of input dimensions for each phylum group
        taxonomy_map: Mapping of phylum names to OTU lists
        config: Model configuration dictionary
        
    Returns:
        Compiled TaxoCapsNet model
    """
    # Create TaxoCapsNet instance
    taxocapsnet = TaxoCapsNet(input_shapes, taxonomy_map, config)
    
    # Build and compile model
    model = taxocapsnet.build_model()
    
    # Get learning rate from config
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    taxocapsnet.compile_model(learning_rate)
    
    return model
