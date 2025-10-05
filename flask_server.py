"""
Flask API Server for TaxoCapsNet with SHAP Analysis
=================================================

Production-ready Flask server with TaxoSHAP interpretability analysis.
Includes optional ngrok tunneling for public access and comprehensive API endpoints.

"""

import os
import io
import base64
import traceback
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Flask and web components
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ML and visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# TaxoCapsNet imports
from src.models.taxocapsnet import TaxoCapsNet
from src.interpretability.taxoshap import TaxoSHAPExplainer
from src.utils.taxonomy import generate_taxonomy_map
from src.data.preprocessing import load_and_preprocess_data

# Optional ngrok for public tunneling
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("⚠️  pyngrok not installed. Install with: pip install pyngrok")

# Configuration
HOST = '127.0.0.1'
PORT = 5000
DEBUG_MODE = False
NGROK_AUTH_TOKEN = None  # Set your ngrok auth token here if needed

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration


class TaxoSHAPServer:
    """
    Main server class handling TaxoCapsNet model and SHAP analysis.
    
    This class manages:
    - Model loading and initialization
    - Test data management  
    - SHAP analysis computation
    - Sample predictions and interpretability
    """
    
    def __init__(self):
        """Initialize the TaxoSHAP server."""
        self.model = None
        self.X_test_list = None
        self.y_test = None
        self.phylums = None
        self.taxonomy_map = None
        self.available_samples = {}
        self.is_initialized = False
        self.shap_explainer = None
        
        logger.info("TaxoSHAP Server initialized")
    
    def initialize_with_real_data(self, model, X_test_list: List[np.ndarray], 
                                 phylums: List[str], y_test: np.ndarray = None,
                                 taxonomy_map: Dict = None, num_samples: int = 20):
        """
        Initialize server with actual test dataset.
        
        Args:
            model: Trained TaxoCapsNet model
            X_test_list: List of test data arrays (one per phylum)
            phylums: List of phylum names
            y_test: Test labels (optional)
            taxonomy_map: Taxonomy mapping (optional)
            num_samples: Number of samples to make available via API
        """
        try:
            self.model = model
            self.X_test_list = X_test_list
            self.phylums = phylums
            self.y_test = y_test
            self.taxonomy_map = taxonomy_map or {}
            
            logger.info(f"Model loaded: {type(model).__name__}")
            logger.info(f"Test data loaded: {len(X_test_list)} phylums")
            logger.info(f"Total test samples: {X_test_list[0].shape[0]}")
            
            # Create sample indices from real test data
            self._create_real_sample_indices(num_samples)
            
            # Initialize SHAP explainer
            self._initialize_shap_explainer()
            
            self.is_initialized = True
            logger.info("Server initialized successfully with real data")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise e
    
    def _create_real_sample_indices(self, num_samples: int = 20):
        """Create available samples from real test dataset indices."""
        total_test_samples = self.X_test_list[0].shape[0]
        logger.info(f"Total test samples available: {total_test_samples}")
        
        # Select random indices from actual test data
        np.random.seed(42)  # For reproducible sample selection
        if num_samples >= total_test_samples:
            selected_indices = list(range(total_test_samples))
        else:
            selected_indices = np.random.choice(
                total_test_samples, size=num_samples, replace=False
            )
        
        # Create sample metadata using real test data
        for i, test_idx in enumerate(selected_indices):
            sample_id = f"sample_{test_idx}"
            
            # Get real statistics from actual test sample
            real_stats = self._get_real_sample_statistics(test_idx)
            
            self.available_samples[sample_id] = {
                'test_index': int(test_idx),
                'sample_number': i,
                'description': f'Real test sample from index {test_idx}',
                'metadata': {
                    'test_index': int(test_idx),
                    'total_features': real_stats['total_features'],
                    'phylums_count': len(self.phylums),
                    'true_label': int(self.y_test[test_idx]) if self.y_test is not None else None,
                    'feature_distribution': real_stats['phylum_features']
                },
                'statistics': real_stats
            }
        
        logger.info(f"Created {len(self.available_samples)} real test samples")
    
    def _get_real_sample_statistics(self, test_index: int) -> Dict[str, Any]:
        """Get statistics from real test sample."""
        try:
            stats = {
                'total_features': 0,
                'phylum_features': {},
                'feature_ranges': {},
                'sample_preview': {}
            }
            
            for i, phylum_data in enumerate(self.X_test_list):
                phylum_name = self.phylums[i] if i < len(self.phylums) else f"Phylum_{i}"
                
                # Get real sample data
                real_sample_data = phylum_data[test_index]
                clean_name = str(phylum_name).replace('p__', '').replace('_', ' ').strip()
                
                stats['phylum_features'][clean_name] = int(phylum_data.shape[1])
                stats['total_features'] += int(phylum_data.shape[1])
                
                # Real feature statistics
                stats['feature_ranges'][clean_name] = {
                    'min': float(np.min(real_sample_data)),
                    'max': float(np.max(real_sample_data)),
                    'mean': float(np.mean(real_sample_data)),
                    'std': float(np.std(real_sample_data)),
                    'nonzero_count': int(np.count_nonzero(real_sample_data))
                }
                
                # Sample preview (first few features)
                stats['sample_preview'][clean_name] = real_sample_data[:5].tolist()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting sample statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer with model and background data."""
        try:
            if self.X_test_list and len(self.X_test_list) > 0:
                # Use subset of test data as background
                background_size = min(50, self.X_test_list[0].shape[0] // 4)
                background_indices = np.random.choice(
                    self.X_test_list[0].shape[0], 
                    size=background_size, 
                    replace=False
                )
                
                background_data = []
                for phylum_data in self.X_test_list:
                    background_samples = phylum_data[background_indices]
                    background_data.append(background_samples)
                
                # Initialize TaxoSHAP explainer
                self.shap_explainer = TaxoSHAPExplainer(
                    model=self.model,
                    background_data=background_data,
                    phylum_names=self.phylums,
                    taxonomy_map=self.taxonomy_map
                )
                
                logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"SHAP explainer initialization failed: {e}")
    
    def get_real_sample_data(self, sample_id: str) -> List[np.ndarray]:
        """Get real sample data for SHAP analysis."""
        if sample_id not in self.available_samples:
            raise ValueError(f"Sample {sample_id} not found")
        
        real_test_index = self.available_samples[sample_id]['test_index']
        
        # Extract real sample data for each phylum
        real_sample_data = []
        for phylum_data in self.X_test_list:
            sample_data = phylum_data[real_test_index].reshape(1, -1)
            real_sample_data.append(sample_data)
        
        return real_sample_data
    
    def predict_on_real_sample(self, sample_id: str) -> Dict[str, Any]:
        """Make prediction on real test sample."""
        if sample_id not in self.available_samples:
            raise ValueError(f"Sample {sample_id} not found")
        
        # Get real sample data
        real_sample_data = self.get_real_sample_data(sample_id)
        real_test_index = self.available_samples[sample_id]['test_index']
        
        # Make prediction using actual model
        prediction = self.model.predict(real_sample_data, verbose=0)
        
        # Get true label for comparison
        true_label = self.y_test[real_test_index] if self.y_test is not None else None
        
        return {
            'prediction': prediction,
            'predicted_class': int(np.argmax(prediction, axis=1)[0]) if prediction.shape[1] > 1 else int(prediction[0] > 0.5),
            'prediction_probability': float(np.max(prediction, axis=1)[0]) if prediction.shape[1] > 1 else float(prediction[0]),
            'true_label': int(true_label) if true_label is not None else None,
            'test_index': real_test_index
        }


# Initialize server instance
server = TaxoSHAPServer()


# Utility functions
def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def safe_jsonify(data, status=200):
    """Safe JSON response with numpy conversion."""
    return Response(
        json.dumps(numpy_to_python(data)),
        status=status,
        mimetype='application/json'
    )


def save_plot_as_base64() -> str:
    """Save current matplotlib plot as base64 string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return image_base64


# Flask Routes
@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint."""
    return jsonify({
        'service': 'TaxoSHAP Analysis Server - Complete Edition',
        'status': 'running',
        'version': '2.0.0',
        'endpoints': {
            'health': '/health',
            'samples': '/samples',
            'sample_info': '/samples/<id>/info',
            'phylums': '/phylums',
            'model_info': '/model/info',
            'analyze': '/analyze (POST)'
        },
        'timestamp': datetime.now().isoformat(),
        'server_initialized': server.is_initialized
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'TaxoSHAP Analysis Server',
            'version': '2.0.0',
            'server_initialized': server.is_initialized,
            'model_loaded': server.model is not None,
            'samples_available': len(server.available_samples),
            'port': PORT
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/samples', methods=['GET'])
def get_available_real_samples():
    """Get all available real test samples."""
    try:
        if not server.is_initialized:
            return jsonify({
                'status': 'error',
                'message': 'Server not initialized with real test data. Call /initialize first.'
            }), 500
        
        samples_info = {}
        for sample_id, sample_data in server.available_samples.items():
            samples_info[sample_id] = {
                'test_index': sample_data['test_index'],
                'description': sample_data['description'],
                'metadata': sample_data['metadata'],
                'statistics': sample_data.get('statistics', {}),
                'is_real_data': True
            }
        
        return jsonify({
            'status': 'success',
            'message': 'Real test samples from your dataset',
            'samples': samples_info,
            'total_count': len(samples_info),
            'total_test_samples': server.X_test_list[0].shape[0] if server.X_test_list else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Get real samples failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/samples/<sample_id>/info', methods=['GET'])
def get_sample_info(sample_id):
    """Get detailed sample information."""
    try:
        if not server.is_initialized:
            return jsonify({
                'status': 'error',
                'message': 'Server not initialized'
            }), 500
        
        if sample_id not in server.available_samples:
            return jsonify({
                'status': 'error',
                'message': f'Sample {sample_id} not found'
            }), 404
        
        sample_data = server.available_samples[sample_id]
        
        info = {
            'sample_id': sample_id,
            'description': sample_data.get('description', 'No description'),
            'metadata': sample_data.get('metadata', {}),
            'statistics': sample_data.get('statistics', {}),
            'phylum_details': []
        }
        
        # Add phylum details
        for i, phylum in enumerate(server.phylums or []):
            if i < len(server.X_test_list):
                info['phylum_details'].append({
                    'phylum_name': phylum,
                    'feature_count': server.X_test_list[i].shape[1],
                    'data_shape': list(server.X_test_list[i].shape)
                })
        
        return jsonify({
            'status': 'success',
            'sample_info': info
        }), 200
        
    except Exception as e:
        logger.error(f"Get sample info failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/phylums', methods=['GET'])
def get_phylums():
    """Get phylums information."""
    try:
        if not server.is_initialized:
            # Return dummy data if not initialized
            phylums_info = [
                {'index': 0, 'name': 'Bacteroidetes', 'clean_name': 'Bacteroidetes'},
                {'index': 1, 'name': 'Firmicutes', 'clean_name': 'Firmicutes'},
                {'index': 2, 'name': 'Proteobacteria', 'clean_name': 'Proteobacteria'},
                {'index': 3, 'name': 'Actinobacteria', 'clean_name': 'Actinobacteria'}
            ]
            return jsonify({
                'status': 'success',
                'phylums': phylums_info,
                'count': len(phylums_info),
                'note': 'Using dummy data - server not fully initialized'
            }), 200
        
        phylums_info = []
        for i, phylum in enumerate(server.phylums or []):
            phylums_info.append({
                'index': i,
                'name': phylum,
                'clean_name': str(phylum).replace('p__', '').replace('_', ' ').strip()
            })
        
        return jsonify({
            'status': 'success',
            'phylums': phylums_info,
            'count': len(phylums_info)
        }), 200
        
    except Exception as e:
        logger.error(f"Get phylums failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information."""
    try:
        if not server.is_initialized or server.model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 404
        
        model_info = {
            'status': 'loaded',
            'type': str(type(server.model).__name__),
            'initialized': server.is_initialized
        }
        
        # Try to get additional model details
        try:
            if hasattr(server.model, 'count_params'):
                model_info['parameters'] = server.model.count_params()
        except Exception:
            pass
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Get model info failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze_real_sample():
    """Analyze real test sample with TaxoCapsNet SHAP analysis."""
    analysis_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting REAL sample analysis: {analysis_id}")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        sample_id = data.get('sample_id')
        method = data.get('method', 'tree')
        max_display = data.get('max_display', 15)
        n_background = data.get('n_background', 50)
        
        if not sample_id:
            return jsonify({
                'status': 'error',
                'message': 'sample_id is required'
            }), 400
        
        if not server.is_initialized:
            return jsonify({
                'status': 'error',
                'message': 'Server not initialized with real data'
            }), 500
        
        if sample_id not in server.available_samples:
            return jsonify({
                'status': 'error',
                'message': f'Sample {sample_id} not found'
            }), 404
        
        real_test_index = server.available_samples[sample_id]['test_index']
        logger.info(f"Analyzing REAL test sample at index {real_test_index}")
        
        # 1. Make prediction on real sample
        logger.info("Making prediction on real sample...")
        prediction_results = server.predict_on_real_sample(sample_id)
        
        # 2. Get real sample data for SHAP
        real_sample_data = server.get_real_sample_data(sample_id)
        
        # 3. Run SHAP analysis if explainer is available
        if server.shap_explainer:
            logger.info("Running SHAP analysis...")
            try:
                shap_results = server.shap_explainer.explain_sample(
                    sample_data=real_sample_data,
                    method=method,
                    n_background=n_background
                )
                
                # Generate visualizations
                visualizations = generate_shap_visualizations(shap_results, max_display)
                
                # Get feature importance
                top_features = get_top_features(shap_results, max_display)
                
            except Exception as shap_error:
                logger.warning(f"SHAP analysis failed: {shap_error}")
                shap_results = None
                visualizations = {}
                top_features = {'positive': [], 'negative': []}
        else:
            logger.warning("SHAP explainer not available")
            shap_results = None
            visualizations = {}
            top_features = {'positive': [], 'negative': []}
        
        # 4. Compile response
        response_data = {
            'status': 'success',
            'analysis_id': analysis_id,
            'sample_id': sample_id,
            'real_test_index': real_test_index,
            'timestamp': datetime.now().isoformat(),
            
            # Model prediction results
            'prediction_results': {
                'predicted_class': prediction_results['predicted_class'],
                'prediction_probability': prediction_results['prediction_probability'],
                'true_label': prediction_results['true_label'],
                'prediction_correct': prediction_results['predicted_class'] == prediction_results['true_label'] if prediction_results['true_label'] is not None else None,
                'raw_prediction': prediction_results['prediction'].tolist()
            },
            
            # Sample information
            'sample_info': {
                'test_index': real_test_index,
                'description': server.available_samples[sample_id]['description'],
                'metadata': server.available_samples[sample_id]['metadata'],
                'statistics': server.available_samples[sample_id]['statistics']
            },
            
            # SHAP analysis results
            'shap_results': shap_results,
            'top_features': top_features,
            'visualizations': visualizations,
            
            # Analysis metadata
            'analysis_metadata': {
                'method': method,
                'background_samples_used': n_background,
                'features_analyzed': len(server.phylums) if server.phylums else 0,
                'phylums_analyzed': len(server.phylums) if server.phylums else 0,
                'is_real_data': True
            }
        }
        
        logger.info(f"REAL sample analysis {analysis_id} completed successfully")
        logger.info(f"Prediction: {prediction_results['predicted_class']} (prob: {prediction_results['prediction_probability']:.4f})")
        logger.info(f"True label: {prediction_results['true_label']}")
        logger.info(f"Correct: {response_data['prediction_results']['prediction_correct']}")
        
        return safe_jsonify(response_data, 200)
        
    except Exception as e:
        logger.error(f"REAL sample analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat()
        }), 500


def generate_shap_visualizations(shap_results, max_display: int = 15) -> Dict[str, str]:
    """Generate SHAP visualization images as base64."""
    if not shap_results:
        return {}
    
    images = {}
    
    try:
        shap_values = shap_results.get('shap_values', [])
        feature_names = shap_results.get('feature_names', [])
        
        if len(shap_values) == 0:
            return {}
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(10, 6))
        top_idx = np.argsort(np.abs(shap_values))[-max_display:]
        top_values = shap_values[top_idx]
        top_names = [feature_names[i][:25] + '...' if len(feature_names[i]) > 25 
                    else feature_names[i] for i in top_idx]
        
        colors = ['red' if v < 0 else 'blue' for v in top_values]
        plt.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_values)), top_names, fontsize=8)
        plt.xlabel('SHAP Value')
        plt.title('Top Feature Contributions')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        images['feature_importance'] = save_plot_as_base64()
        plt.close()
        
        # 2. SHAP Distribution  
        plt.figure(figsize=(8, 5))
        plt.hist(shap_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(shap_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(shap_values):.4f}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Baseline')
        plt.xlabel('SHAP Values')
        plt.ylabel('Frequency')
        plt.title('Distribution of SHAP Values')
        plt.legend()
        plt.tight_layout()
        images['shap_distribution'] = save_plot_as_base64()
        plt.close()
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
    
    return images


def get_top_features(shap_results, max_display: int = 15) -> Dict[str, List]:
    """Extract top positive and negative features."""
    if not shap_results:
        return {'positive': [], 'negative': []}
    
    try:
        shap_values = shap_results.get('shap_values', [])
        feature_names = shap_results.get('feature_names', [])
        
        if len(shap_values) == 0:
            return {'positive': [], 'negative': []}
        
        # Sort by SHAP values
        sorted_idx = np.argsort(shap_values)
        
        # Top negative (most negative values)
        negative_idx = sorted_idx[:max_display//2]
        negative_features = []
        for idx in negative_idx:
            if shap_values[idx] < 0:
                negative_features.append({
                    'feature': feature_names[idx],
                    'shap_value': float(shap_values[idx]),
                    'importance': abs(float(shap_values[idx]))
                })
        
        # Top positive (most positive values)
        positive_idx = sorted_idx[-(max_display//2):]
        positive_features = []
        for idx in reversed(positive_idx):
            if shap_values[idx] > 0:
                positive_features.append({
                    'feature': feature_names[idx],
                    'shap_value': float(shap_values[idx]),
                    'importance': abs(float(shap_values[idx]))
                })
        
        return {
            'positive': positive_features,
            'negative': negative_features
        }
        
    except Exception as e:
        logger.error(f"Top features extraction failed: {e}")
        return {'positive': [], 'negative': []}


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


def get_free_port():
    """Get a free port for the server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_server_with_ngrok(use_ngrok: bool = False, ngrok_auth_token: str = None):
    """
    Start Flask server with optional ngrok tunneling.
    
    Args:
        use_ngrok: Whether to create ngrok tunnel
        ngrok_auth_token: ngrok auth token (optional)
    """
    global PORT
    
    if use_ngrok and not NGROK_AVAILABLE:
        logger.warning("ngrok not available. Starting without tunnel.")
        use_ngrok = False
    
    # Set ngrok auth token if provided
    if use_ngrok and ngrok_auth_token:
        try:
            ngrok.set_auth_token(ngrok_auth_token)
            logger.info("ngrok auth token set")
        except Exception as e:
            logger.warning(f"ngrok token issue: {e}")
    
    # Get free port
    PORT = get_free_port()
    
    def start_flask_server():
        """Start Flask server in background thread."""
        logger.info(f"Starting Flask on {HOST}:{PORT}")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host=HOST, port=PORT, debug=DEBUG_MODE, use_reloader=False, threaded=True)
    
    # Start Flask server
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()
    
    # Wait for Flask to start
    time.sleep(3)
    
    local_url = f"http://{HOST}:{PORT}"
    public_url = local_url
    
    # Test Flask locally
    try:
        import requests
        response = requests.get(f"{local_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("Flask server running locally")
        else:
            logger.warning(f"Flask server responding with status {response.status_code}")
    except Exception as e:
        logger.warning(f"Local Flask test failed: {e}")
    
    # Create ngrok tunnel if requested
    if use_ngrok:
        try:
            logger.info(f"Creating ngrok tunnel for port {PORT}")
            public_tunnel = ngrok.connect(PORT, bind_tls=True)
            public_url = public_tunnel.public_url
            logger.info(f"Public URL: {public_url}")
            
            # Test ngrok tunnel
            time.sleep(2)
            response = requests.get(f"{public_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("ngrok tunnel working!")
            else:
                logger.warning(f"ngrok tunnel responding with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"ngrok tunnel failed: {e}")
            public_url = local_url
    
    return local_url, public_url


def load_model_and_initialize_server(model_path: str = None, data_path: str = None):
    """
    Load TaxoCapsNet model and initialize server.
    
    Args:
        model_path: Path to saved TaxoCapsNet model
        data_path: Path to test data
    """
    try:
        logger.info("Loading TaxoCapsNet model and test data...")
        
        # This is where you would load your actual model and data
        # Example implementation:
        
        if model_path:
            from tensorflow.keras.models import load_model
            from src.models.taxocapsnet import CapsuleLayer
            
            # Load saved model with custom objects
            custom_objects = {'CapsuleLayer': CapsuleLayer}
            model = load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No model path provided. Server will start but analysis won't work.")
            return False
        
        if data_path:
            # Load test data (implement based on your data format)
            # This should return X_test_list, y_test, phylums, taxonomy_map
            pass
        else:
            logger.warning("No test data path provided.")
            return False
        
        # Initialize server with loaded data
        # server.initialize_with_real_data(model, X_test_list, phylums, y_test, taxonomy_map)
        
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


if __name__ == '__main__':
    print("🚀 TaxoCapsNet Flask Server with SHAP Analysis")
    print("=" * 60)
    
    # Configuration
    USE_NGROK = False  # Set to True to enable public access
    NGROK_TOKEN = None  # Set your ngrok auth token here
    
    print("Starting server...")
    
    # Start server
    local_url, public_url = start_server_with_ngrok(
        use_ngrok=USE_NGROK,
        ngrok_auth_token=NGROK_TOKEN
    )
    
    print("=" * 60)
    print("🎉 TAXOSHAP FLASK SERVER READY!")
    print("=" * 60)
    print(f"📍 Local URL: {local_url}")
    print(f"🌍 Public URL: {public_url}")
    print(f"🔧 Server initialized: {server.is_initialized}")
    print(f"📊 Available samples: {len(server.available_samples)}")
    print()
    print("🔗 AVAILABLE ENDPOINTS:")
    print(f"  GET  {public_url}/health")
    print(f"  GET  {public_url}/samples")  
    print(f"  GET  {public_url}/samples/<id>/info")
    print(f"  GET  {public_url}/phylums")
    print(f"  GET  {public_url}/model/info")
    print(f"  POST {public_url}/analyze")
    print()
    print("🧪 TEST COMMANDS:")
    print(f"  curl {public_url}/health")
    print(f"  curl {public_url}/samples")
    print(f'  curl -X POST {public_url}/analyze -H "Content-Type: application/json" -d \'{{"sample_id":"sample_0","method":"tree"}}\'')
    print("=" * 60)
    
    # Show active tunnels
    if USE_NGROK and NGROK_AVAILABLE:
        tunnels = ngrok.get_tunnels()
        print(f"🌐 Active ngrok tunnels: {len(tunnels)}")
        for tunnel in tunnels:
            print(f"  {tunnel.public_url} -> localhost:{tunnel.config['addr'].split(':')[-1]}")
    
    print()
    print("💡 TO INITIALIZE WITH YOUR REAL DATA:")
    print("   server.initialize_with_real_data(your_model, your_X_test_list, your_phylums, your_y_test)")
    print()
    print("🛑 Server will keep running until you interrupt the kernel")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if USE_NGROK and NGROK_AVAILABLE:
            ngrok.disconnect_all()
            ngrok.kill()
