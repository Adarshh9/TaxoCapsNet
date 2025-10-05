"""
TaxoCapsNet Package - Updated with SHAP Interpretability
========================================================

A taxonomy-aware capsule network for autism prediction from gut microbiome data.

This package provides:
- TaxoCapsNet model implementation
- Comprehensive baseline model comparisons  
- Ablation studies and statistical analysis
- Data preprocessing for microbiome data
- Visualization and reporting tools
- SHAP-based interpretability analysis (TaxoSHAP)
- Flask API server with ngrok support

"""

__version__ = "1.0.0"
__author__ = "Adarsh Kesharwani"
__email__ = "akesherwani900@gmail.com"

# Import main components
from src.models.taxocapsnet import TaxoCapsNet, CapsuleLayer
from src.training.trainer import TaxoCapsNetTrainer, BaselineModelTrainer
from src.training.evaluation import AblationStudyRunner
from src.utils.taxonomy import generate_taxonomy_map
from src.utils.metrics import calculate_comprehensive_metrics_with_ci
from src.data.preprocessing import load_and_preprocess_data, prepare_taxonomy_data
from src.visualization.plots import ModelComparisonVisualizer

# Import interpretability components
from src.interpretability.taxoshap import TaxoSHAPExplainer, create_taxoshap_explainer
from src.interpretability.shap_utils import (
    generate_shap_plots, 
    create_feature_importance_plot,
    generate_interpretation_report
)

__all__ = [
    # Core Model Components
    'TaxoCapsNet',
    'CapsuleLayer', 
    'TaxoCapsNetTrainer',
    'BaselineModelTrainer',
    'AblationStudyRunner',
    
    # Data Processing
    'generate_taxonomy_map',
    'calculate_comprehensive_metrics_with_ci',
    'load_and_preprocess_data',
    'prepare_taxonomy_data',
    
    # Visualization
    'ModelComparisonVisualizer',
    
    # Interpretability (SHAP)
    'TaxoSHAPExplainer',
    'create_taxoshap_explainer',
    'generate_shap_plots',
    'create_feature_importance_plot', 
    'generate_interpretation_report'
]
