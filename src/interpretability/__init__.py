# Interpretability module for TaxoCapsNet
from .taxoshap import TaxoSHAPExplainer, create_taxoshap_explainer
from .shap_utils import (
    generate_shap_plots,
    create_feature_importance_plot,
    create_phylum_impact_plot,
    generate_interpretation_report
)

__all__ = [
    'TaxoSHAPExplainer',
    'create_taxoshap_explainer', 
    'generate_shap_plots',
    'create_feature_importance_plot',
    'create_phylum_impact_plot',
    'generate_interpretation_report'
]