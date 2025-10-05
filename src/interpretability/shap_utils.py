"""
SHAP Utilities and Visualization Functions
==========================================

Utility functions for SHAP analysis visualization and interpretation.
Provides plotting functions and report generation for TaxoSHAP analysis.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_shap_plots(shap_results: Dict[str, Any], max_display: int = 15,
                       save_format: str = 'base64') -> Dict[str, str]:
    """
    Generate comprehensive SHAP visualization plots.
    
    Args:
        shap_results: Results from TaxoSHAP analysis
        max_display: Maximum number of features to display
        save_format: Format to save plots ('base64', 'png', 'svg')
        
    Returns:
        Dictionary with plot images as base64 strings or file paths
    """
    plots = {}
    
    try:
        shap_values = shap_results.get('shap_values', [])
        feature_names = shap_results.get('feature_names', [])
        
        if len(shap_values) == 0:
            logger.warning("No SHAP values available for plotting")
            return plots
        
        # 1. Feature Importance Bar Plot
        plots['feature_importance'] = create_feature_importance_plot(
            shap_values, feature_names, max_display, save_format
        )
        
        # 2. SHAP Values Distribution
        plots['shap_distribution'] = create_shap_distribution_plot(
            shap_values, save_format
        )
        
        # 3. Phylum Impact Analysis
        phylum_analysis = shap_results.get('phylum_analysis', {})
        if phylum_analysis:
            plots['phylum_impact'] = create_phylum_impact_plot(
                phylum_analysis, save_format
            )
        
        # 4. Feature Value vs SHAP Value Scatter
        sample_data = shap_results.get('sample_data', [])
        if len(sample_data) > 0:
            plots['value_vs_shap'] = create_value_vs_shap_plot(
                shap_values, sample_data, feature_names, max_display, save_format
            )
        
        # 5. Top Positive and Negative Features
        plots['positive_negative'] = create_positive_negative_plot(
            shap_values, feature_names, max_display, save_format
        )
        
        logger.info(f"Generated {len(plots)} SHAP visualization plots")
        
    except Exception as e:
        logger.error(f"SHAP plots generation failed: {e}")
    
    return plots


def create_feature_importance_plot(shap_values: np.ndarray, feature_names: List[str],
                                  max_display: int = 15, save_format: str = 'base64') -> str:
    """Create feature importance bar plot."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_idx = np.argsort(abs_shap)[-max_display:]
        
        top_values = shap_values[top_idx]
        top_names = [feature_names[i] for i in top_idx]
        
        # Truncate long feature names
        top_names_short = [name[:30] + '...' if len(name) > 30 else name for name in top_names]
        
        # Color by positive/negative
        colors = ['red' if v < 0 else 'blue' for v in top_values]
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        
        # Customize plot
        plt.yticks(range(len(top_values)), top_names_short, fontsize=10)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.title(f'Top {max_display} Features by SHAP Importance', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                    va='center', ha='left' if value >= 0 else 'right', fontsize=9)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Positive Impact (→ ASD)'),
                          Patch(facecolor='red', alpha=0.7, label='Negative Impact (→ Control)')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        return _save_plot(save_format, 'feature_importance')
        
    except Exception as e:
        logger.error(f"Feature importance plot failed: {e}")
        return ""


def create_shap_distribution_plot(shap_values: np.ndarray, save_format: str = 'base64') -> str:
    """Create SHAP values distribution plot."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(shap_values, bins=30, alpha=0.7, color='skyblue', 
                                   edgecolor='black', density=True)
        
        # Color bars by value (negative = red, positive = blue)
        for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
            bin_center = (bin_left + bin_right) / 2
            if bin_center < 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightblue')
        
        # Add statistics lines
        mean_val = np.mean(shap_values)
        median_val = np.median(shap_values)
        
        plt.axvline(x=mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        plt.axvline(x=median_val, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2,
                   label='Baseline (0)')
        
        # Customize plot
        plt.xlabel('SHAP Values', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of SHAP Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics box
        stats_text = f'Std: {np.std(shap_values):.4f}\nMin: {np.min(shap_values):.4f}\nMax: {np.max(shap_values):.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        return _save_plot(save_format, 'shap_distribution')
        
    except Exception as e:
        logger.error(f"SHAP distribution plot failed: {e}")
        return ""


def create_phylum_impact_plot(phylum_analysis: Dict[str, Any], save_format: str = 'base64') -> str:
    """Create phylum-level impact analysis plot."""
    try:
        phylum_impacts = phylum_analysis.get('phylum_impacts', {})
        if not phylum_impacts:
            return ""
        
        # Prepare data
        phylums = list(phylum_impacts.keys())
        total_impacts = [phylum_impacts[p]['total_impact'] for p in phylums]
        positive_impacts = [phylum_impacts[p]['positive_impact'] for p in phylums]
        negative_impacts = [phylum_impacts[p]['negative_impact'] for p in phylums]
        
        # Sort by absolute total impact
        sorted_data = sorted(zip(phylums, total_impacts, positive_impacts, negative_impacts),
                           key=lambda x: abs(x[1]), reverse=True)
        
        phylums, total_impacts, positive_impacts, negative_impacts = zip(*sorted_data)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Total Impact by Phylum
        colors = ['red' if impact < 0 else 'blue' for impact in total_impacts]
        bars1 = ax1.barh(range(len(phylums)), total_impacts, color=colors, alpha=0.7)
        
        ax1.set_yticks(range(len(phylums)))
        ax1.set_yticklabels(phylums, fontsize=10)
        ax1.set_xlabel('Total SHAP Impact', fontsize=12)
        ax1.set_title('Phylum-level Total Impact', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, total_impacts)):
            ax1.text(value + (0.001 if value >= 0 else -0.001), i, f'{value:.3f}', 
                    va='center', ha='left' if value >= 0 else 'right', fontsize=9)
        
        # Plot 2: Positive vs Negative Impact
        y_pos = np.arange(len(phylums))
        
        bars2_pos = ax2.barh(y_pos, positive_impacts, color='blue', alpha=0.7, 
                            label='Positive Impact')
        bars2_neg = ax2.barh(y_pos, negative_impacts, color='red', alpha=0.7,
                            label='Negative Impact')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(phylums, fontsize=10)
        ax2.set_xlabel('SHAP Impact', fontsize=12)
        ax2.set_title('Phylum-level Positive vs Negative Impact', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return _save_plot(save_format, 'phylum_impact')
        
    except Exception as e:
        logger.error(f"Phylum impact plot failed: {e}")
        return ""


def create_value_vs_shap_plot(shap_values: np.ndarray, sample_data: np.ndarray,
                             feature_names: List[str], max_display: int = 15,
                             save_format: str = 'base64') -> str:
    """Create scatter plot of feature values vs SHAP values."""
    try:
        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_idx = np.argsort(abs_shap)[-max_display:]
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        colors = ['red' if shap_values[i] < 0 else 'blue' for i in top_idx]
        scatter = plt.scatter(sample_data[top_idx], shap_values[top_idx], 
                            c=colors, alpha=0.7, s=100, edgecolors='black')
        
        # Add feature labels for top features
        for i, idx in enumerate(top_idx[-5:]):  # Label only top 5
            plt.annotate(feature_names[idx][:20] + '...' if len(feature_names[idx]) > 20 else feature_names[idx],
                        (sample_data[idx], shap_values[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        # Add trend line
        z = np.polyfit(sample_data[top_idx], shap_values[top_idx], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(np.min(sample_data[top_idx]), np.max(sample_data[top_idx]), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        # Customize plot
        plt.xlabel('Feature Value (Relative Abundance)', fontsize=12)
        plt.ylabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.title(f'Feature Values vs SHAP Impact (Top {max_display} Features)', 
                 fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Positive SHAP'),
                          Patch(facecolor='red', alpha=0.7, label='Negative SHAP')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        return _save_plot(save_format, 'value_vs_shap')
        
    except Exception as e:
        logger.error(f"Value vs SHAP plot failed: {e}")
        return ""


def create_positive_negative_plot(shap_values: np.ndarray, feature_names: List[str],
                                 max_display: int = 15, save_format: str = 'base64') -> str:
    """Create separate plots for top positive and negative features."""
    try:
        # Split positive and negative
        positive_idx = shap_values > 0
        negative_idx = shap_values < 0
        
        positive_values = shap_values[positive_idx]
        positive_names = [feature_names[i] for i, pos in enumerate(positive_idx) if pos]
        
        negative_values = shap_values[negative_idx]
        negative_names = [feature_names[i] for i, neg in enumerate(negative_idx) if neg]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Top positive features
        if len(positive_values) > 0:
            top_pos_idx = np.argsort(positive_values)[-(max_display//2):]
            top_pos_values = positive_values[top_pos_idx]
            top_pos_names = [positive_names[i] for i in top_pos_idx]
            top_pos_names_short = [name[:30] + '...' if len(name) > 30 else name for name in top_pos_names]
            
            bars1 = ax1.barh(range(len(top_pos_values)), top_pos_values, 
                            color='blue', alpha=0.7)
            ax1.set_yticks(range(len(top_pos_values)))
            ax1.set_yticklabels(top_pos_names_short, fontsize=10)
            ax1.set_xlabel('SHAP Value', fontsize=12)
            ax1.set_title(f'Top {len(top_pos_values)} Positive Features (→ ASD)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars1, top_pos_values)):
                ax1.text(value + 0.001, i, f'{value:.3f}', 
                        va='center', ha='left', fontsize=9)
        
        # Top negative features
        if len(negative_values) > 0:
            top_neg_idx = np.argsort(negative_values)[:(max_display//2)]
            top_neg_values = negative_values[top_neg_idx]
            top_neg_names = [negative_names[i] for i in top_neg_idx]
            top_neg_names_short = [name[:30] + '...' if len(name) > 30 else name for name in top_neg_names]
            
            bars2 = ax2.barh(range(len(top_neg_values)), top_neg_values, 
                            color='red', alpha=0.7)
            ax2.set_yticks(range(len(top_neg_values)))
            ax2.set_yticklabels(top_neg_names_short, fontsize=10)
            ax2.set_xlabel('SHAP Value', fontsize=12)
            ax2.set_title(f'Top {len(top_neg_values)} Negative Features (→ Control)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars2, top_neg_values)):
                ax2.text(value - 0.001, i, f'{value:.3f}', 
                        va='center', ha='right', fontsize=9)
        
        plt.tight_layout()
        
        return _save_plot(save_format, 'positive_negative')
        
    except Exception as e:
        logger.error(f"Positive/negative features plot failed: {e}")
        return ""


def generate_interpretation_report(shap_results: Dict[str, Any]) -> str:
    """Generate comprehensive interpretation report as formatted text."""
    try:
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TAXOCAPSNET INTERPRETABILITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Prediction Summary
        prediction = shap_results.get('prediction', {})
        predicted_class = prediction.get('predicted_class', 0)
        prediction_prob = prediction.get('taxocaps_prediction', 0.5)
        
        class_label = "ASD" if predicted_class == 1 else "Control"
        confidence = abs(prediction_prob - 0.5) * 2
        
        report_lines.append("PREDICTION SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Predicted Class: {class_label}")
        report_lines.append(f"Confidence: {confidence:.1%}")
        report_lines.append(f"Raw Probability: {prediction_prob:.4f}")
        report_lines.append("")
        
        # SHAP Analysis Summary
        shap_values = shap_results.get('shap_values', [])
        if len(shap_values) > 0:
            report_lines.append("FEATURE IMPORTANCE SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Features Analyzed: {len(shap_values)}")
            report_lines.append(f"Mean SHAP Value: {np.mean(shap_values):.4f}")
            report_lines.append(f"Standard Deviation: {np.std(shap_values):.4f}")
            report_lines.append(f"Range: [{np.min(shap_values):.4f}, {np.max(shap_values):.4f}]")
            report_lines.append("")
            
            # Top Features
            abs_shap = np.abs(shap_values)
            top_idx = np.argsort(abs_shap)[-10:]
            feature_names = shap_results.get('feature_names', [])
            
            report_lines.append("TOP 10 MOST IMPORTANT FEATURES")
            report_lines.append("-" * 40)
            for i, idx in enumerate(reversed(top_idx)):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                shap_val = shap_values[idx]
                impact = "→ ASD" if shap_val > 0 else "→ Control"
                report_lines.append(f"{i+1:2d}. {feature_name[:50]:<50} {shap_val:+8.4f} {impact}")
            report_lines.append("")
        
        # Phylum Analysis
        phylum_analysis = shap_results.get('phylum_analysis', {})
        phylum_impacts = phylum_analysis.get('phylum_impacts', {})
        
        if phylum_impacts:
            report_lines.append("PHYLUM-LEVEL ANALYSIS")
            report_lines.append("-" * 40)
            
            # Sort phylums by absolute impact
            sorted_phylums = sorted(phylum_impacts.items(), 
                                  key=lambda x: abs(x[1]['total_impact']), reverse=True)
            
            report_lines.append(f"{'Phylum':<20} {'Total Impact':<12} {'Positive':<10} {'Negative':<10} {'Features':<8}")
            report_lines.append("-" * 65)
            
            for phylum, impact_data in sorted_phylums[:8]:  # Top 8 phylums
                total = impact_data['total_impact']
                positive = impact_data['positive_impact'] 
                negative = impact_data['negative_impact']
                n_features = impact_data['feature_count']
                
                report_lines.append(f"{phylum[:19]:<20} {total:+11.4f} {positive:+9.4f} {negative:+9.4f} {n_features:7d}")
            
            report_lines.append("")
        
        # Biological Interpretation
        report_lines.append("BIOLOGICAL INTERPRETATION")
        report_lines.append("-" * 40)
        
        if predicted_class == 1:
            report_lines.append("• Sample shows microbial signatures consistent with ASD")
            report_lines.append("• Key taxonomic features support ASD classification")
        else:
            report_lines.append("• Sample shows typical control microbiome profile")
            report_lines.append("• Taxonomic features align with neurotypical patterns")
        
        if phylum_impacts:
            most_important = max(phylum_impacts.items(), key=lambda x: abs(x[1]['total_impact']))
            phylum_name = most_important[0]
            impact_val = most_important[1]['total_impact']
            direction = "strongly supports" if impact_val > 0 else "opposes"
            
            report_lines.append(f"• {phylum_name} {direction} the {class_label} prediction")
            report_lines.append(f"• Analysis based on comprehensive taxonomic profiling")
        
        report_lines.append("")
        report_lines.append("CONFIDENCE ASSESSMENT")
        report_lines.append("-" * 40)
        
        if confidence > 0.8:
            report_lines.append("• HIGH confidence - Strong taxonomic evidence")
        elif confidence > 0.5:
            report_lines.append("• MEDIUM confidence - Moderate taxonomic evidence")
        else:
            report_lines.append("• LOW confidence - Weak taxonomic evidence")
        
        report_lines.append(f"• Model certainty: {confidence:.1%}")
        report_lines.append(f"• Prediction reliability: {'Good' if confidence > 0.6 else 'Fair'}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    except Exception as e:
        logger.error(f"Interpretation report generation failed: {e}")
        return f"Error generating report: {str(e)}"


def _save_plot(save_format: str, plot_name: str) -> str:
    """Save plot in specified format."""
    try:
        if save_format == 'base64':
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()
            return image_base64
        
        elif save_format in ['png', 'svg', 'pdf']:
            filename = f"shap_{plot_name}.{save_format}"
            plt.savefig(filename, format=save_format, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return filename
        
        else:
            plt.close()
            return ""
            
    except Exception as e:
        logger.error(f"Plot saving failed: {e}")
        plt.close()
        return ""


if __name__ == "__main__":
    # Test SHAP utilities
    print("Testing SHAP utilities...")
    
    # Sample data for testing
    np.random.seed(42)
    sample_shap_values = np.random.normal(0, 0.1, 100)
    sample_feature_names = [f"Feature_{i}" for i in range(100)]
    
    # Test feature importance plot
    try:
        plot_b64 = create_feature_importance_plot(sample_shap_values, sample_feature_names)
        print(f"Feature importance plot generated: {len(plot_b64)} characters")
    except Exception as e:
        print(f"Feature importance plot failed: {e}")
    
    print("SHAP utilities test completed!")
