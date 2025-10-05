"""
Visualization and Plotting Functions  
===================================

Functions for generating comprehensive plots and visualizations for model comparison,
training history, ROC curves, and ablation study results.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("Set2")


class ModelComparisonVisualizer:
    """Class for creating comprehensive model comparison visualizations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary for plot styling
        """
        self.config = config or {}
        self.viz_config = self.config.get('visualization', {})
        
        # Plot configuration
        self.figure_size = self.viz_config.get('style', {}).get('figure_size', [12, 8])
        self.dpi = self.viz_config.get('style', {}).get('dpi', 300)
        self.color_palette = self.viz_config.get('style', {}).get('color_palette', 'Set2')
        self.save_format = self.viz_config.get('style', {}).get('save_format', 'png')
        
        # Set seaborn style
        sns.set_palette(self.color_palette)
        
    def generate_comparison_plots(self, results: Dict[str, Dict], save_path: str = None) -> plt.Figure:
        """
        Generate comprehensive model comparison plots.
        
        This function recreates the plotting functionality from your original notebook
        with enhanced confidence intervals and error bars.
        
        Args:
            results: Dictionary with model results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if not results:
            logger.warning("No results to plot")
            return None
        
        model_names = list(results.keys())
        
        # Prepare data for plotting
        metrics_data = {
            'Model': model_names,
            'Val_Accuracy': [results[name].get('val_accuracy', 0) for name in model_names],
            'Val_Sensitivity': [results[name].get('val_sensitivity', 0) for name in model_names],
            'Val_Specificity': [results[name].get('val_specificity', 0) for name in model_names],
            'Val_Precision': [results[name].get('val_precision', 0) for name in model_names],
            'Val_F1': [results[name].get('val_f1', 0) for name in model_names],
            'Val_AUC': [results[name].get('val_auc', 0) for name in model_names],
            'Val_IC': [results[name].get('val_ic', 0) for name in model_names],
            'IC_CI_Lower': [results[name].get('val_ic_ci', (np.nan, np.nan))[0] for name in model_names],
            'IC_CI_Upper': [results[name].get('val_ic_ci', (np.nan, np.nan))[1] for name in model_names],
            'Training_Time': [results[name].get('training_time', 0) for name in model_names]
        }
        
        df_results = pd.DataFrame(metrics_data)
        
        # Create enhanced plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Enhanced Model Performance Comparison with Confidence Intervals', 
                    fontsize=16, fontweight='bold')
        
        # Define colors
        colors = sns.color_palette(self.color_palette, len(model_names))
        
        # Plot metrics
        metrics_to_plot = [
            ('Val_Accuracy', 'Validation Accuracy'),
            ('Val_Sensitivity', 'Validation Sensitivity'), 
            ('Val_Specificity', 'Validation Specificity'),
            ('Val_Precision', 'Validation Precision'),
            ('Val_F1', 'Validation F1-Score'),
            ('Val_AUC', 'Validation AUC'),
            ('Val_IC', 'Validation IC with CI'),
            ('Training_Time', 'Training Time (seconds)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            row = idx // 4
            col = idx % 4
            
            if metric == 'Val_IC':
                # Special handling for IC with confidence intervals
                x_pos = range(len(model_names))
                y_values = df_results[metric]
                y_errors = []
                
                for i in range(len(model_names)):
                    ci_lower = df_results['IC_CI_Lower'].iloc[i]
                    ci_upper = df_results['IC_CI_Upper'].iloc[i]
                    ic_value = y_values.iloc[i]
                    
                    if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                        error_lower = max(0, ic_value - ci_lower)
                        error_upper = max(0, ci_upper - ic_value)
                        y_errors.append([error_lower, error_upper])
                    else:
                        y_errors.append([0, 0])
                
                y_errors = np.array(y_errors).T
                
                bars = axes[row, col].bar(x_pos, y_values, color=colors, alpha=0.7)
                axes[row, col].errorbar(x_pos, y_values, yerr=y_errors, 
                                      fmt='none', color='black', capsize=3)
                axes[row, col].set_xticks(x_pos)
                axes[row, col].set_xticklabels(model_names, rotation=45, ha='right')
            else:
                # Regular bar plot for other metrics
                bars = axes[row, col].bar(df_results['Model'], df_results[metric], 
                                        color=colors, alpha=0.7)
                axes[row, col].tick_params(axis='x', rotation=45)
            
            axes[row, col].set_title(title, fontsize=12, fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(df_results[metric]):
                if not np.isnan(v):
                    axes[row, col].text(i, v + (max(df_results[metric]) * 0.01), 
                                      f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, models: Dict, X_val: np.ndarray, y_val: np.ndarray, 
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for all models.
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation labels
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        plt.figure(figsize=self.figure_size)
        
        colors = sns.color_palette(self.color_palette, len(models))
        
        for i, (model_name, model) in enumerate(models.items()):
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'predict'):
                    y_proba = model.predict(X_val, verbose=0)
                    if len(y_proba.shape) > 1:
                        y_proba = y_proba.ravel()
                else:
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_val, y_proba)
                auc = roc_auc_score(y_val, y_proba)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{model_name} (AUC={auc:.3f})')
                
            except Exception as e:
                logger.warning(f"Error plotting ROC for {model_name}: {e}")
                continue
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
            logger.info(f"ROC curves saved to {save_path}")
        
        return plt.gcf()
    
    def plot_confusion_matrices(self, models: Dict, X_val: np.ndarray, y_val: np.ndarray,
                               save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for all models.
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation labels
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        n_models = len(models)
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(models.items()):
            try:
                # Get predictions
                if hasattr(model, 'predict'):
                    if hasattr(model, 'predict_proba'):
                        y_pred = model.predict(X_val)
                    else:
                        y_proba = model.predict(X_val, verbose=0)
                        y_pred = (y_proba > 0.5).astype(int) if len(y_proba.shape) > 1 else y_proba.ravel() > 0.5
                else:
                    continue
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_val, y_pred)
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Control', 'ASD'], 
                           yticklabels=['Control', 'ASD'],
                           ax=axes[i])
                
                axes[i].set_title(f'{model_name}', fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                
            except Exception as e:
                logger.warning(f"Error plotting confusion matrix for {model_name}: {e}")
                continue
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
            logger.info(f"Confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, training_histories: Dict[str, Dict], 
                             save_path: str = None) -> plt.Figure:
        """
        Plot training history for deep learning models.
        
        Args:
            training_histories: Dictionary with training histories
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        if not training_histories:
            logger.warning("No training histories to plot")
            return None
        
        n_models = len(training_histories)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        colors = sns.color_palette(self.color_palette, 2)
        
        for i, (model_name, history) in enumerate(training_histories.items()):
            # Plot accuracy
            if 'accuracy' in history and 'val_accuracy' in history:
                axes[0, i].plot(history['accuracy'], color=colors[0], label='Training', linewidth=2)
                axes[0, i].plot(history['val_accuracy'], color=colors[1], label='Validation', linewidth=2)
                axes[0, i].set_title(f'{model_name} - Accuracy')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Accuracy')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
            
            # Plot loss
            if 'loss' in history and 'val_loss' in history:
                axes[1, i].plot(history['loss'], color=colors[0], label='Training', linewidth=2)
                axes[1, i].plot(history['val_loss'], color=colors[1], label='Validation', linewidth=2)
                axes[1, i].set_title(f'{model_name} - Loss')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Loss')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
            logger.info(f"Training history saved to {save_path}")
        
        return fig
    
    def plot_ablation_results(self, ablation_results: Dict[str, Dict], 
                             save_path: str = None) -> plt.Figure:
        """
        Plot ablation study results with error bars.
        
        Args:
            ablation_results: Dictionary with ablation study results
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        if not ablation_results:
            logger.warning("No ablation results to plot")
            return None
        
        model_names = list(ablation_results.keys())
        
        # Prepare data
        accuracies = []
        accuracy_stds = []
        
        for model_name in model_names:
            accuracies.append(ablation_results[model_name].get('accuracy_mean', 0))
            accuracy_stds.append(ablation_results[model_name].get('accuracy_std', 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        colors = sns.color_palette(self.color_palette, len(model_names))
        x_pos = range(len(model_names))
        
        bars = ax.bar(x_pos, accuracies, yerr=accuracy_stds, 
                     capsize=5, color=colors, alpha=0.7, 
                     error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        # Customize plot
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Ablation Study Results\\n(Mean ± Standard Deviation)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (acc, std) in enumerate(zip(accuracies, accuracy_stds)):
            ax.text(i, acc + std + 0.01, f'{acc:.3f}±{std:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
            logger.info(f"Ablation results saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, results: Dict, models: Dict, 
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   training_histories: Dict = None,
                                   ablation_results: Dict = None,
                                   save_dir: str = "results/figures/") -> Dict[str, plt.Figure]:
        """
        Create comprehensive visual report with all plots.
        
        Args:
            results: Model results dictionary
            models: Trained models dictionary  
            X_val: Validation features
            y_val: Validation labels
            training_histories: Training histories (optional)
            ablation_results: Ablation study results (optional)
            save_dir: Directory to save figures
            
        Returns:
            Dictionary with figure objects
        """
        figures = {}
        
        # Model comparison plot
        logger.info("Creating model comparison plots...")
        fig_comparison = self.generate_comparison_plots(results, f"{save_dir}model_comparison.{self.save_format}")
        figures['comparison'] = fig_comparison
        
        # ROC curves
        logger.info("Creating ROC curves...")
        fig_roc = self.plot_roc_curves(models, X_val, y_val, f"{save_dir}roc_curves.{self.save_format}")
        figures['roc'] = fig_roc
        
        # Confusion matrices
        logger.info("Creating confusion matrices...")
        fig_cm = self.plot_confusion_matrices(models, X_val, y_val, f"{save_dir}confusion_matrices.{self.save_format}")
        figures['confusion_matrices'] = fig_cm
        
        # Training history (if available)
        if training_histories:
            logger.info("Creating training history plots...")
            fig_history = self.plot_training_history(training_histories, f"{save_dir}training_history.{self.save_format}")
            figures['training_history'] = fig_history
        
        # Ablation results (if available)
        if ablation_results:
            logger.info("Creating ablation study plots...")
            fig_ablation = self.plot_ablation_results(ablation_results, f"{save_dir}ablation_results.{self.save_format}")
            figures['ablation'] = fig_ablation
        
        logger.info(f"Comprehensive report created with {len(figures)} figures")
        
        return figures


def generate_comparison_plots(results: Dict[str, Dict], save_dir: str = "results/figures/") -> Dict[str, plt.Figure]:
    """
    Convenience function to generate comparison plots.
    
    Args:
        results: Model results dictionary
        save_dir: Directory to save figures
        
    Returns:
        Dictionary with figure objects
    """
    visualizer = ModelComparisonVisualizer()
    
    # Create comparison plot
    fig = visualizer.generate_comparison_plots(results, f"{save_dir}comparison.png")
    
    return {'comparison': fig}


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Sample results data
    sample_results = {
        'TaxoCapsNet': {
            'val_accuracy': 0.937,
            'val_sensitivity': 0.873,
            'val_specificity': 0.986,
            'val_precision': 0.983,
            'val_f1': 0.922,
            'val_auc': 0.957,
            'val_ic': 0.385,
            'val_ic_ci': (0.32, 0.45),
            'training_time': 70
        },
        'Random_Forest': {
            'val_accuracy': 0.912,
            'val_sensitivity': 0.845,
            'val_specificity': 0.954,
            'val_precision': 0.934,
            'val_f1': 0.887,
            'val_auc': 0.934,
            'val_ic': 0.342,
            'val_ic_ci': (0.28, 0.41),
            'training_time': 2.5
        }
    }
    
    # Test comparison plots
    visualizer = ModelComparisonVisualizer()
    fig = visualizer.generate_comparison_plots(sample_results)
    
    print("Visualization test completed!")
    print(f"Generated figure with size: {fig.get_size_inches()}")
