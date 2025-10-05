"""
Evaluation and Ablation Studies
===============================

Comprehensive evaluation pipeline including ablation studies for TaxoCapsNet.
Contains multi-seed evaluation, statistical significance testing, and model comparison.

"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
import time

# Local imports
from src.models.taxocapsnet_model import TaxoCapsNet
from src.models.ablation_models import TaxoDenseModel, RandomCapsNetModel, FlatCapsNetModel
from src.utils.metrics import calculate_comprehensive_metrics_with_ci, mcnemar_test, paired_t_test
from src.utils.taxonomy import create_random_taxonomy_map
from src.training.trainer import TaxoCapsNetTrainer

logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """
    Runner for comprehensive ablation studies of TaxoCapsNet.
    
    Tests three key components:
    1. Capsule Architecture (vs Dense Networks)
    2. Biological Taxonomy (vs Random Grouping) 
    3. Hierarchical Design (vs Flat Input)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ablation study runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ablation_config = config.get('ablation', {})
        self.results = {}
        
        logger.info("Ablation study runner initialized")
    
    def create_ablation_models(self, input_shapes: List[int], taxonomy_map: Dict[str, List[str]]) -> Dict:
        """
        Create all ablation models for comparison.
        
        Args:
            input_shapes: List of input dimensions for each phylum group
            taxonomy_map: Mapping of phylum names to OTU lists
            
        Returns:
            Dictionary of model instances
        """
        models = {}
        
        # Main TaxoCapsNet model
        if self.ablation_config.get('models', {}).get('taxocapsnet', {}).get('enabled', True):
            models['TaxoCapsNet'] = TaxoCapsNet(input_shapes, taxonomy_map, self.config)
        
        # Ablation 1: TaxoDense (Dense networks with taxonomy)
        if self.ablation_config.get('models', {}).get('taxo_dense', {}).get('enabled', True):
            models['TaxoDense'] = TaxoDenseModel(input_shapes, taxonomy_map, self.config)
        
        # Ablation 2: RandomCapsNet (Capsules with random grouping)
        if self.ablation_config.get('models', {}).get('random_capsnet', {}).get('enabled', True):
            total_features = sum(input_shapes)
            random_taxonomy = create_random_taxonomy_map(total_features, seed=42)
            models['RandomCapsNet'] = RandomCapsNetModel(input_shapes, random_taxonomy, self.config)
        
        # Ablation 3: FlatCapsNet (Capsules without hierarchy)
        if self.ablation_config.get('models', {}).get('flat_capsnet', {}).get('enabled', True):
            total_features = sum(input_shapes)
            models['FlatCapsNet'] = FlatCapsNetModel(total_features, self.config)
        
        logger.info(f"Created {len(models)} ablation models: {list(models.keys())}")
        return models
    
    def run_single_seed_study(self, X_grouped: List[np.ndarray], y: np.ndarray, 
                             taxonomy_map: Dict[str, List[str]], seed: int) -> Dict[str, Dict]:
        """
        Run ablation study for a single random seed.
        
        Args:
            X_grouped: Input data grouped by phylum
            y: Target labels
            taxonomy_map: Taxonomy mapping
            seed: Random seed
            
        Returns:
            Dictionary with results for all models
        """
        logger.info(f"Running ablation study with seed {seed}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Create train-test split
        test_size = self.config.get('data', {}).get('test_size', 0.2)
        
        # Split each phylum group
        train_indices, test_indices = train_test_split(
            range(len(y)), test_size=test_size, random_state=seed, stratify=y
        )
        
        X_train_grouped = [data[train_indices] for data in X_grouped]
        X_test_grouped = [data[test_indices] for data in X_grouped]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        logger.info(f"Seed {seed}: Train={len(y_train)}, Test={len(y_test)}")
        
        # Get input shapes
        input_shapes = [data.shape[1] for data in X_train_grouped]
        
        # Create ablation models
        models = self.create_ablation_models(input_shapes, taxonomy_map)
        
        results = {}
        
        for model_name, model_instance in models.items():
            logger.info(f"Training {model_name} (seed {seed})...")
            
            try:
                # Build and compile model
                model = model_instance.build_model()
                learning_rate = self.config.get('training', {}).get('learning_rate', 0.001)
                model_instance.compile_model(learning_rate)
                
                # Create trainer
                trainer = TaxoCapsNetTrainer(self.config, random_state=seed)
                
                # Train model
                start_time = time.time()
                history = trainer.train_model(
                    model, X_train_grouped, y_train, X_test_grouped, y_test
                )
                training_time = time.time() - start_time
                
                # Evaluate model
                metrics = trainer.evaluate_model(model, X_test_grouped, y_test)
                metrics['training_time'] = training_time
                metrics['seed'] = seed
                
                results[model_name] = metrics
                
                logger.info(f"✅ {model_name} (seed {seed}) - Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name} (seed {seed}): {e}")
                continue
        
        return results
    
    def run_multi_seed_studies(self, X_grouped: List[np.ndarray], y: np.ndarray,
                              taxonomy_map: Dict[str, List[str]], seeds: List[int] = None) -> Dict:
        """
        Run ablation studies across multiple random seeds.
        
        Args:
            X_grouped: Input data grouped by phylum
            y: Target labels
            taxonomy_map: Taxonomy mapping
            seeds: List of random seeds (default from config)
            
        Returns:
            Dictionary with results for all models across all seeds
        """
        if seeds is None:
            seeds = self.ablation_config.get('seeds', [42, 123, 456])
        
        logger.info(f"Running multi-seed ablation studies with seeds: {seeds}")
        
        all_results = {}
        
        for seed in seeds:
            seed_results = self.run_single_seed_study(X_grouped, y, taxonomy_map, seed)
            
            # Store results by model name
            for model_name, metrics in seed_results.items():
                if model_name not in all_results:
                    all_results[model_name] = []
                all_results[model_name].append(metrics)
        
        logger.info(f"Completed multi-seed studies for {len(all_results)} models")
        
        # Store results
        self.results = all_results
        
        return all_results
    
    def calculate_averaged_results(self, all_results: Dict) -> Dict[str, Dict]:
        """
        Calculate averaged results across seeds with confidence intervals.
        
        Args:
            all_results: Results from all seeds
            
        Returns:
            Dictionary with averaged metrics and confidence intervals
        """
        averaged_results = {}
        
        for model_name, results_list in all_results.items():
            if not results_list:
                continue
            
            metrics = {}
            
            # Calculate mean and std for each metric
            metric_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'ic']
            
            for metric in metric_names:
                values = [result[metric] for result in results_list if metric in result and not np.isnan(result[metric])]
                
                if values:
                    metrics[f'{metric}_mean'] = np.mean(values)
                    metrics[f'{metric}_std'] = np.std(values)
                    metrics[f'{metric}_values'] = values
                    
                    # 95% confidence interval
                    n = len(values)
                    if n > 1:
                        sem = np.std(values) / np.sqrt(n)
                        ci = 1.96 * sem  # 95% CI
                        metrics[f'{metric}_ci_lower'] = metrics[f'{metric}_mean'] - ci
                        metrics[f'{metric}_ci_upper'] = metrics[f'{metric}_mean'] + ci
                    else:
                        metrics[f'{metric}_ci_lower'] = metrics[f'{metric}_mean']
                        metrics[f'{metric}_ci_upper'] = metrics[f'{metric}_mean']
                else:
                    metrics[f'{metric}_mean'] = np.nan
                    metrics[f'{metric}_std'] = np.nan
                    metrics[f'{metric}_ci_lower'] = np.nan
                    metrics[f'{metric}_ci_upper'] = np.nan
            
            # Average training time
            training_times = [result['training_time'] for result in results_list if 'training_time' in result]
            metrics['training_time_mean'] = np.mean(training_times) if training_times else np.nan
            
            # Store seed results for statistical tests
            metrics['seed_results'] = results_list
            
            averaged_results[model_name] = metrics
        
        logger.info(f"Calculated averaged results for {len(averaged_results)} models")
        
        return averaged_results
    
    def perform_statistical_tests(self, averaged_results: Dict[str, Dict]) -> Dict:
        """
        Perform statistical significance tests between models.
        
        Args:
            averaged_results: Averaged results from multiple seeds
            
        Returns:
            Dictionary with statistical test results
        """
        statistical_results = {}
        
        # Compare TaxoCapsNet with each ablation model
        if 'TaxoCapsNet' not in averaged_results:
            logger.warning("TaxoCapsNet results not found for statistical testing")
            return statistical_results
        
        taxocapsnet_results = averaged_results['TaxoCapsNet']
        
        for model_name, model_results in averaged_results.items():
            if model_name == 'TaxoCapsNet':
                continue
            
            logger.info(f"Statistical tests: TaxoCapsNet vs {model_name}")
            
            # Extract accuracy values for both models
            if 'accuracy_values' in taxocapsnet_results and 'accuracy_values' in model_results:
                taxo_accuracies = np.array(taxocapsnet_results['accuracy_values'])
                model_accuracies = np.array(model_results['accuracy_values'])
                
                # Paired t-test
                if len(taxo_accuracies) == len(model_accuracies) and len(taxo_accuracies) > 1:
                    t_stat, p_value = paired_t_test(taxo_accuracies, model_accuracies)
                    
                    statistical_results[f'TaxoCapsNet_vs_{model_name}'] = {
                        'metric': 'accuracy',
                        'taxocapsnet_mean': np.mean(taxo_accuracies),
                        'comparison_mean': np.mean(model_accuracies),
                        'mean_difference': np.mean(taxo_accuracies) - np.mean(model_accuracies),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'taxocapsnet_better': np.mean(taxo_accuracies) > np.mean(model_accuracies)
                    }
                    
                    logger.info(f"  Accuracy difference: {np.mean(taxo_accuracies) - np.mean(model_accuracies):.4f}")
                    logger.info(f"  p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
        
        return statistical_results
    
    def print_ablation_results_table(self, averaged_results: Dict[str, Dict]):
        """
        Print comprehensive ablation results table.
        
        Args:
            averaged_results: Averaged results from multiple seeds
        """
        logger.info("\\n" + "="*120)
        logger.info("📊 ABLATION STUDY RESULTS (Multi-seed Average ± Std)")
        logger.info("="*120)
        
        # Header
        header = (f"{'Model':<15} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12} "
                 f"{'Precision':<12} {'F1 Score':<10} {'AUC':<10} {'IC':<10} {'Time (s)':<10}")
        logger.info(header)
        logger.info("-" * 120)
        
        # Sort models by accuracy
        sorted_models = sorted(averaged_results.items(), 
                             key=lambda x: x[1].get('accuracy_mean', 0), reverse=True)
        
        for model_name, metrics in sorted_models:
            # Format metrics with std
            acc_str = f"{metrics.get('accuracy_mean', np.nan):.3f}±{metrics.get('accuracy_std', np.nan):.3f}"
            sens_str = f"{metrics.get('sensitivity_mean', np.nan):.3f}±{metrics.get('sensitivity_std', np.nan):.3f}"
            spec_str = f"{metrics.get('specificity_mean', np.nan):.3f}±{metrics.get('specificity_std', np.nan):.3f}"
            prec_str = f"{metrics.get('precision_mean', np.nan):.3f}±{metrics.get('precision_std', np.nan):.3f}"
            f1_str = f"{metrics.get('f1_mean', np.nan):.3f}±{metrics.get('f1_std', np.nan):.3f}"
            auc_str = f"{metrics.get('auc_mean', np.nan):.3f}±{metrics.get('auc_std', np.nan):.3f}"
            ic_str = f"{metrics.get('ic_mean', np.nan):.3f}±{metrics.get('ic_std', np.nan):.3f}"
            time_str = f"{metrics.get('training_time_mean', np.nan):.1f}"
            
            logger.info(f"{model_name:<15} {acc_str:<12} {sens_str:<12} {spec_str:<12} "
                       f"{prec_str:<12} {f1_str:<10} {auc_str:<10} {ic_str:<10} {time_str:<10}")
        
        logger.info("="*120)
        
        # Statistical significance tests
        statistical_results = self.perform_statistical_tests(averaged_results)
        
        if statistical_results:
            logger.info("\\n📈 STATISTICAL SIGNIFICANCE TESTS")
            logger.info("-" * 60)
            
            for comparison, stats in statistical_results.items():
                significance = "✅ Significant" if stats['significant'] else "❌ Not significant"
                better = "✅ TaxoCapsNet better" if stats['taxocapsnet_better'] else "❌ Comparison better"
                
                logger.info(f"{comparison}:")
                logger.info(f"  Mean difference: {stats['mean_difference']:.4f}")
                logger.info(f"  p-value: {stats['p_value']:.4f} ({significance})")
                logger.info(f"  {better}")
                logger.info("")
    
    def generate_ablation_summary(self, averaged_results: Dict[str, Dict]) -> str:
        """
        Generate text summary of ablation study results.
        
        Args:
            averaged_results: Averaged results from multiple seeds
            
        Returns:
            Text summary string
        """
        summary_lines = []
        summary_lines.append("ABLATION STUDY SUMMARY")
        summary_lines.append("=" * 50)
        
        if 'TaxoCapsNet' in averaged_results:
            taxo_acc = averaged_results['TaxoCapsNet'].get('accuracy_mean', np.nan)
            summary_lines.append(f"TaxoCapsNet Accuracy: {taxo_acc:.3f}±{averaged_results['TaxoCapsNet'].get('accuracy_std', np.nan):.3f}")
            summary_lines.append("")
            
            # Compare with ablations
            if 'TaxoDense' in averaged_results:
                dense_acc = averaged_results['TaxoDense'].get('accuracy_mean', np.nan)
                improvement = taxo_acc - dense_acc
                summary_lines.append(f"Capsule vs Dense: +{improvement:.3f} ({improvement/dense_acc*100:.1f}% improvement)")
            
            if 'RandomCapsNet' in averaged_results:
                random_acc = averaged_results['RandomCapsNet'].get('accuracy_mean', np.nan)
                improvement = taxo_acc - random_acc
                summary_lines.append(f"Taxonomy vs Random: +{improvement:.3f} ({improvement/random_acc*100:.1f}% improvement)")
            
            if 'FlatCapsNet' in averaged_results:
                flat_acc = averaged_results['FlatCapsNet'].get('accuracy_mean', np.nan)
                improvement = taxo_acc - flat_acc
                summary_lines.append(f"Hierarchical vs Flat: +{improvement:.3f} ({improvement/flat_acc*100:.1f}% improvement)")
        
        return "\\n".join(summary_lines)


def run_comprehensive_ablation_study(X_grouped: List[np.ndarray], y: np.ndarray,
                                    taxonomy_map: Dict[str, List[str]], config: Dict[str, Any],
                                    seeds: List[int] = None) -> Tuple[Dict, str]:
    """
    Run comprehensive ablation study with multiple seeds.
    
    Args:
        X_grouped: Input data grouped by phylum
        y: Target labels
        taxonomy_map: Taxonomy mapping
        config: Configuration dictionary
        seeds: Random seeds for evaluation
        
    Returns:
        Tuple of (averaged_results, summary_text)
    """
    # Initialize runner
    runner = AblationStudyRunner(config)
    
    # Run multi-seed studies
    all_results = runner.run_multi_seed_studies(X_grouped, y, taxonomy_map, seeds)
    
    # Calculate averaged results
    averaged_results = runner.calculate_averaged_results(all_results)
    
    # Print results table
    runner.print_ablation_results_table(averaged_results)
    
    # Generate summary
    summary = runner.generate_ablation_summary(averaged_results)
    logger.info(f"\\n{summary}")
    
    return averaged_results, summary


if __name__ == "__main__":
    # Test ablation study runner
    print("Testing ablation study runner...")
    
    # Sample data and config
    np.random.seed(42)
    X_grouped = [np.random.randn(100, 50), np.random.randn(100, 30)]
    y = np.random.randint(0, 2, 100)
    taxonomy_map = {'Phylum1': [f'OTU{i}' for i in range(50)], 
                   'Phylum2': [f'OTU{i}' for i in range(50, 80)]}
    
    config = {
        'data': {'test_size': 0.2},
        'training': {'epochs': 5, 'learning_rate': 0.001},  # Small for testing
        'ablation': {
            'seeds': [42, 123],  # Just 2 seeds for testing
            'models': {
                'taxocapsnet': {'enabled': True},
                'taxo_dense': {'enabled': True}
            }
        }
    }
    
    # This would normally run the full ablation study
    print("Ablation study runner test setup completed!")
    print(f"Data shapes: {[arr.shape for arr in X_grouped]}")
    print(f"Labels shape: {y.shape}")
    print(f"Config: {list(config.keys())}")
