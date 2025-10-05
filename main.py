#!/usr/bin/env python3
"""
TaxoCapsNet Main Execution Script
=====================================

Main script for running TaxoCapsNet experiments, baseline comparisons, and ablation studies.

Usage:
    python main.py --mode train --config config/config.yaml
    python main.py --mode baseline --data data/GSE_df.csv
    python main.py --mode ablation --seeds 42,123,456
    
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessing import load_and_preprocess_data, prepare_taxonomy_data
from src.models.taxocapsnet import TaxoCapsNet
from src.training.trainer import TaxoCapsNetTrainer, BaselineModelTrainer
from src.training.evaluation import AblationStudyRunner
from src.utils.taxonomy import generate_taxonomy_map
from src.visualization.plots import generate_comparison_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)


def setup_directories():
    """Create necessary directories"""
    dirs = ['results', 'results/models', 'results/figures', 'results/reports']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")


def train_taxocapsnet(config: dict) -> dict:
    """Train TaxoCapsNet model"""
    logger.info("🚀 Starting TaxoCapsNet Training")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv(config['data']['path'])
    X_processed, y, feature_names = load_and_preprocess_data(df, config)
    
    # Generate taxonomy mapping
    taxonomy_map = generate_taxonomy_map(
        config.get('data', {}).get('taxonomy_file', None)
    )
    
    # Prepare taxonomy-grouped data
    X_grouped, y, used_phyla = prepare_taxonomy_data(df, taxonomy_map)
    
    # Initialize trainer
    trainer = TaxoCapsNetTrainer(config, random_state=config['data']['random_state'])
    
    # Split data
    train_data, val_data = trainer.prepare_train_val_split(X_grouped, y, config)
    X_train_grouped, y_train, X_val_grouped, y_val = train_data + val_data
    
    # Build model
    model = TaxoCapsNet(
        input_shapes=[X.shape[1] for X in X_train_grouped],
        taxonomy_map=taxonomy_map,
        config=config
    )
    
    # Train model
    history = trainer.train_model(model, X_train_grouped, y_train, X_val_grouped, y_val)
    
    # Evaluate model
    results = trainer.evaluate_model(model, X_val_grouped, y_val)
    
    # Save model and results
    model.save('results/models/taxocapsnet.h5')
    
    # Save training history
    pd.DataFrame(history.history).to_csv('results/reports/training_history.csv', index=False)
    
    logger.info(f"✅ TaxoCapsNet training completed. Validation accuracy: {results['val_accuracy']:.4f}")
    return results


def run_baseline_comparison(config: dict) -> pd.DataFrame:
    """Run baseline model comparison"""
    logger.info("🔬 Starting Baseline Model Comparison")
    
    # Load data
    df = pd.read_csv(config['data']['path'])
    X, y, _ = load_and_preprocess_data(df, config)
    
    # Initialize baseline trainer
    baseline_trainer = BaselineModelTrainer(config, random_state=config['data']['random_state'])
    
    # Prepare data
    baseline_trainer.prepare_data(X, y)
    
    # Train all baseline models
    baseline_trainer.train_all_models()
    
    # Generate comparison plots
    results_df = baseline_trainer.generate_comparison_plots()
    
    # Save results
    results_df.to_csv('results/reports/baseline_comparison.csv', index=False)
    
    logger.info("✅ Baseline comparison completed")
    return results_df


def run_ablation_studies(config: dict, seeds: list) -> dict:
    """Run comprehensive ablation studies"""
    logger.info("🧪 Starting Ablation Studies")
    
    # Load data
    df = pd.read_csv(config['data']['path'])
    taxonomy_map = generate_taxonomy_map(
        config.get('data', {}).get('taxonomy_file', None)
    )
    X_grouped, y, used_phyla = prepare_taxonomy_data(df, taxonomy_map)
    
    # Initialize ablation runner
    ablation_runner = AblationStudyRunner(config)
    
    # Run multi-seed ablation studies
    all_results = ablation_runner.run_multi_seed_studies(
        X_grouped, y, taxonomy_map, seeds=seeds
    )
    
    # Calculate averaged results
    averaged_results = ablation_runner.calculate_averaged_results(all_results)
    
    # Generate comparison table
    ablation_runner.print_ablation_results_table(averaged_results)
    
    # Save results
    ablation_results_df = pd.DataFrame.from_dict(averaged_results, orient='index')
    ablation_results_df.to_csv('results/reports/ablation_results.csv')
    
    logger.info("✅ Ablation studies completed")
    return averaged_results


def evaluate_saved_model(config: dict, model_path: str):
    """Evaluate a saved model"""
    logger.info(f"📊 Evaluating saved model: {model_path}")
    
    # Load model
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    
    # Load test data
    df = pd.read_csv(config['data']['path'])
    taxonomy_map = generate_taxonomy_map()
    X_grouped, y, _ = prepare_taxonomy_data(df, taxonomy_map)
    
    # Evaluate
    trainer = TaxoCapsNetTrainer(config)
    results = trainer.evaluate_model(model, X_grouped, y)
    
    logger.info(f"Model evaluation results: {results}")
    return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='TaxoCapsNet Training and Evaluation')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'baseline', 'ablation', 'evaluate'],
                       help='Execution mode')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    
    # Data
    parser.add_argument('--data', type=str, 
                       help='Override data path from config')
    
    # Ablation specific
    parser.add_argument('--seeds', type=str, default='42,123,456',
                       help='Random seeds for ablation (comma-separated)')
    
    # Model evaluation
    parser.add_argument('--model', type=str,
                       help='Path to saved model for evaluation')
    
    # Additional parameters
    parser.add_argument('--batch_size', type=int,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--learning_rate', type=float,
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data:
        config['data']['path'] = args.data
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    logger.info(f"Starting TaxoCapsNet in {args.mode} mode")
    
    try:
        if args.mode == 'train':
            results = train_taxocapsnet(config)
            logger.info(f"Training completed with validation accuracy: {results.get('val_accuracy', 'N/A')}")
            
        elif args.mode == 'baseline':
            results_df = run_baseline_comparison(config)
            logger.info(f"Baseline comparison completed with {len(results_df)} models")
            
        elif args.mode == 'ablation':
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            results = run_ablation_studies(config, seeds)
            logger.info(f"Ablation studies completed with {len(results)} model variants")
            
        elif args.mode == 'evaluate':
            if not args.model:
                logger.error("Model path required for evaluation mode")
                sys.exit(1)
            results = evaluate_saved_model(config, args.model)
            logger.info(f"Model evaluation completed")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
    
    logger.info("🎉 Execution completed successfully!")


if __name__ == "__main__":
    main()
