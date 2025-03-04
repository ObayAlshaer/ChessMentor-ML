import os
import torch
import argparse
import yaml
import json
import logging
import datetime
import random
import numpy as np
from pathlib import Path

from models.chess_recognition_model import ChessBoardRecognitionModel
from src.data.chess_dataset import create_chess_dataloaders
from src.training.trainer import ChessBoardTrainer


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment']['name']}_{timestamp}"
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(experiment_dir, 'logs'))
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_chess_dataloaders(config)
    
    # Create model
    logger.info("Creating model...")
    model = ChessBoardRecognitionModel(config)
    model.to(device)
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = ChessBoardTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        logger=logger
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(resume=args.resume)
    
    # Save training history
    with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    logger.info(f"Training completed. Results saved to {experiment_dir}")
    
    # Export model for deployment if specified
    if args.export:
        logger.info("Exporting model for deployment...")
        from src.inference.inference import ChessBoardRecognitionService
        
        # Create inference service
        service = ChessBoardRecognitionService(
            model_path=os.path.join(experiment_dir, 'best_model.pth'),
            config_path=os.path.join(experiment_dir, 'config.yaml'),
            device=device
        )
        
        # Export model
        export_format = config.get('deployment', {}).get('format', 'coreml')
        export_path = os.path.join(experiment_dir, f'exported_model.{export_format}')
        service.export_model(export_path, format=export_format)
        
        logger.info(f"Model exported to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chess Board Recognition Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for training")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    
    args = parser.parse_args()
    main(args)