# src/test.py

import os
import argparse
import yaml
import json
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from models.chess_recognition_model import ChessBoardRecognitionModel
from src.data.chess_dataset import create_chess_dataloaders
from src.evaluation.metrics import calculate_metrics, evaluate_fen_accuracy, evaluate_corner_detection
from src.inference.inference import ChessBoardRecognitionService


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'testing.log')
    
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


def visualize_predictions(
    model_service: ChessBoardRecognitionService,
    test_loader,
    output_dir: str,
    num_samples: int = 10
):
    """
    Visualize model predictions on test samples.
    
    Args:
        model_service: Chess board recognition service
        test_loader: Test data loader
        output_dir: Output directory for visualization
        num_samples: Number of samples to visualize
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples from the test loader
    samples = []
    for batch in test_loader:
        samples.extend([(img, corners, board) for img, corners, board in 
                       zip(batch['image'], batch['corners'], batch['board_state'])])
        if len(samples) >= num_samples:
            break
    
    # Select a subset of samples
    samples = samples[:num_samples]
    
    # Piece to unicode mapping for visualization
    piece_to_unicode = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        'empty': ' '
    }
    
    # Process each sample
    for i, (image, corners_gt, board_gt) in enumerate(samples):
        # Convert tensor to PIL Image
        img = Image.fromarray(
            (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        
        # Run inference
        img_np = np.array(img)
        prediction = model_service.predict(img_np)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image with detected corners
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Draw ground truth corners
        img_corners_gt = img.copy()
        draw = ImageDraw.Draw(img_corners_gt)
        for j, (x, y) in enumerate(corners_gt.numpy()):
            draw.ellipse((x-5, y-5, x+5, y+5), fill='green')
            draw.text((x+5, y+5), f"{j}", fill='green')
        
        # Draw predicted corners
        for j, (x, y) in enumerate(prediction['corners']):
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red')
            draw.text((x+5, y+5), f"{j}", fill='red')
        
        axes[1].imshow(img_corners_gt)
        axes[1].set_title("Detected Corners (Green: GT, Red: Pred)")
        axes[1].axis('off')
        
        # Draw board state
        fig_board = plt.figure(figsize=(8, 8))
        ax_board = fig_board.add_subplot(111)
        
        # Create a chess board grid
        board_img = np.zeros((8, 8, 3), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                color = [240, 240, 240] if (row + col) % 2 == 0 else [180, 180, 180]
                board_img[row, col] = color
        
        # Plot the board
        ax_board.imshow(board_img)
        
        # Place pieces based on prediction
        for row in range(8):
            for col in range(8):
                piece = prediction['board_state'][row][col]
                if piece != 'empty':
                    ax_board.text(col, row, piece_to_unicode.get(piece, '?'),
                                  ha='center', va='center', fontsize=24)
        
        # Add grid lines
        for i in range(9):
            ax_board.axhline(i - 0.5, color='black', linewidth=1)
            ax_board.axvline(i - 0.5, color='black', linewidth=1)
        
        # Set board properties
        ax_board.set_xticks(range(8))
        ax_board.set_yticks(range(8))
        ax_board.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax_board.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'][::-1])
        ax_board.set_title(f"Predicted FEN: {prediction['fen']}")
        
        # Save board visualization
        board_path = os.path.join(output_dir, f"sample_{i}_board.png")
        fig_board.savefig(board_path, bbox_inches='tight')
        plt.close(fig_board)
        
        # Load the saved board image and display in the third subplot
        board_img = plt.imread(board_path)
        axes[2].imshow(board_img)
        axes[2].set_title("Recognized Board State")
        axes[2].axis('off')
        
        # Save the complete visualization
        output_path = os.path.join(output_dir, f"sample_{i}_visualization.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)


def test_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    output_dir: str
):
    """
    Test the model on the test dataset.
    
    Args:
        model: Chess board recognition model
        test_loader: Test data loader
        device: Device to run test on
        output_dir: Output directory for results
    
    Returns:
        Dictionary of test metrics
    """
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_board_preds = []
    all_board_targets = []
    all_corner_preds = []
    all_corner_targets = []
    all_fen_preds = []
    all_fen_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            corners_gt = batch['corners'].to(device)
            board_states_gt = batch['board_state'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predicted corners and piece logits
            corners_pred = outputs['corners']
            piece_logits = outputs['piece_logits']
            
            # Get piece predictions
            piece_preds = torch.argmax(piece_logits, dim=1)
            
            # Collect predictions and targets
            all_board_preds.append(piece_preds.cpu())
            all_board_targets.append(board_states_gt.cpu())
            all_corner_preds.append(corners_pred.cpu())
            all_corner_targets.append(corners_gt.cpu())
            
            # Convert predictions to FEN
            for i in range(piece_preds.size(0)):
                pred_board = piece_preds[i].cpu().numpy()
                target_board = board_states_gt[i].cpu().numpy()
                
                # Create inference service for FEN conversion
                # Note: This is inefficient but keeps code modular
                service = ChessBoardRecognitionService(None, None, device)
                
                pred_fen = service.board_to_fen(pred_board)
                target_fen = service.board_to_fen(target_board)
                
                all_fen_preds.append(pred_fen)
                all_fen_targets.append(target_fen)
    
    # Concatenate all predictions and targets
    all_board_preds = torch.cat(all_board_preds, dim=0)
    all_board_targets = torch.cat(all_board_targets, dim=0)
    all_corner_preds = torch.cat(all_corner_preds, dim=0)
    all_corner_targets = torch.cat(all_corner_targets, dim=0)
    
    # Calculate metrics
    board_metrics = calculate_metrics(all_board_preds, all_board_targets)
    corner_metrics = evaluate_corner_detection(all_corner_preds, all_corner_targets)
    fen_metrics = evaluate_fen_accuracy(all_fen_preds, all_fen_targets)
    
    # Combine all metrics
    metrics = {
        **board_metrics,
        **corner_metrics,
        **fen_metrics
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_chess_dataloaders(config)
    test_loader = dataloaders['test']
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = ChessBoardRecognitionModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create inference service
    model_service = ChessBoardRecognitionService(
        model_path=args.model_path,
        config_path=args.config,
        device=device
    )
    
    # Test model
    logger.info("Testing model...")
    metrics = test_model(model, test_loader, device, output_dir)
    
    # Print metrics
    logger.info("Test metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Visualize predictions
    if args.visualize:
        logger.info("Visualizing predictions...")
        visualize_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(visualize_dir, exist_ok=True)
        
        visualize_predictions(
            model_service=model_service,
            test_loader=test_loader,
            output_dir=visualize_dir,
            num_samples=args.num_visualizations
        )
        
        logger.info(f"Visualizations saved to {visualize_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Chess Board Recognition Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for testing")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--num-visualizations", type=int, default=10, help="Number of samples to visualize")
    
    args = parser.parse_args()
    main(args)