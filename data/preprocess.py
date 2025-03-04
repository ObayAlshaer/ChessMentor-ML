# src/data/preprocess.py

import os
import argparse
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
import shutil
from tqdm import tqdm
from typing import Dict, List, Tuple
import albumentations as A


def detect_chess_board(image_path: str) -> Tuple[bool, np.ndarray]:
    """
    Detect chess board corners in an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (success, corners)
        - success: True if board was detected, False otherwise
        - corners: Array of corner coordinates (4, 2) if success, None otherwise
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return False, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Loop through contours and find the chess board
    for contour in contours[:10]:  # Check only the 10 largest contours
        # Approximate contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # If we have a quadrilateral with a large enough area, assume it's the board
        if len(approx) == 4 and cv2.contourArea(contour) > 10000:
            # Get corner points
            corners = approx.reshape(4, 2)
            
            # Sort corners (top-left, top-right, bottom-right, bottom-left)
            corners = sort_corners(corners)
            
            return True, corners
    
    return False, None


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """
    Sort corners in order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        corners: Array of corner coordinates (4, 2)
    
    Returns:
        Sorted corners array
    """
    # Calculate center
    center = corners.mean(axis=0)
    
    # Sort corners based on their position relative to the center
    sorted_corners = []
    for sector in [(False, False), (False, True), (True, True), (True, False)]:
        sector_corners = [
            corner for corner in corners
            if (corner[0] >= center[0]) == sector[0] and (corner[1] >= center[1]) == sector[1]
        ]
        
        # If we have multiple corners in a sector, choose the one closest to the sector's ideal position
        if sector_corners:
            if sector == (False, False):  # Top-left
                idx = np.argmin([np.sum(corner**2) for corner in sector_corners])
            elif sector == (False, True):  # Top-right
                idx = np.argmin([np.sum((corner - [0, center[1]*2])**2) for corner in sector_corners])
            elif sector == (True, True):  # Bottom-right
                idx = np.argmin([np.sum((corner - [center[0]*2, center[1]*2])**2) for corner in sector_corners])
            else:  # Bottom-left
                idx = np.argmin([np.sum((corner - [center[0]*2, 0])**2) for corner in sector_corners])
            
            sorted_corners.append(sector_corners[idx])
    
    return np.array(sorted_corners)


def create_synthetic_data(
    output_dir: str,
    num_samples: int = 1000,
    piece_images_dir: str = "data/piece_images",
    board_images_dir: str = "data/board_images"
) -> None:
    """
    Create synthetic chess board images with annotations.
    
    Args:
        output_dir: Directory to save generated data
        num_samples: Number of synthetic samples to generate
        piece_images_dir: Directory containing piece images
        board_images_dir: Directory containing chess board images
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load piece images
    piece_images = {}
    for piece_type in ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]:
        piece_dir = os.path.join(piece_images_dir, piece_type)
        piece_files = [f for f in os.listdir(piece_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        piece_images[piece_type] = [os.path.join(piece_dir, f) for f in piece_files]
    
    # Load board images
    board_files = [f for f in os.listdir(board_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    board_paths = [os.path.join(board_images_dir, f) for f in board_files]
    
    # Standard chess starting position in FEN
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Create augmentation pipeline
    augmentation = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.Rotate(limit=15, p=0.5),
    ])
    
    # Generate samples
    annotations = []
    
    for i in tqdm(range(num_samples)):
        # Select random board image
        board_path = random.choice(board_paths)
        board_img = cv2.imread(board_path)
        
        # Extract board dimensions
        board_h, board_w = board_img.shape[:2]
        square_size = min(board_h, board_w) // 8
        
        # Create empty board state
        board_state = np.zeros((8, 8), dtype=object)
        
        # Decide whether to use starting position or random position
        use_starting_position = random.random() < 0.3
        
        if use_starting_position:
            # Parse starting FEN
            board_state = parse_fen_to_board(starting_fen)
        else:
            # Generate random board state
            num_pieces = random.randint(5, 32)
            placed_pieces = 0
            
            while placed_pieces < num_pieces:
                row = random.randint(0, 7)
                col = random.randint(0, 7)
                
                if board_state[row, col] == 0:  # Empty square
                    piece_type = random.choice(list(piece_images.keys()))
                    board_state[row, col] = piece_type
                    placed_pieces += 1
        
        # Place pieces on the board
        for row in range(8):
            for col in range(8):
                piece = board_state[row, col]
                if piece != 0:
                    # Select random piece image
                    piece_path = random.choice(piece_images[piece])
                    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
                    
                    # Resize piece to fit square
                    piece_size = int(square_size * 0.8)
                    piece_img = cv2.resize(piece_img, (piece_size, piece_size))
                    
                    # Calculate position
                    x = col * square_size + (square_size - piece_size) // 2
                    y = row * square_size + (square_size - piece_size) // 2
                    
                    # Place piece on board
                    if piece_img.shape[2] == 4:  # With alpha channel
                        alpha = piece_img[:, :, 3] / 255.0
                        for c in range(3):
                            board_img[y:y+piece_size, x:x+piece_size, c] = (
                                (1 - alpha) * board_img[y:y+piece_size, x:x+piece_size, c] +
                                alpha * piece_img[:, :, c]
                            )
                    else:
                        board_img[y:y+piece_size, x:x+piece_size] = piece_img
        
        # Apply augmentation
        augmented = augmentation(image=board_img)
        augmented_img = augmented["image"]
        
        # Save image
        output_path = os.path.join(output_dir, "images", f"synthetic_{i:06d}.jpg")
        cv2.imwrite(output_path, augmented_img)
        
        # Convert board state to FEN
        fen = board_to_fen(board_state)
        
        # Create annotation
        annotation = {
            "image_filename": f"synthetic_{i:06d}.jpg",
            "board_state": fen,
            "synthetic": True,
            "split": "train" if random.random() < 0.8 else "val"
        }
        
        annotations.append(annotation)
    
    # Save annotations
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)


def parse_fen_to_board(fen: str) -> np.ndarray:
    """
    Parse FEN notation to board state array.
    
    Args:
        fen: FEN notation string
    
    Returns:
        8x8 numpy array representing the board
    """
    # Extract board part
    board_part = fen.split(" ")[0]
    rows = board_part.split("/")
    
    board = np.zeros((8, 8), dtype=object)
    
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                board[i, j] = char
                j += 1
    
    return board


def board_to_fen(board: np.ndarray) -> str:
    """
    Convert board array to FEN notation.
    
    Args:
        board: 8x8 numpy array representing the board
    
    Returns:
        FEN notation string
    """
    fen_parts = []
    
    for i in range(8):
        empty_count = 0
        row_str = ""
        
        for j in range(8):
            piece = board[i, j]
            
            if piece == 0:  # Empty square
                empty_count += 1
            else:
                # If there were empty squares before, add the count
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                
                # Add the piece
                row_str += piece
        
        # Add any remaining empty squares
        if empty_count > 0:
            row_str += str(empty_count)
        
        fen_parts.append(row_str)
    
    # Join rows with slashes
    board_fen = '/'.join(fen_parts)
    
    # Add default values for additional FEN components
    fen = f"{board_fen} w - - 0 1"
    
    return fen


def process_dataset(
    input_dir: str,
    output_dir: str,
    config: Dict,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> None:
    """
    Process a dataset of chess board images.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory
        config: Configuration dictionary
        split_ratio: Train/val/test split ratio
        
    Returns:
        None
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
    
    # Shuffle image files
    random.shuffle(image_files)
    
    # Split into train/val/test
    n = len(image_files)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Process each split
    train_annotations = process_split(train_files, os.path.join(output_dir, "train"), "train")
    val_annotations = process_split(val_files, os.path.join(output_dir, "val"), "val")
    test_annotations = process_split(test_files, os.path.join(output_dir, "test"), "test")
    
    # Save annotations
    with open(os.path.join(output_dir, "train_annotations.json"), "w") as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(os.path.join(output_dir, "val_annotations.json"), "w") as f:
        json.dump(val_annotations, f, indent=2)
    
    with open(os.path.join(output_dir, "test_annotations.json"), "w") as f:
        json.dump(test_annotations, f, indent=2)


def process_split(
    files: List[Path],
    output_dir: str,
    split: str
) -> List[Dict]:
    """
    Process a split of the dataset.
    
    Args:
        files: List of image file paths
        output_dir: Output directory
        split: Split name (train, val, test)
        
    Returns:
        List of annotations
    """
    annotations = []
    
    for i, file_path in enumerate(tqdm(files, desc=f"Processing {split} split")):
        # Detect chess board
        success, corners = detect_chess_board(str(file_path))
        
        if success:
            # Copy image to output directory
            output_filename = f"{split}_{i:06d}{file_path.suffix}"
            output_path = os.path.join(output_dir, output_filename)
            shutil.copy(file_path, output_path)
            
            # Create annotation
            annotation = {
                "image_filename": output_filename,
                "corners": corners.tolist(),
                "split": split
            }
            
            # If this is a labeled dataset with FEN in the filename
            if "_fen_" in file_path.stem:
                fen = file_path.stem.split("_fen_")[1].replace("_", "/")
                annotation["board_state"] = fen
            
            annotations.append(annotation)
    
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chess board dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input-dir", type=str, help="Input directory with raw images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num-synthetic", type=int, default=1000, help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.synthetic:
        # Generate synthetic data
        create_synthetic_data(
            output_dir=args.output_dir,
            num_samples=args.num_synthetic
        )
    elif args.input_dir:
        # Process real dataset
        process_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config
        )
    else:
        parser.error("Either --synthetic or --input-dir must be provided")