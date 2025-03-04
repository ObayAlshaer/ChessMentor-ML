# ChessMentor - Chess Board Recognition Model

This repository contains the machine learning components for the ChessMentor iOS app, which uses augmented reality to analyze chess games and provide strategic move suggestions.

## Project Structure

```
ml/
├── configs/           # Configuration files for experiments
├── data/              # Training and validation datasets
├── docs/              # Documentation for model architecture and usage
├── experiments/       # Experiment results and model checkpoints
├── models/            # Model architecture definitions
├── notebooks/         # Exploratory analysis and visualization
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── training/      # Training pipeline code
│   ├── evaluation/    # Evaluation metrics and testing
│   └── inference/     # Inference pipeline for deployment
├── tests/             # Unit tests for critical components
└── .gitignore         # Git ignore file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- OpenCV
- Numpy
- Matplotlib
- Pillow
- tqdm
- PyYAML
- albumentations

You can install all required packages with:

```bash
pip install -r requirements.txt
```

### Data Preparation

1. The model requires a dataset of chess board images with annotations. You can either:
   - Use existing datasets like Chess-ID or ChessBoard Image Dataset
   - Generate synthetic data using our data generation script
   - Create your own dataset with annotated chess positions

For synthetic data generation:

```bash
python src/data/preprocess.py --config configs/baseline_config.yaml --synthetic --num-synthetic 5000 --output-dir data/synthetic
```

For processing a real dataset:

```bash
python src/data/preprocess.py --config configs/baseline_config.yaml --input-dir path/to/raw/images --output-dir data/processed
```

### Training the Model

To train the model using the baseline configuration:

```bash
python src/train.py --config configs/baseline_config.yaml --output-dir experiments
```

Optional arguments:
- `--resume`: Resume training from the latest checkpoint
- `--cpu`: Use CPU for training (not recommended)
- `--export`: Export the model for deployment after training

### Testing the Model

To evaluate the model performance on the test set:

```bash
python src/test.py --config configs/baseline_config.yaml --model-path experiments/best_model.pth --output-dir results
```

Optional arguments:
- `--visualize`: Generate visualizations of model predictions
- `--num-visualizations`: Number of samples to visualize (default: 10)

## Model Architecture

The chess board recognition model consists of two main components:

1. **Board Detection**: Locates the chess board in the image and extracts its four corners for perspective correction.
2. **Piece Classification**: Identifies the chess pieces on each of the 64 squares.

The model uses a two-stage approach:
- First, a neural network detects the chess board and extracts its corners
- Then, a perspective transformation is applied to get a standardized view of the board
- Finally, a second neural network classifies the contents of each square

## Deployment

The model can be exported to various formats for deployment:

### CoreML (for iOS)

```bash
python src/inference/export.py --model-path experiments/best_model.pth --config configs/baseline_config.yaml --format coreml --output-path exported/chess_recognition.mlmodel
```

### ONNX (cross-platform)

```bash
python src/inference/export.py --model-path experiments/best_model.pth --config configs/baseline_config.yaml --format onnx --output-path exported/chess_recognition.onnx
```

## Performance Metrics

The model is evaluated using the following metrics:

- **Square Accuracy**: Percentage of correctly classified squares
- **Board Accuracy**: Percentage of boards where all squares are correctly classified
- **Corner Detection Error**: Average Euclidean distance between predicted and ground truth corners
- **FEN Accuracy**: Percentage of boards with correctly generated FEN notation

## Contributing

1. Create a new branch for your feature or bugfix
2. Add appropriate unit tests
3. Ensure all tests pass
4. Submit a pull request

## License

Proprietary - ChessMentor Project

## Acknowledgments

- Chess-ID Dataset
- ChessSetGenerator for synthetic data generation
- PyTorch and torchvision libraries