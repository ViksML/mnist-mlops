# MNIST Lightweight CNN Classifier

![ML Pipeline](https://github.com/<your-username>/<your-repo-name>/workflows/ML%20Pipeline/badge.svg)

A PyTorch implementation of a lightweight CNN for MNIST digit classification with specific performance requirements.

## Project Requirements

1. Model Parameters: < 25,000
2. Training Accuracy: > 95% in 1 epoch

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D (16 filters, 3x3)
    ↓
MaxPool2D (2x2)
    ↓
Conv2D (25 filters, 3x3)
    ↓
MaxPool2D (2x2)
    ↓
Flatten
    ↓
Linear (625 → 32)
    ↓
Linear (32 → 10)
```

## Training Configuration

- Batch Size: 8 (training), 32 (testing)
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Data Augmentation: Random rotation (±5°)

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

## Project Structure

- `model.py`: Model architecture definition
- `train.py`: Training and evaluation code
- `test_basic.py`: Core requirement tests
- `ci_pipeline.py`: CI/CD pipeline implementation

## Running Tests

Basic Requirements Tests:
```bash
python -m unittest test_basic.py -v
```
Tests:
- Parameter count verification (<25,000)
- Training accuracy validation (>95% in 1 epoch)

## Training

Run the training script:
```bash
python train.py
```

Output includes:
- Model parameter count
- Training progress (per 100 batches)
- Final training accuracy
- Test accuracy
- Augmented samples visualization (augmented_samples.png)

## Expected Performance

- Training Time: ~2-3 minutes (CPU)
- Parameter Count: ~20,000
- Training Accuracy: >95% (1 epoch)
- Test Accuracy: ~92-94%

## CI/CD Pipeline

Run the CI pipeline locally:
```bash
python ci_pipeline.py
```

The pipeline checks:
1. Dependencies installation
2. CUDA availability
3. Core requirements tests