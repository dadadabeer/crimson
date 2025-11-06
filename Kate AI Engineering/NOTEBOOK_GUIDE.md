# Art Classifier Training Notebook - Complete Guide

This document provides a detailed overview of what each section in `art_classifier_training.ipynb` does.

## 📚 Notebook Structure

### Section 1: Import Libraries and Setup
**Purpose**: Imports all required Python libraries for deep learning, data processing, and visualization.
- Sets up PyTorch, torchvision for neural networks
- Imports matplotlib/seaborn for plotting
- Configures reproducibility with random seeds
- Detects GPU availability for faster training

### Section 2: Configuration and Data Paths
**Purpose**: Defines all hyperparameters and data locations.
- Batch size: 32 images per batch
- Learning rate: 0.001
- Max epochs: 20 (with early stopping)
- Image size: 224×224 pixels (ResNet50 standard)
- Automatically discovers 4 art style classes from folder structure
- Counts images in train/valid/test splits

### Section 3: Data Visualization - Class Distribution
**Purpose**: Visualizes how images are distributed across classes.
- Creates bar charts for train/valid/test splits
- Helps identify data imbalance issues
- Shows if splits maintain similar proportions

### Section 4: Custom Dataset Class
**Purpose**: Creates PyTorch Dataset for loading art images.
- `__init__`: Scans directories and collects image paths with labels
- `__len__`: Returns dataset size
- `__getitem__`: Loads and transforms individual images
- Maps folder names to numeric class indices

### Section 5: Data Augmentation
**Purpose**: Defines image transformations for training and evaluation.

**Training transforms** (with augmentation):
- Random horizontal flips
- Random rotations (±15°)
- Color jittering (brightness, contrast, saturation)
- Random translations and scaling
- Normalization using ImageNet statistics

**Validation/Test transforms** (no augmentation):
- Only resize and normalize
- Ensures consistent evaluation

### Section 6: Create DataLoaders
**Purpose**: Wraps datasets for efficient batch processing.
- Batch size: 32
- Training data: Shuffled each epoch
- Validation/Test: Not shuffled (consistent order)
- num_workers: 0 (avoids multiprocessing issues in notebooks)
- Calculates number of batches per epoch

### Section 7: Model Architecture - ResNet50
**Purpose**: Creates the neural network using transfer learning.
- Loads pre-trained ResNet50 (trained on ImageNet)
- Freezes early layers (keeps learned features)
- Replaces final layer for 4-class output
- Custom classifier head: 2048 → 512 → 4
- Adds dropout layers (0.5 and 0.3) for regularization
- Shows total vs trainable parameters

### Section 8: Training Setup
**Purpose**: Configures loss, optimizer, and learning rate scheduling.
- **Loss**: CrossEntropyLoss (standard for classification)
- **Optimizer**: Adam with weight decay (L2 regularization)
- **Scheduler**: ReduceLROnPlateau (reduces LR when stuck)
  - Reduces by 50% after 3 epochs without improvement

### Section 9: Training Functions
**Purpose**: Implements core training and validation logic.

**`train_epoch()`**:
- Sets model to training mode
- Forward pass: images → predictions
- Backward pass: calculates gradients
- Updates weights with optimizer
- Tracks loss and accuracy
- Shows progress bar

**`validate_epoch()`**:
- Sets model to evaluation mode
- Disables gradient computation (saves memory)
- Forward pass only (no weight updates)
- Tracks validation metrics
- Helps detect overfitting

### Section 10: Main Training Loop
**Purpose**: Orchestrates complete training process.
- Trains for up to 20 epochs
- Each epoch: train → validate → update LR if needed
- Saves best model (lowest validation loss)
- Early stopping: stops if no improvement for 5 epochs
- Loads best weights at end
- Returns trained model and training history

### Section 11: Execute Training
**Purpose**: Runs the training process.
- Calls `train_model()` with all components
- Displays progress bars for each epoch
- Shows loss/accuracy metrics
- Saves best model automatically
- Typical duration: 8-15 epochs (with early stopping)

### Section 12: Plot Training History
**Purpose**: Visualizes training progress.
- **Loss plot**: Shows train/valid loss over epochs
- **Accuracy plot**: Shows train/valid accuracy over epochs
- Helps diagnose:
  - Overfitting (train >> valid accuracy)
  - Underfitting (both accuracies low)
  - Good training (both improve together)
- Identifies best epoch (lowest validation loss)

### Section 13: Evaluate on Test Set
**Purpose**: Tests model on completely unseen data.
- Processes all test images
- Collects predictions and probabilities
- Calculates final test accuracy
- Test set was never used during training/validation
- Provides unbiased performance estimate

### Section 14: Classification Report
**Purpose**: Generates detailed per-class metrics.
- **Precision**: Accuracy of positive predictions
- **Recall**: Percentage of actual positives found
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class
- Macro/Weighted averages
- Shows which art styles are easy/hard to classify

### Section 15: Confusion Matrix
**Purpose**: Visualizes prediction patterns.
- Rows: True labels, Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Heat map shows counts with color intensity
- Per-class accuracy breakdown
- Reveals which styles confuse the model

### Section 16: Visualize Predictions
**Purpose**: Shows sample predictions with images.
- Displays 12 random test images in grid
- Shows predicted vs true class
- Displays confidence percentage
- Green title = correct, Red = incorrect
- Helps understand model behavior visually

### Section 17: Inference Function
**Purpose**: Provides reusable prediction function.
- `predict_single_image()`: Classifies any image
- Loads, preprocesses, and predicts
- Shows image with prediction
- Displays probability distribution bar chart
- Production-ready for deployment

### Section 18: Example Prediction
**Purpose**: Demonstrates inference on random test image.
- Randomly selects one test image
- Shows filename and true class
- Makes prediction and displays results
- Run multiple times to test different images

### Section 19: Load Saved Model
**Purpose**: Template for loading model in new session.
- Creates same architecture
- Loads saved weights from .pth file
- Sets to evaluation mode
- Enables model reuse without retraining

### Section 20: Summary
**Purpose**: Provides overview and next steps.
- Summarizes all features
- Lists model performance
- Suggests improvements:
  - Try different architectures
  - Fine-tune more layers
  - Adjust hyperparameters
  - Collect more data

## 🎯 Key Concepts Explained

### Transfer Learning
Using a pre-trained model (ResNet50 trained on ImageNet) and adapting it to our art classification task. Much faster and more accurate than training from scratch.

### Data Augmentation
Artificially increasing dataset variety by applying random transformations (flips, rotations, color changes). Helps model generalize better.

### Early Stopping
Stops training when validation performance stops improving, preventing overfitting and saving time.

### Learning Rate Scheduling
Automatically reduces learning rate when training plateaus, helping the model converge better.

### Batch Processing
Processing multiple images together (batch size 32) for efficiency and stable gradient updates.

### Validation vs Test Sets
- **Validation**: Used during training to tune model and decide when to stop
- **Test**: Completely unseen data, used only at the end for final evaluation

## 📊 Expected Performance

With proper training, expect:
- **Training accuracy**: 85-95%
- **Validation accuracy**: 75-85%
- **Test accuracy**: 70-85%

Gap between train and valid/test accuracy is normal due to data augmentation and generalization challenges.

## 🚀 How to Use

1. **Run cells sequentially** from top to bottom
2. **Training time**: ~10-30 min/epoch on CPU, ~1-3 min/epoch on GPU
3. **Best model saved** as `best_art_classifier.pth`
4. **Experiment**:
   - Adjust CONFIG parameters
   - Try different augmentations
   - Modify model architecture
   - Add more data if available

## 🔧 Troubleshooting

- **Out of memory**: Reduce batch_size to 16 or 8
- **Training too slow**: Use GPU or reduce image_size to 128
- **Low accuracy**: Train more epochs, add more data, or try different model
- **Overfitting**: Increase dropout, add more augmentation, or collect more data

## 📝 Files Generated

- `best_art_classifier.pth`: Saved model weights
- Training produces various plots and metrics during execution

---

**Happy Training!** 🎨🤖


