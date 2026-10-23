# Project A — CNN from Scratch: CIFAR-10 Image Classifier

## 1. Project Information
- **Subject:** Deep Learning — Practical Report 4
- **Dataset:** CIFAR-10
- **Framework:** Python, TensorFlow, Keras
- **Notebook:** `Project_A_CNN_CIFAR10_FAST_RUN.ipynb`
- **Tasks:** A1 to A6

## 2. Project Overview
This project implements a CNN-based image classification workflow using the CIFAR-10 dataset. It covers preprocessing, convolution, pooling, CNN architecture design, model training, data augmentation, transfer learning, and residual connections.

CIFAR-10 contains 60,000 RGB images of size 32 × 32, belonging to 10 classes.

## 3. Tasks Covered
- **A1:** Dataset loading, EDA, normalisation and one-hot encoding.
- **A2:** Convolution, edge detection, padding, strides and feature maps.
- **A3:** MaxPooling, AveragePooling, Flatten, LeNet-5-inspired architecture and custom CNN.
- **A4:** Backpropagation, model training, training curves, evaluation and confusion matrix.
- **A5:** Image augmentation using ImageDataGenerator and Keras preprocessing layers.
- **A6:** MobileNetV2 transfer learning, feature extraction, fine-tuning, model comparison and residual CNN.

## 4. Requirements

Install the required libraries:

```bash
pip install tensorflow matplotlib seaborn scikit-learn pandas numpy jupyter
```

## 5. How to Run
1. Open the notebook in Jupyter Notebook or VS Code.
2. Run the cells sequentially from top to bottom.
3. Allow Keras to download CIFAR-10 when prompted by the dataset-loading cell.
4. Review the visualisations and evaluation results.
5. Save the notebook with the generated outputs.

## 6. Fast-Run Configuration
The fast-run notebook uses:

```python
QUICK_RUN = True
EPOCHS = 2
BATCH_SIZE = 64
```

Quick mode uses 10,000 training images and 2,000 test images to reduce execution time. It skips MobileNetV2 training and fine-tuning.

For a full run, set `QUICK_RUN = False`. Training may take considerably longer on a CPU.

## 7. Evaluation
The notebook includes:
- Training and validation accuracy/loss plots
- Test accuracy and loss
- Classification report
- Confusion matrix
- Misclassified image examples
- Model comparison table

Report the actual results obtained when running the notebook. Do not use assumed accuracy values.

## 8. Conclusion
This practical demonstrates the complete CNN image-classification workflow, from preprocessing and convolution operations to model evaluation, augmentation, transfer learning and residual connections.
