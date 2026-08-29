# DL_PR1 — Breast Cancer Classification with Deep Learning

A single Jupyter notebook (`DL_PR1.ipynb`) that builds and compares a progression of neural network models — from a single-layer perceptron up to a regularized, dropout-protected, early-stopped MLP — on the Wisconsin Breast Cancer dataset. Every task opens with a short Markdown cell explaining *why* the technique is used, not just how to code it.

## Dataset
`sklearn.datasets.load_breast_cancer` — 569 samples, 30 numeric features (cell nucleus measurements: radius, texture, perimeter, area, smoothness, concavity, etc.), binary target (0 = Malignant, 1 = Benign).

## Repository contents
| File | Description |
|---|---|
| `DL_PR1.ipynb` | The full notebook — restart-kernel-and-run-all clean, no errors |
| `DL_PR1.html` | Exported HTML version of the executed notebook |
| `requirement--



s.txt` | Pinned dependencies to reproduce the environment |
| `README.md` | This file |


## Notebook structure

**Task 1 — Data Loading, EDA & Preprocessing**
Loads the dataset, plots class distribution (63% Benign / 37% Malignant — mild imbalance, addressed via stratified split + precision/recall tracking rather than resampling), plots a feature correlation heatmap (reveals strong redundancy among `radius`/`perimeter`/`area`), splits train/test (stratified, 80/20), and applies `StandardScaler` (fit on train only).

**Task 2 — Single-Layer Perceptron (baseline)**
One neuron, 31 parameters — mathematically equivalent to logistic regression. Establishes the accuracy floor and shows its inherent linear-boundary limitation.

**Task 3 — Multi-Layer Perceptron & Activation Functions**
Dense(64)→Dense(32)→Dense(1) MLP compared across ReLU / Tanh / Sigmoid hidden activations. ReLU converges fastest with the least noise, confirmed as the best fit for this architecture.

**Task 4 — Early Stopping**
A wider, overfitting-prone MLP (128→64) trained for a 300-epoch budget with and without `EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)`, showing the divergence point between training and validation loss.

**Task 5 — Dropout**
Dropout(0.3) after each hidden layer (verified to add zero trainable parameters), trained with Early Stopping, plus a rate comparison across 0.1 / 0.3 / 0.5 to show the regularization-strength trade-off.

**Task 6 — Regularization (L1, L2, Elastic-Net)**
Three identical architectures differing only in `kernel_regularizer`: L1 (sparsity/feature selection), L2 (shrinkage, better suited to the correlated feature groups from Task 1), and L1_L2 combined. Compared on shared loss-axis scale and by train-validation gap.

**Task 7 — Final Model, Results Table & Clinical Insight**
A combined Dropout + L2 + Early Stopping model, a full results table across every model built, and a written recommendation on model choice, classification threshold, and which technique most improved generalization — framed around the cost asymmetry of false negatives in cancer diagnosis.

## Results

| Model | Regularization | Dropout | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| SLP (Task 2) | None | None | 0.9649 | 0.9722 | 0.9722 | 0.9722 |
| MLP-best / ReLU (Task 3) | None | None | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| MLP + Early Stopping (Task 4) | None | None | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| MLP + Dropout (Task 5) | None | 0.3 | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| L1 Regularized MLP (Task 6) | L1 (0.001) | None | 0.9649 | 0.9857 | 0.9583 | 0.9718 |
| L2 Regularized MLP (Task 6) | L2 (0.001) | None | 0.9386 | 0.9851 | 0.9167 | 0.9496 |
| **L1-L2 Regularized MLP (Task 6)** | L1_L2 (0.001/0.001) | None | **0.9737** | 0.9859 | **0.9722** | **0.9790** |
| Final Combined Model (Task 7.1) | L2 (0.001) | 0.3 | 0.9561 | 0.9855 | 0.9444 | 0.9645 |

The **L1-L2 (Elastic-Net) model** achieved the best test accuracy and F1-score in this run, combining L1's sparsity with L2's handling of correlated features. Note results vary slightly run-to-run due to weight initialization and the stochastic nature of training.

## Video
5–10 min walkthrough covering all 7 tasks, model summaries, live training curves, and verbal explanation of each technique.

📹 **Video:** _paste your Google Drive / YouTube link here_

## Key takeaways
- Feature scaling is required before gradient-based training due to large scale disparities across the 30 features.
- Hidden layers let the network learn non-linear feature interactions the SLP structurally cannot.
- ReLU converges faster and avoids vanishing gradients relative to Tanh/Sigmoid in hidden layers.
- Early Stopping (`monitor='val_loss'`, `restore_best_weights=True`) and Dropout were the most effective techniques for controlling overfitting on this small (569-row) dataset.
- L1 induces sparsity (implicit feature selection); L2 shrinks correlated weights together; Elastic-Net combines both and performed best overall here.
- For clinical deployment, recall on the malignant class should be prioritized over raw accuracy, and the default 0.5 threshold should likely be lowered to reduce false negatives.
