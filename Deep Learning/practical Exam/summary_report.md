# Deep Learning Practical Exam — Set C: Summary Report

## Tasks and datasets
- **ANN:** Wine Recognition dataset (178 observations, 13 features, 3 classes).
- **OpenCV:** five sample images; HSV thresholding, morphology, Sobel edges, Haar-cascade face detection.
- **CNN:** MNIST subset (20,000 training samples and 4,000 test samples), compared with a dense ANN.
- **RNN:** daily minimum temperatures (1981–1990), 30-day lookback and GRU compared with naive persistence.
- **Transformers:** pretrained fill-mask, extractive QA, and text-generation pipelines.

## Results
| Step / Topic         | Task type                   | Dataset used                           | Key metric                         | Result achieved                                                       |   Time taken (min) |
|:---------------------|:----------------------------|:---------------------------------------|:-----------------------------------|:----------------------------------------------------------------------|-------------------:|
| ANN (Wine)           | Multi-class classification  | Wine: 178 rows, 13 features, 3 classes | Accuracy                           | 0.9167                                                                |               0.2  |
| OpenCV               | Preprocessing / detection   | 5 sample images                        | Faces detected                     | {'astronaut': 1, 'coffee': 0, 'chelsea': 0, 'rocket': 0, 'hubble': 0} |             nan    |
| CNN (MNIST)          | Digit classification        | 20,000 train / 4,000 test samples      | Accuracy                           | 0.9765                                                                |               0.3  |
| ANN baseline (MNIST) | Digit classification        | 20,000 train / 4,000 test samples      | Accuracy                           | 0.9563                                                                |               0.17 |
| GRU (Temperature)    | Time-series regression      | Daily minimum temperatures, 1981–1990  | RMSE / MAE (°C)                    | RMSE=2.218; MAE=1.744                                                 |               0.26 |
| Transformers         | Fill-mask / QA / generation | India-related text prompts             | Manual QA correctness / confidence | 0 QA examples                                                         |             nan    |

## What I learned
The ANN provides a baseline for tabular classification. OpenCV demonstrates that classical image processing can isolate colour ranges and edges without model training. The CNN uses spatial structure and learned filters for digit recognition. The GRU uses gated recurrent memory for time-series prediction and should be judged against a simple baseline. Transformer pipelines are powerful but can produce plausible yet incorrect output.

## Next steps
I would improve hyperparameter selection with a dedicated validation strategy, evaluate multiple random seeds, add more varied face images, compare GRU with LSTM and additional forecasting baselines, and use retrieval plus human verification for factual Transformer answers.

## Reproducibility note
Replace any "Not run" entries with actual results after running all notebook cells. Training time and metrics depend on hardware, package versions, and network conditions.
