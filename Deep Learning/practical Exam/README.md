# Deep Learning Practical Exam — Set C

## Project overview
This repository contains a reproducible notebook covering five mini-projects:
1. ANN multi-class classification on the Wine dataset.
2. OpenCV colour-space conversion, morphology, Sobel edges, and Haar-cascade face detection.
3. CNN digit recognition on MNIST compared with an ANN baseline.
4. GRU forecasting for daily minimum temperatures compared with a naive persistence baseline.
5. Hugging Face pipelines for fill-mask, question answering, and text generation.

## Files
- `DeepLearning_Practical_Exam_SetC.ipynb` — complete notebook.
- `ann_wine_model.h5` — trained Wine ANN.
- `cnn_mnist_model.h5` — trained MNIST CNN.
- `ann_mnist_baseline.h5` — trained MNIST ANN baseline.
- `rnn_temperature_model.h5` — trained temperature GRU.
- `outputs/` — OpenCV face-detection images and Transformer result CSVs (generated when cells run).
- `summary_results.csv` — metrics collected by the notebook.
- `summary_report.md` — report template/results summary.
- `requirements.txt` — Python dependencies.

## How to run
1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Start Jupyter: `jupyter notebook`.
5. Open `DeepLearning_Practical_Exam_SetC.ipynb` and run cells from top to bottom.

Internet access is required to download the temperature CSV, MNIST data (on first use), and pretrained Hugging Face models. TensorFlow and Transformer models may take several minutes to download/train.

## Notes
- Metrics vary with software versions, hardware, and training randomness; report the values produced by your run.
- The notebook does not fine-tune pretrained Transformer models.
- Verify model-generated answers before using them as factual claims.
