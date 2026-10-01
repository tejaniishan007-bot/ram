# Deep Learning PR 3 — Complete Submission

**Institute:** Red & White Skill Education  
**Subject:** Deep Learning  
**Project:** PR 3  
**Total Marks:** 10

## Project Overview

This project implements a complete computer-vision pipeline that moves from classical image processing to modern deep-learning detection:

**Morphological Operations → Bitwise Operations & Histograms → YuNet Face Detection → YOLOv8 Object Detection → Integrated Pipeline**

The notebook is designed to run in **Jupyter Notebook or VS Code** and uses Python.

## Topics Covered

### Task 1 — Morphological Operations
- Environment setup and image I/O
- Grayscale conversion and binary thresholding
- Erosion
- Dilation
- Opening
- Closing
- Kernel shape and size comparison
- Morphology explanation and real-world use case

### Task 2 — Bitwise Operations & Histograms
- Bitwise AND
- Bitwise OR
- Bitwise XOR
- Bitwise NOT
- ROI masking and overlay
- Grayscale histogram
- Blue/Green/Red channel histograms
- Brightness and contrast adjustment
- Histogram-based image-quality analysis

### Task 3 — Face Detection with YuNet
- YuNet model loading
- Static-image face detection
- Face bounding boxes
- Five facial landmarks
- Confidence-score threshold experiment
- Privacy/anonymisation using Gaussian blur
- Optional real-time webcam detection
- Explanation of why YuNet is suitable for lightweight face detection

### Task 4 — Object Detection with YOLOv8
- YOLOv8 Nano pretrained model
- Static-image object detection
- Bounding boxes, class names and confidence scores
- Optional real-time webcam detection
- Confidence threshold tuning
- IoU threshold tuning
- Non-Max Suppression discussion
- Per-class detection summary
- Explanation of YOLO's single-stage design

### Task 5 — Integrated Pipeline
- YuNet + YOLOv8 combined pipeline
- Optional morphological pre-cleaning
- 100-frame FPS benchmark
- Final comparison table
- Reflection and deployment recommendations

## Folder Structure

```text
CV_PR3_Deep_Learning_PR3/
│
├── CV_PR3_Deep_Learning.ipynb
├── README.md
├── requirements.txt
│
└── data/
    ├── images/
    │   └── .gitkeep
    ├── models/
    │   └── .gitkeep
    └── outputs/
        └── .gitkeep
```

During execution, the notebook automatically downloads the public demo images and YuNet model into `data/images/` and `data/models/`. Generated result images and CSV files are saved in `data/outputs/`.

## Requirements

Recommended:
- Python 3.11+
- Jupyter Notebook or VS Code
- Internet connection for the first run so public demo images and model weights can be downloaded
- Webcam for optional real-time webcam tasks

Install all packages from the project folder:

```bash
pip install -r requirements.txt
```

### Important OpenCV note

The project uses `opencv-contrib-python`, which provides the normal `cv2` functionality as well as OpenCV's extra/contrib modules. Therefore, the requirements file intentionally does **not** install both `opencv-python` and `opencv-contrib-python` at the same time.

## How to Run

### 1. Open the project

Extract the ZIP file and open the folder in VS Code or Jupyter.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
CV_PR3_Deep_Learning.ipynb
```

### 4. Run cells from top to bottom

The notebook will:
1. Create the required `data/` directories.
2. Generate synthetic images for morphology and bitwise experiments.
3. Download OpenCV's public `lena.jpg` sample when required.
4. Download the YuNet face detector ONNX model when required.
5. Download Ultralytics' `bus.jpg` sample when required.
6. Download/load YOLOv8 Nano when required.
7. Save generated results under `data/outputs/`.

## Optional Webcam Tasks

The webcam cells are intentionally provided as optional cells.

If a webcam is available, uncomment and run the appropriate function:

```python
run_yunet_webcam(0)
```

or:

```python
run_yolo_webcam(0, conf=0.25)
```

Press **q** to stop the OpenCV window.

If a webcam is unavailable, the static-image experiments can still be completed.

## Models and Data Sources

The notebook uses:
- OpenCV public sample image: `lena.jpg`
- Ultralytics public sample image: `bus.jpg`
- YuNet face detector: `face_detection_yunet_2023mar.onnx`
- YOLOv8 Nano: `yolov8n.pt`

These are downloaded automatically by the notebook rather than being unnecessarily bundled into the submission ZIP.

## Expected Outputs

After running the notebook, `data/outputs/` may contain files such as:

```text
01_binary.png
02_erosion.png
03_dilation.png
04_opening.png
05_closing.png
06_yunet_face_detection.jpg
07_yolov8_static.jpg
08_integrated_pipeline.jpg
final_comparison.csv
```

Additional output files may be created depending on which optional cells are executed.

## Key Findings

### Morphology
Opening is useful for removing small bright noise, while closing is useful for filling small holes and gaps. Kernel size controls the strength of the operation.

### Bitwise Operations
Bitwise operations are useful for creating masks and extracting regions of interest before further image processing.

### Histograms
Histograms provide a quick diagnostic of brightness, contrast and channel intensity distribution.

### YuNet
YuNet is a lightweight learned face detector that provides bounding boxes, confidence scores and five facial landmarks. It is suitable for real-time or edge-oriented face detection.

### YOLOv8
YOLOv8 is a single-stage detector capable of detecting multiple COCO object classes in one forward pass. YOLOv8 Nano provides a useful speed/accuracy trade-off for lightweight applications.

### Integrated System
A practical smart-surveillance or retail-analytics system can combine:
- classical preprocessing when needed,
- YuNet for faces,
- YOLOv8 for general objects.

The exact FPS and detection counts depend on the computer hardware, input resolution, lighting and model settings.

## Submission Checklist

Before submission:

- [ ] Run all required notebook cells.
- [ ] Confirm there are no Python errors.
- [ ] Confirm `data/outputs/` contains generated results.
- [ ] Check that the FPS benchmark has been executed.
- [ ] Check the final comparison table.
- [ ] Save the completed notebook.
- [ ] Record the required 5–10 minute explanation video.
- [ ] Upload the notebook, HTML/plots/clips as required by the institute.
- [ ] Include this README and `requirements.txt`.

## PR 3 Marking Alignment

| Component | Marks | Submission Content |
|---|---:|---|
| A. Technical Completion | 5 | Complete Jupyter Notebook covering Tasks 1–5 |
| B. Recorded Video | 3 | Face + screen explanation of concepts and results |
| C. GitHub Repository | 2 | Notebook + README + outputs/plots/clips |

**Total: 10 Marks**

---

**Project:** Deep Learning PR 3  
**Institute:** Red & White Skill Education
