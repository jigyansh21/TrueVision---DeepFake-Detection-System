# TrueVision — DeepFake Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Production-ready deepfake video screening powered by hybrid CNN–RNN models**

</div>

---

## Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
- [Operation Guide](#operation-guide)
- [Project Layout](#project-layout)
- [Models & Training](#models--training)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Limitations & Roadmap](#limitations--roadmap)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [License & Support](#license--support)

---

## Overview

TrueVision is a deep learning system that detects manipulated (deepfake) videos by combining frame-level convolutional encoders with sequence-aware GRU layers. The pipeline ingests uploaded clips, isolates faces, extracts discriminative embeddings, and classifies each video as **REAL** or **FAKE** in under a minute for a typical 10-second sequence. The solution is packaged as a Flask web application with an opinionated training stack for reproducibility.

---

## Key Capabilities

- **Automated Video Screening** – Upload MP4/AVI/MOV assets and receive binary authenticity verdicts.
- **Hybrid CNN–RNN Modeling** – EfficientNetB2 or InceptionV3 feature extractors coupled with GRU layers for temporal reasoning.
- **Operational Accuracy** – ~85 % accuracy on DFDC samples with strong precision on FAKE predictions.
- **Responsive UI** – Streamlined dashboard for file uploads, processing status, and classification results.
- **Multi-model Support** – Plug-and-play with alternative CNN backbones or retrained weights.
- **Deployment-focused** – Includes pre-built virtual-environment guidance, model storage, and inference scripts.

---

## System Architecture

1. **Ingestion** – Video file uploads through the Flask interface or API endpoints.
2. **Frame Processing** – Frame extraction followed by OpenCV-based face localization and cropping.
3. **Feature Extraction** – TimeDistributed EfficientNetB2/InceptionV3 generates frame embeddings.
4. **Temporal Modeling** – GRU layers capture cross-frame dynamics; dropout mitigates overfitting.
5. **Classification** – Dense layers output REAL/FAKE probabilities with softmax activation.
6. **Presentation** – Results rendered in the UI with associated metadata and logs.

> Pre-processing flow: `Video → Frames → Face Detection → Cropping → CNN Embeddings`  
> Prediction flow: `Embeddings → GRU Sequence Modeling → Dense Classification → Verdict`

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- CUDA-enabled GPU (recommended)

### Installation

```bash
git clone https://github.com/jigyansh21/TrueVision---DeepFake-Detection-System.git
cd TrueVision---DeepFake-Detection-System/Deploy
pip install -r requirements.txt
```

> Troubleshooting dlib: install the matching wheel from `Deploy/Dlib-python whl packages/` if the default build fails.

### Model Weights

Ensure `Deploy/models/inceptionNet_model.h5` is present. Replace with retrained weights if needed (see `Model_Training_Colab.ipynb`).

---

## Operation Guide

1. Start the server:
   ```bash
   cd Deploy
   python app.py
   ```
2. Navigate to `http://localhost:5000`.
3. Upload a video (≤16 MB recommended) and select **Analyze**.
4. Review the REAL/FAKE classification and console logs for troubleshooting.

Supported formats: MP4 (preferred), AVI, MOV, or any format compatible with OpenCV. For best accuracy, provide well-lit videos with a clear view of a single face.

---

## Project Layout

```
TrueVision---DeepFake-Detection-System/
├── Deploy/                     # Flask app, inference scripts, dependencies
│   ├── app.py
│   ├── models/
│   │   └── inceptionNet_model.h5
│   ├── static/ and templates/  # UI assets
│   └── Dlib-python whl packages/
├── Pre-Processing/             # Face extraction & cleaning notebooks
├── Model Training/             # Training descriptors
├── Model_Training_Colab.ipynb  # Colab notebook for retraining
├── Saved Models/               # Additional CNN-RNN checkpoints
└── README.md
```

---

## Models & Training

- **CNN-only baselines**: MesoNet, ResNet50, EfficientNetB0 (frame classification).
- **Hybrid models**:
  - *InceptionV3 + GRU*: ~82 % accuracy; excels on single-face clips.
  - *EfficientNetB2 + GRU*: ~85 % accuracy; preferred production model.

Training configuration (see Colab notebook):
- Frames per video: 30
- Batch size: 8 (adjust by GPU memory)
- Optimizer: Adam (1e-4)
- Loss: Sparse categorical cross-entropy
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

---

## Dataset

Reference dataset: [DeepFake Detection Challenge (DFDC) – Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

- Training set: 400 MP4 videos
- Test set: 400 MP4 videos
- Metadata fields: `filename`, `label` (0=REAL,1=FAKE), `original`, `split`

Replace these samples with your internal corpus as needed; ensure metadata follows the same schema.

---

## Evaluation

| Model                | Architecture | Accuracy | Notes                                   |
|----------------------|-------------|----------|-----------------------------------------|
| MesoNet              | CNN         | Low      | Image-focused, weak temporal reasoning   |
| ResNet50             | CNN         | Moderate | Frame-level only                         |
| EfficientNetB0       | CNN         | Moderate | Frame-level only                         |
| InceptionV3 + GRU    | CNN-RNN     | ~82 %    | Sensitive to multi-face sequences        |
| **EfficientNetB2 + GRU** | **CNN-RNN** | **~85 %** | **Preferred deployment model**           |

- Processing time: ≈60 s for a 10 s clip (30 fps) on GPU
- Precision: High for FAKE class; monitor recall on low-light footage

---

## Limitations & Roadmap

1. **Multiple faces** – Accuracy drops when several faces appear simultaneously.
2. **Low-light / occlusions** – Detection quality degrades on poorly lit videos.
3. **Video quality** – Heavy compression reduces feature fidelity.
4. **Streaming** – Current implementation handles uploaded files only; no live ingestion.

Planned enhancements: MediaPipe face detection, ONNX/TFLite export for edge inference, and multi-face consensus logic.

---

## Technology Stack

- **Backend**: Python 3.9+, Flask 2.2, TensorFlow/Keras, NumPy, OpenCV, imageio
- **Frontend**: HTML5, CSS3, vanilla JavaScript
- **ML Ops**: Google Colab training notebook, SavedModel/H5 artifacts, optional dlib wheels

---

## Contributing

Contributions are welcome via pull requests. Please open an issue to discuss significant changes before implementation.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit with descriptive messages.
4. Push and open a pull request against `main`.

---

## Maintainers

- **Jigyansh** – Lead Developer
- **Nikhil Joshi** – Co-developer

---

## License & Support

Licensed under the MIT License. For questions or support, open an issue in this repository or contact the maintainers via GitHub.

<div align="center">

If TrueVision helps your workflow, please consider starring the project.

</div>
