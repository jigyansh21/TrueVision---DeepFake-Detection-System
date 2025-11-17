# 🔍 DeepFake Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An advanced deep learning system for detecting deepfake videos using CNN-RNN hybrid architecture**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Dataset](#-dataset)
- [Results](#-results)
- [Limitations](#-limitations)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

DeepFake Detection System is a machine learning project that identifies manipulated videos created using deepfake technology. The system uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to analyze video frames and classify them as **REAL** or **FAKE** with high accuracy.

### What are DeepFakes?

DeepFakes are synthetic media created using artificial intelligence, where a person's face is replaced with someone else's face in an existing image or video. These are typically generated using Generative Adversarial Networks (GANs) and pose significant risks including:

- 🚨 **Fake News**: Spreading misinformation through manipulated videos
- 🎭 **Celebrity Impersonation**: Creating unauthorized content featuring public figures
- 💰 **Financial Fraud**: Using deepfakes for fraudulent activities
- 🗳️ **Political Manipulation**: Influencing public opinion with fake political content

---

## ✨ Features

- 🎥 **Video Analysis**: Upload and analyze video files for deepfake detection
- 🧠 **Hybrid Architecture**: Combines CNN for feature extraction and RNN for temporal analysis
- 📊 **High Accuracy**: Achieves ~85% test accuracy on DFDC dataset
- 🚀 **Real-time Processing**: Fast analysis (approximately 1 minute for a 10-second 30fps video)
- 🎨 **Modern Web Interface**: User-friendly dashboard for video upload and analysis
- 🔍 **Frame-by-Frame Analysis**: Detects subtle imperfections in facial features
- 📈 **Multiple Model Support**: Supports various CNN architectures (EfficientNet, InceptionV3, ResNet)

---

## 🏗️ Architecture

### System Pipeline

The detection pipeline consists of the following steps:

1. **Video Loading**: Load input video file
2. **Frame Extraction**: Extract all frames from the video
3. **Face Detection**: Identify and crop face regions from each frame
4. **Feature Extraction**: Use pre-trained CNN models to extract features
5. **Temporal Analysis**: Apply RNN (GRU) to analyze temporal patterns
6. **Classification**: Classify video as REAL or FAKE

### Model Architecture

#### CNN-RNN Hybrid Model

The best-performing model combines:

- **CNN Backbone**: EfficientNetB2 or InceptionV3 (pre-trained on ImageNet)
- **RNN Layer**: GRU (Gated Recurrent Unit) for sequence modeling
- **Output**: Binary classification (REAL/FAKE)

**Key Hyperparameters:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy

### Workflow Diagrams

#### Pre-processing Workflow
```
Video Input → Frame Extraction → Face Detection → Face Cropping → Feature Extraction
```

#### Prediction Workflow
```
Extracted Features → CNN Feature Vectors → GRU Sequence Analysis → Classification → Result
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- GPU (recommended for faster processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/DeepFake_Detection.git
cd DeepFake_Detection-main
```

### Step 2: Install Dependencies

```bash
cd Deploy
pip install -r requirements.txt
```

**Note**: The original `requirments.txt` file has been corrected to `requirements.txt` with proper package versions.

**Note**: If you encounter issues with `dlib`, install the appropriate wheel file from `Deploy/Dlib-python whl packages/` based on your Python version:

```bash
pip install Dlib-python\ whl\ packages/dlib-19.22.99-cp39-cp39-win_amd64.whl
```

### Step 3: Download Model Weights

Ensure the model file `inceptionNet_model.h5` is present in `Deploy/models/` directory.

---

## 💻 Usage

### Running the Application

1. **Navigate to the Deploy directory:**
   ```bash
   cd Deploy
   ```

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Access the web interface:**
   - Open your browser and navigate to `http://localhost:5000`
   - Upload a video file (MP4 format recommended)
   - Click "Analyze" to process the video
   - View the classification result (REAL or FAKE)

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- Other formats supported by OpenCV

### Video Requirements

- Maximum file size: 16 MB (configurable)
- Recommended: Videos with clear face visibility
- Best results: Single face per frame, well-lit environment

---

## 📁 Project Structure

```
DeepFake_Detection-main/
│
├── Deploy/                    # Deployment files
│   ├── app.py                 # Flask application
│   ├── models/                # Trained model files
│   │   └── inceptionNet_model.h5
│   ├── static/                # Static files
│   │   └── uploads/             # Uploaded videos
│   ├── templates/             # HTML templates
│   │   ├── upload.html        # Main dashboard
│   │   └── upload.css         # Stylesheet
│   └── requirements.txt       # Python dependencies
│
├── Pre-Processing/            # Data preprocessing notebooks
│   └── DeepFake_Detection_Pre-Processing.ipynb
│
├── Model Training/            # Model training scripts
│
├── Saved Models/              # Additional saved models
│   └── models/
│       └── CNN_RNN/
│
├── Dataset/                   # Dataset information
│   └── Readme.md
│
└── README.md                  # This file
```

---

## 🤖 Models

### Tested Architectures

#### 1. CNN-Only Models

**MesoNet**
- Pre-trained for deepfake image detection
- Limited performance on video frames

**ResNet50**
- Trained on deepfake images cropped from videos
- ImageNet pre-trained weights

**EfficientNetB0**
- Trained on deepfake images cropped from videos
- ImageNet pre-trained weights

#### 2. CNN-RNN Hybrid Models

**InceptionV3 + GRU**
- Test Accuracy: ~82%
- Good performance on single-face videos
- Limitation: Struggles with multiple faces

**EfficientNetB2 + GRU** ⭐ (Best Performance)
- Test Accuracy: ~85%
- Excellent feature extraction capabilities
- Limitation: Reduced accuracy in dark backgrounds

---

## 📊 Dataset

The project uses the **DeepFake Detection Challenge (DFDC) Dataset** from Kaggle.

- **Dataset Link**: [DFDC on Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
- **Training Data**: 400 videos (MP4 format)
- **Test Data**: 400 videos (MP4 format)

### Dataset Metadata

- `filename`: Video filename
- `label`: Classification label (0 = REAL, 1 = FAKE)
- `original`: Original video name (for FAKE videos)
- `split`: Dataset split identifier

---

## 📈 Results

### Performance Metrics

- **Test Accuracy**: ~85% (EfficientNetB2 + GRU)
- **Processing Time**: ~1 minute for 10-second video (30 fps)
- **Precision**: High precision for FAKE video detection

### Model Comparison

| Model | Architecture | Accuracy | Notes |
|-------|-------------|----------|-------|
| MesoNet | CNN | Low | Good for images, poor for videos |
| ResNet50 | CNN | Moderate | Frame-level analysis |
| EfficientNetB0 | CNN | Moderate | Frame-level analysis |
| InceptionV3 + GRU | CNN-RNN | ~82% | Good for single-face videos |
| **EfficientNetB2 + GRU** | **CNN-RNN** | **~85%** | **Best overall performance** |

---

## ⚠️ Limitations

1. **Multiple Faces**: Reduced accuracy when multiple faces are present in the video
2. **Dark Backgrounds**: Difficulty detecting faces in poorly lit environments
3. **Video Quality**: Performance may degrade with low-resolution or heavily compressed videos
4. **Real-time Processing**: Current implementation requires video upload (not real-time streaming)

---

## 🛠️ Technologies Used

### Backend
- **Python 3.9+**: Core programming language
- **Flask 2.2**: Web framework
- **TensorFlow**: Deep learning framework
- **OpenCV (cv2)**: Computer vision and video processing
- **NumPy**: Numerical computations
- **face_recognition**: Face detection and recognition
- **imageio**: Image and video I/O

### Frontend
- **HTML5**: Markup language
- **CSS3**: Styling
- **JavaScript**: Client-side interactivity

### Machine Learning
- **TensorFlow/Keras**: Model training and inference
- **Pre-trained Models**: EfficientNet, InceptionV3, ResNet
- **GRU**: Recurrent neural network layer

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Team

This project was developed by:

1. **[Balaji Kartheek](https://github.com/Balaji-Kartheek)** - Project Lead
2. **[Aaron Dsouza](https://github.com/DsouzaAaron)** - Co-developer

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- DeepFake Detection Challenge (DFDC) for providing the dataset
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- All contributors and open-source libraries used in this project

---

## 📞 Contact & Support

For questions, issues, or contributions, please open an issue on GitHub or contact the project maintainers.

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ by the DeepFake Detection Team

</div>
