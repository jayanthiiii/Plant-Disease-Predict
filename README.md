# 🌿 PlantGuard AI — Plant Disease Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

**Real-time plant disease classification using EfficientNetB0 & MobileNetV3 with GradCAM explainability**

</div>

---

## 🔍 Overview

PlantGuard AI is a deep learning pipeline for automated plant disease detection from leaf images. It supports **38 disease classes across 14 crop species** using transfer learning on the PlantVillage dataset.

| Model | Params | Val Accuracy | Inference |
|---|---|---|---|
| EfficientNetB0 | 5.3 M | ~96.4% | ~120 ms |
| MobileNetV3Small | 3.0 M | ~94.1% | ~55 ms |

---

## 🗂 Project Structure

```
plant-disease-detection/
├── app.py                  # Streamlit web application
├── train.py                # Two-phase fine-tuning script
├── evaluate.py             # Evaluation + confusion matrix
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── predictor.py        # DiseasePredictor class (inference + GradCAM)
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # Image normalisation, TTA
│   ├── visualization.py    # Plotly charts, GradCAM overlay
│   └── disease_info.py     # 38-class metadata & treatment info
├── data/
│   └── sample_images/      # Place test images here
├── weights/                # Saved .h5 model weights (gitignored)
├── notebooks/
│   └── exploration.ipynb
└── tests/
    └── test_predictor.py
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
```

### 2. Run the web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a leaf image, and click **Analyze Disease**.

---

## 🏋️ Training

### Download the dataset

Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract it to `data/plantvillage/`.

```
data/plantvillage/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Tomato___Late_blight/
└── ...
```

### Fine-tune EfficientNetB0

```bash
python train.py \
  --model EfficientNetB0 \
  --data_dir data/plantvillage \
  --epochs 30 \
  --batch_size 32 \
  --unfreeze_at 10
```

### Fine-tune MobileNetV3 (faster, edge-friendly)

```bash
python train.py \
  --model MobileNetV3 \
  --data_dir data/plantvillage \
  --epochs 30 \
  --lr 5e-5
```

Trained weights are saved to `weights/`.

---

## 📊 Evaluation

```bash
python evaluate.py \
  --model EfficientNetB0 \
  --data_dir data/plantvillage \
  --weights weights/efficientnetb0_plantvillage.h5
```

Outputs:
- `results/classification_report.txt`
- `results/confusion_matrix.png`

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 / MobileNetV3Small (ImageNet pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dropout(0.3)
    ↓
Dense(512, relu)
    ↓
Dropout(0.2)
    ↓
Dense(38, softmax)
```

**Training strategy:**
1. **Phase 1** (epochs 1–10): Freeze base, train classification head only.
2. **Phase 2** (epochs 11–30): Unfreeze entire network, fine-tune with lr=1e-5.

**Data augmentation:** random flips, brightness, contrast, saturation, hue jitter.

---

## 🌡️ GradCAM Explainability

The app generates GradCAM heatmaps showing which regions of the leaf influenced the prediction, providing interpretable visual explanations alongside each result.

---

## 🌾 Supported Crops & Diseases

| Crop | Disease Classes |
|---|---|
| Tomato | Late blight, Early blight, Bacterial spot, Leaf mold, Septoria, Spider mites, Target spot, Yellow Leaf Curl Virus, Mosaic virus |
| Apple | Apple scab, Black rot, Cedar apple rust |
| Corn (Maize) | Cercospora leaf spot, Common rust, Northern Leaf Blight |
| Grape | Black rot, Esca, Leaf blight |
| Potato | Early blight, Late blight |
| Pepper | Bacterial spot |
| Peach | Bacterial spot |
| Cherry | Powdery mildew |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch |
| Orange | Huanglongbing (Citrus greening) |
| + Healthy class for each crop |

---

## 🛠 Configuration

Key settings are exposed in the Streamlit sidebar:

| Setting | Default | Description |
|---|---|---|
| Model Architecture | EfficientNetB0 | Switch between accuracy and speed |
| Confidence Threshold | 0.50 | Flag uncertain predictions |
| Show GradCAM | On | Visualise attention regions |
| Top-K Predictions | 5 | Number of results to display |

---

## 📦 Tech Stack

| Library | Role |
|---|---|
| TensorFlow / Keras | Model training & inference |
| EfficientNetB0 / MobileNetV3 | Transfer learning backbones |
| NumPy / SciPy | Numerical operations |
| Streamlit | Web interface |
| Plotly | Interactive charts |
| Pillow | Image I/O |
| scikit-learn | Evaluation metrics |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://plantvillage.psu.edu/) — Hughes & Salathé (2015)
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le (2019)
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Howard et al. (2019)
