
# 🧠 Static Filipino Sign Language (FSL) Alphabet Recognition using HOG + SVM

This project implements a **machine learning system for recognizing Static Filipino Sign Language (FSL) alphabets** using classical computer vision and machine learning techniques:

* Histogram of Oriented Gradients (HOG) for feature extraction
* Support Vector Machine (SVM) for classification

⚠️ This model is ONLY for **STATIC alphabet gestures (A–Z or dataset classes)**. It does NOT support dynamic/moving signs.

---

## 📁 Project Structure

```
fsl-svm/
│
├── server/
│   ├── FSL-dataset/          # Extracted dataset (NOT included in repo)
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   └── ...
│   │
│   ├── csvs/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── fsl_svm_model.py      # Main SVM + HOG pipeline
│   ├── pipeline.py           # Training & evaluation script
│   ├── svm_hog_model.pkl     # Saved trained model (generated after training)
│   ├── requirements.txt      # Needed dependencies
│   └── venv/
│ 
└── README.md
```

---

## ⚙️ Requirements

* Python 3.9+
* pip
* virtualenv (recommended)

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/RaiGZ971/fsl-svm.git
cd fsl-svm/server
```

---

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset Setup

⚠️ The dataset is required for training.

---

### 📥 Step 1: Download dataset

Download `FSL-dataset.zip`

---

### 📂 Step 2: Extract dataset

Extract it inside the `server/` directory:

```bash
unzip FSL-dataset.zip -d server/
```

---

### 📁 Final dataset structure

```
server/FSL-dataset/
│
├── A/
├── B/
├── C/
├── D/
├── E/
└── ...
```

Each folder represents a **STATIC FSL alphabet class**.

---

## 🚀 Running the Project

### ▶️ Train model

```bash
python pipeline.py
```

This will:

* Load dataset from CSV
* Preprocess images
* Extract HOG features
* Train SVM classifier
* Evaluate model performance

---

## 📊 Evaluation Outputs

* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix
* Per-class Specificity
* Training Accuracy
* Testing Accuracy
* Prediction confidence scores

---

## 🧠 Model Details

* **Type**: Static FSL Alphabet Recognition
* **Features**: HOG (Histogram of Oriented Gradients)
* **Classifier**: Support Vector Machine (SVM)
* **Kernel**: Linear
* **Probability Output**: Enabled
* **Input Size**: 128 × 128 grayscale images

---

## ⚙️ Preprocessing Pipeline

Each image undergoes:

1. Resize to 128×128
2. Convert to grayscale
3. Otsu thresholding
4. Morphological operations (opening & closing)
5. Hand masking
6. HOG feature extraction

---

## 💾 Saving & Loading Model

### Save trained model

```python
model.save_svm_model()
```

### Load trained model

```python
model.load_svm_model()
```

---

## ⚠️ Important Notes

* This model ONLY supports **static gestures**
* It does NOT handle motion-based signs
* Poor image quality may reduce accuracy
* Dataset must be properly labeled and structured

---

## ❌ Common Issues

### Empty dataset error

* Ensure `FSL-dataset/` is correctly extracted
* Check CSV file paths

