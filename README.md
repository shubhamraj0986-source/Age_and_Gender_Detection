# 🧑‍🔬 Face Detection with Age & Gender Prediction using Deep Learning

A comprehensive deep learning project for **real-time face detection**, **gender classification**, and **age estimation** using the UTKFace dataset. Three architectures — a lightweight **Custom CNN**, **VGG16**, and **ResNet50** — are trained, evaluated, and compared. The best-performing model is deployed for live webcam inference with temporal smoothing for stable predictions.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Development](#-model-development)
  - [Model 1: Custom CNN](#model-1-custom-cnn)
  - [Model 2: VGG16 (Transfer Learning)](#model-2-vgg16-transfer-learning)
  - [Model 3: ResNet50 (Transfer Learning)](#model-3-resnet50-transfer-learning)
- [Results](#-results)
- [Performance Comparison](#-performance-comparison)
- [Training Curves](#-training-curves)
- [Webcam Testing](#-webcam-testing)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Future Improvements](#-future-improvements)
- [Credits](#-credits)

---

## 🎯 Project Overview

This project tackles three interconnected computer vision tasks:

| Task | Type | Description |
|------|------|-------------|
| **Face Detection** | Detection | Localize faces in images/video frames using MTCNN |
| **Gender Classification** | Binary Classification | Predict Male (0) or Female (1) from facial images |
| **Age Prediction** | Regression | Estimate a person's age (in years) from facial images |
| **Real-time Inference** | Deployment | Live webcam pipeline with temporal smoothing |

The project explores three model architectures of increasing complexity:
1. **Custom CNN** — a lightweight from-scratch model
2. **VGG16** — transfer learning with ImageNet-pretrained weights
3. **ResNet50** — transfer learning with residual connections

Each model uses a **multi-output architecture** with shared feature extraction and two independent output branches: one for gender classification (sigmoid + binary crossentropy) and one for age regression (MAE or Huber loss).

---

## 📊 Dataset

### UTKFace (Cropped)

| Property | Details |
|----------|---------|
| **Dataset Name** | UTKFace (cropped version) |
| **Total Samples** | 23,709 face images |
| **Image Format** | `.jpg.chip.jpg` (pre-cropped and aligned) |
| **Age Range** | 0–116 years |
| **Gender Labels** | `0` = Male, `1` = Female |
| **Race Labels** | `0` = White, `1` = Black, `2` = Asian, `3` = Indian, `4` = Others (not used for training) |
| **Image Size** | Variable (resized to 128×128 or 224×224 during training) |

### Filename Convention

Each image filename encodes its labels:
```
[age]_[gender]_[race]_[timestamp].jpg.chip.jpg
```
**Example:** `25_0_2_20170104023102422.jpg.chip.jpg` → Age: 25, Gender: Male, Race: Asian

### Preprocessing & Cleaning

| Step | Details |
|------|---------|
| **Label Parsing** | Age, gender, and race extracted from filenames via string splitting |
| **Invalid File Filtering** | Non-image files (e.g., `.DS_Store`) skipped during parsing |
| **Gender Filtering** | Samples with `gender == 3` (1 sample) removed as invalid → **23,708 valid samples** |
| **Age Normalization** | VGG16 & ResNet50: Age divided by 116.0 to normalize to [0, 1] range; Custom CNN: raw age values used |
| **Image Preprocessing** | Custom CNN: grayscale + rescale to [0, 1]; VGG16/ResNet50: backbone-specific `preprocess_input` |

### Data Splits

| Model | Train | Validation | Test | Split Strategy |
|-------|-------|------------|------|----------------|
| **Custom CNN** | 18,967 (80%) | — | 4,742 (20%) | 80/20 train-test |
| **VGG16** | 18,966 (80%) | — | 4,742 (20%) | 80/20 train-test |
| **ResNet50** | 15,172 (64%) | 3,794 (16%) | 4,742 (20%) | 64/16/20 train-val-test |

> All splits use `random_state=42` for reproducibility.

### Data Augmentation

| Technique | Custom CNN | VGG16 (Phase 1) | ResNet50 (Phase 3) |
|-----------|:----------:|:----------------:|:------------------:|
| Random Rotation (±15°) | ✅ | ✅ (±5°) | ✅ |
| Width Shift (10%) | ✅ | — | ✅ |
| Height Shift (10%) | ✅ | — | ✅ |
| Zoom (10%) | ✅ | — | ✅ |
| Horizontal Flip | ✅ | ✅ | ✅ |

---

## 🏗 Model Development

### Model 1: Custom CNN

> **Notebook:** `Face_detection_model.ipynb`
> **Saved Model:** `utkface_model_3.keras` (~7.5 MB)

| Property | Details |
|----------|---------|
| **Architecture** | Custom CNN (from scratch) |
| **Backbone** | None (fully custom) |
| **Input Size** | 128 × 128 × 1 (grayscale) |
| **Convolutional Blocks** | 4 × (Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)) |
| **Gender Branch** | Dense(256, ReLU) → Dropout(0.3) → Dense(1, sigmoid) |
| **Age Branch** | Dense(256, ReLU) → Dropout(0.3) → Dense(1, ReLU) |
| **Optimizer** | Adam (default lr) |
| **Gender Loss** | Binary Crossentropy |
| **Age Loss** | MAE |
| **Metrics** | Gender: Accuracy · Age: MAE |
| **Epochs** | 30 (with Early Stopping, patience=5) |
| **Batch Size** | 32 |
| **Checkpointing** | Best model saved on `val_loss` |
| **Data Loading** | Full dataset loaded into memory as NumPy arrays |
| **Trainable Parameters** | All layers trainable |

**Training Strategy:**
- Single-phase end-to-end training
- Images loaded as grayscale NumPy arrays (not via generators)
- Augmentation defined via `ImageDataGenerator` but applied at array level
- Early stopping monitors validation loss with patience of 5 epochs

---

### Model 2: VGG16 (Transfer Learning)

> **Notebook:** `VGG16_model.ipynb`
> **Saved Model:** `VGG_final.keras` (~180 MB)

| Property | Details |
|----------|---------|
| **Architecture** | VGG16 + Custom Multi-Output Head |
| **Backbone** | VGG16 (ImageNet pretrained, `include_top=False`) |
| **Input Size** | 128 × 128 × 3 (RGB) |
| **Age Branch** | Dense(256, ReLU) → BatchNorm → Dropout(0.2) → Dense(1, sigmoid) |
| **Gender Branch** | Dense(256, ReLU) → BatchNorm → Dropout(0.2) → Dense(1, sigmoid) |
| **Preprocessing** | `vgg16.preprocess_input` (mean subtraction + channel normalization) |
| **Data Loading** | `ImageDataGenerator.flow_from_dataframe` (disk-based batching) |

#### Phase 1: Head-Only Training

| Property | Details |
|----------|---------|
| **Frozen Layers** | All VGG16 base layers |
| **Optimizer** | Adam (lr = 1e-3) |
| **Age Loss** | Huber Loss |
| **Gender Loss** | Binary Crossentropy |
| **Epochs** | 10 (with Early Stopping, patience=3) |
| **Batch Size** | 32 |
| **Checkpoint** | `VGG16_phase1.keras` |

#### Phase 2: Fine-Tuning

| Property | Details |
|----------|---------|
| **Unfrozen Layers** | Last 30 layers of VGG16 base |
| **Optimizer** | Adam (lr = 1e-4) |
| **Age Loss** | Huber Loss |
| **Gender Loss** | Binary Crossentropy |
| **Epochs** | 30 (with Early Stopping, patience=5) |
| **Learning Rate Schedule** | ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7) |
| **Checkpoint** | `VGG_final.keras` |

**Training Strategy:**
- Two-phase transfer learning: freeze-then-finetune
- Lower learning rate in Phase 2 to preserve pretrained features
- ReduceLROnPlateau for adaptive learning rate decay
- Age normalized to [0, 1] for sigmoid output compatibility

---

### Model 3: ResNet50 (Transfer Learning)

> **Notebook:** `ResNet50_model (1).ipynb`
> **Saved Model:** `ResNet50_Data_Aug.keras` (~210 MB)

| Property | Details |
|----------|---------|
| **Architecture** | ResNet50 + Custom Multi-Output Head |
| **Backbone** | ResNet50 (ImageNet pretrained, `include_top=False`) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Preprocessing** | `resnet50.preprocess_input` |
| **Data Loading** | `ImageDataGenerator.flow_from_dataframe` (disk-based batching) |
| **Validation Set** | Dedicated 16% validation split (separate from test) |

#### Phase 1: Head-Only Training

| Property | Details |
|----------|---------|
| **Frozen Layers** | All ResNet50 base layers |
| **Age Branch** | Dense(512) → BN → Dense(256) → BN → Dense(128) → BN → Dense(1, sigmoid) |
| **Gender Branch** | Dense(128) → BN → Dropout(0.5) → Dense(1, sigmoid) |
| **Optimizer** | Adam (lr = 1e-3) |
| **Age Loss** | Huber Loss |
| **Gender Loss** | Binary Crossentropy |
| **Epochs** | 10 (Early Stopping, patience=3) |
| **Checkpoint** | `ResNet50_phase1.keras` |

#### Phase 2: Fine-Tuning

| Property | Details |
|----------|---------|
| **Unfrozen Layers** | Last 30 layers of ResNet50 |
| **Optimizer** | Adam (lr = 1e-4) |
| **Learning Rate Schedule** | ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7) |
| **Epochs** | 30 (Early Stopping, patience=5) |
| **Checkpoint** | `ResNet50_final.keras` |

#### Phase 3: Data Augmentation + Revised Head (Overfitting Fix)

| Property | Details |
|----------|---------|
| **Age Branch (Revised)** | Dense(256, ReLU) → Dropout(0.3) → Dense(1, linear) |
| **Gender Branch** | Dense(128, ReLU) → BN → Dropout(0.5) → Dense(1, sigmoid) |
| **Loss Weights** | Age: 1.0 · Gender: 0.3 |
| **Data Augmentation** | Rotation ±15°, width/height shifts 10%, zoom 10%, horizontal flip |
| **Optimizer** | Adam (lr = 1e-4) |
| **Epochs** | 30 (Early Stopping, patience=5) |
| **Checkpoint** | `ResNet50_Data_Aug.keras` |

**Training Strategy:**
- Three-phase progressive training approach
- Phase 1 warms up the head → Phase 2 fine-tunes backbone → Phase 3 addresses overfitting
- Phase 3 introduces augmentation, simplified head, and loss weighting to combat overfitting
- Age output changed from sigmoid to linear activation in Phase 3

---

## 📈 Results

### Individual Model Results

| Model | Phase | Gender Accuracy | Age MAE (years) | Notes |
|-------|-------|:--------------:|:--------------:|-------|
| **Custom CNN** | Single Phase | **87.24%** | **6.76** | Lightweight grayscale model; best age MAE |
| **VGG16** | Phase 2 (Fine-tuned) | **97.81%** | **7.42** | Best gender accuracy; final selected model |
| **ResNet50** | Phase 1 (Head-only) | 88.97% | 7.11 | Baseline before fine-tuning |
| **ResNet50** | Phase 2 (Fine-tuned) | 91.75% | 10.24 | Overfitting observed |
| **ResNet50** | Phase 3 (Augmented) | 92.58% | 7.46 | Augmentation reduced overfitting |

### Key Metrics Extracted from Notebooks

| Metric | Custom CNN | VGG16 | ResNet50 (P2) | ResNet50 (P3) |
|--------|:----------:|:-----:|:-------------:|:-------------:|
| Gender Accuracy | 87.24% | **97.81%** | 91.75% | 92.58% |
| Age MAE (years) | **6.76** | 7.42 | 10.24 | 7.46 |
| Model Size | ~7.5 MB | ~180 MB | ~210 MB | ~210 MB |
| Input | Grayscale 128² | RGB 128² | RGB 224² | RGB 224² |

---

## 🏆 Performance Comparison

### Final Ranking

| Rank | Model | Gender Accuracy | Age MAE (years) | Recommendation |
|:----:|-------|:--------------:|:--------------:|----------------|
| 🥇 1 | **VGG16 (Fine-tuned)** | **97.81%** | 7.42 | ✅ **Selected for deployment** — best gender accuracy by a large margin |
| 🥈 2 | ResNet50 (Phase 3 — Augmented) | 92.58% | 7.46 | Strong overall but 5%+ behind VGG16 on gender |
| 🥉 3 | Custom CNN | 87.24% | **6.76** | Best age MAE; ideal for resource-constrained deployment |
| 4 | ResNet50 (Phase 2 — No Aug) | 91.75% | 10.24 | Suffered from overfitting |

### Summary

| Category | Winner | Value |
|----------|--------|-------|
| 🏅 **Best Gender Classification** | VGG16 (Fine-tuned) | **97.81%** accuracy |
| 🏅 **Best Age Prediction** | Custom CNN | **6.76** years MAE |
| 🚀 **Final Selected Model** | VGG16 (Fine-tuned) | Best overall balance for deployment |
| 🪶 **Most Lightweight** | Custom CNN | ~7.5 MB (24× smaller than VGG16) |

---

## 📉 Training Curves

### Custom CNN
- **Gender Accuracy:** Steadily improves from ~53% to ~87% over 30 epochs
- **Total Loss:** Decreases from ~17.5 to ~7.5, with training and validation curves tracking closely
- **Overfitting:** Minimal — early stopping effectively prevents overfitting; train and val curves remain close

### VGG16
- **Phase 1 (Head-Only):** Gender accuracy reaches ~85% validation accuracy within 10 epochs
- **Phase 2 (Fine-Tuned):** Gender accuracy jumps to ~97% with fine-tuning; age MAE stabilizes around 0.08 (normalized)
- **Loss Trends:** Smooth convergence in Phase 2; ReduceLROnPlateau helps fine-tune the learning rate
- **Overfitting:** Well-controlled through two-phase training and dropout regularization

### ResNet50
- **Phase 2 (No Augmentation):** Clear overfitting — training loss drops significantly below validation loss; age MAE degrades to 10.24 years on the test set
- **Phase 3 (With Augmentation):** Augmentation + simplified head closes the train-val gap; age MAE improves from 10.24 → 7.46 years; gender accuracy improves from 91.75% → 92.58%
- **Key Observation:** Data augmentation and a simpler head were critical for ResNet50 generalization

---

## 📹 Webcam Testing

> **Notebook:** `WebCam_test.ipynb`
> **Model Used:** `VGG_final.keras` (VGG16 Fine-tuned)

### Real-Time Prediction Pipeline

```
Webcam Frame → MTCNN Face Detection → Padded Crop → Resize to 128×128
→ VGG16 preprocess_input → Model Prediction → Age Smoothing + Gender Voting
→ Overlay Labels on Frame → Display
```

### Components

| Component | Implementation |
|-----------|---------------|
| **Face Detection** | MTCNN (Multi-task Cascaded CNNs) via `mtcnn` library |
| **Face Cropping** | Padded bounding box (30% top, 12% bottom/left/right padding) |
| **Preprocessing** | `cv2.resize` to 128×128 → `vgg16.preprocess_input` |
| **Model** | `VGG_final.keras` (VGG16 fine-tuned) |
| **Age Output** | `pred[0][0][0] * 116` (denormalized from [0,1] back to years) |
| **Gender Output** | `pred[1][0][0]` → threshold at 0.5 (>0.5 = Female) |

### Temporal Smoothing

| Technique | Parameter | Purpose |
|-----------|-----------|---------|
| **EMA (Exponential Moving Average)** | α = 0.15 | Smooths age predictions over time to reduce frame-to-frame jitter |
| **Jump Threshold** | 12 years | Large age jumps (>12) bypass EMA smoothing to allow fast recovery |
| **Median Filter** | History length = 15 frames | `np.median(age_history)` produces the final displayed age |
| **Gender Majority Vote** | Window = 15 frames | `statistics.mode()` over recent predictions stabilizes gender label |

### Display Features

- **Green Rectangle** — tight MTCNN bounding box around the face
- **Orange Rectangle** — padded crop region sent to the model
- **Red Dots** — facial landmarks (eyes, nose, mouth) from MTCNN
- **Text Overlay** — `"{Gender}, {Age}"` and `"Det:{confidence:.2f}"` labels
- **Exit** — Press `x` to quit

### Limitations

- Requires a working webcam (`cv2.VideoCapture(0)`)
- MTCNN runs per-frame (can be slow without GPU)
- Predictions may be less accurate for extreme ages (very young/very old), non-frontal faces, or poor lighting
- Single-person optimization — multiple faces are supported but smoothing is per-face only for the first detected face

---

## 📁 Project Structure

```
Face_detection/
├── Face_detection_model.ipynb   # Custom CNN — training & evaluation
├── VGG16_model.ipynb            # VGG16 transfer learning — training & evaluation
├── ResNet50_model (1).ipynb     # ResNet50 transfer learning — 3-phase training
├── WebCam_test.ipynb            # Real-time webcam inference pipeline
├── Requirements.ipynb           # Dependency installation notebook
├── README.md                    # This file
│
├── utkface_model_3.keras        # Saved Custom CNN model (~7.5 MB)
├── VGG_final.keras              # Saved VGG16 fine-tuned model (~180 MB)
├── ResNet50_Data_Aug.keras       # Saved ResNet50 model (~210 MB)
│
└── utkcropped/                  # UTKFace dataset (23,709 cropped face images)
    ├── 1_0_0_20161219203337055.jpg.chip.jpg
    ├── 25_1_2_20170104021040316.jpg.chip.jpg
    ├── ...
    └── 116_1_0_20170120134646399.jpg.chip.jpg
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- pip or conda package manager
- Webcam (for real-time inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Face_detection.git
cd Face_detection
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
conda create -n utk_face python=3.11
conda activate utk_face
```

### Step 3: Install Dependencies

```bash
# Core libraries
pip install numpy pandas matplotlib seaborn

# Computer vision
pip install opencv-python pillow

# Deep learning framework
pip install tensorflow

# Machine learning utilities
pip install scikit-learn

# Face detection
pip install mtcnn
```

Or install all at once:
```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow tensorflow scikit-learn mtcnn
```

### Step 4: Download the Dataset

Download the **UTKFace cropped** dataset and place the images in the `utkcropped/` directory:
- [UTKFace Cropped Dataset (Kaggle)](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped)

### Step 5: Download Pretrained Models

Download the pretrained model weights and place them in the project root directory:
- [Download Models (Google Drive)](<!-- ADD YOUR GOOGLE DRIVE LINK HERE -->)

---

## 🚀 Usage

### Training a Model

1. **Custom CNN:**
   ```bash
   jupyter notebook Face_detection_model.ipynb
   ```
   Run all cells sequentially. The best model will be saved as `utkface_model_3.keras`.

2. **VGG16 (Transfer Learning):**
   ```bash
   jupyter notebook VGG16_model.ipynb
   ```
   Run all cells. Phase 1 saves `VGG16_phase1.keras`; Phase 2 saves `VGG_final.keras`.

3. **ResNet50 (Transfer Learning):**
   ```bash
   jupyter notebook "ResNet50_model (1).ipynb"
   ```
   Run all cells through 3 phases. Final model: `ResNet50_Data_Aug.keras`.

### Evaluation

Each training notebook includes evaluation cells at the end that:
1. Reload the best saved checkpoint
2. Generate predictions on the held-out test set
3. Report **Gender Accuracy** and **Age MAE (years)**

### Real-Time Webcam Testing

```bash
jupyter notebook WebCam_test.ipynb
```
Run all cells. The webcam feed will open with real-time age and gender predictions overlaid.

> **Controls:** Press `x` to exit the webcam window.

> **Note:** Ensure `VGG_final.keras` exists in the project directory before running.

---

## 🔑 Key Findings

### Best Achieved Results

| Metric | Best Value | Model |
|--------|-----------|-------|
| **Gender Accuracy** | **97.81%** | VGG16 (Fine-tuned) |
| **Age MAE** | **6.76 years** | Custom CNN |

### Major Challenges

1. **Overfitting in ResNet50** — Phase 2 (without augmentation) showed severe overfitting, with age MAE degrading from 7.11 to 10.24 years on the test set. This was mitigated in Phase 3 through data augmentation, a simplified head, and loss weighting.

2. **Age Regression Difficulty** — Age prediction is inherently harder than gender classification. Even the best model achieves ~6.8 years MAE, reflecting the ambiguity in predicting age from facial appearance alone.

3. **Multi-Output Balancing** — Balancing the age regression and gender classification losses required careful loss weighting (ResNet50 Phase 3 used 1.0:0.3) and compatible activation functions.

4. **Dataset Imbalance** — The UTKFace dataset has an uneven age distribution (concentrated in 20–40 range), which affects prediction accuracy for extreme ages.

### Lessons Learned

- **Transfer learning dramatically boosts gender accuracy** — VGG16 jumped from 87% (Custom CNN) to 97.8% with pretrained features.
- **Simpler architectures can win on regression** — The Custom CNN achieved the best age MAE despite being the smallest model, suggesting that age regression benefits from less complex representations.
- **Two-phase training is critical** — Freezing the backbone first and fine-tuning later consistently outperforms end-to-end training with pretrained weights.
- **Temporal smoothing is essential for deployment** — Raw per-frame predictions are too noisy for a usable webcam experience; EMA + median filtering provides stable, natural-looking outputs.
- **Grayscale input works well for lightweight models** — The Custom CNN's grayscale approach reduces model size by 24× with only a modest accuracy trade-off.

---

## 🔮 Future Improvements

| Area | Suggestion | Expected Impact |
|------|-----------|-----------------|
| **Architecture** | Try EfficientNet-B0/B3 as backbone | Better accuracy-efficiency trade-off |
| **Age Estimation** | Frame age as ordinal regression or classification into bins | May reduce MAE for extreme ages |
| **Data Augmentation** | Add CutMix, MixUp, and color jitter | Better generalization, especially for ResNet50 |
| **Larger Datasets** | Combine UTKFace with AgeDB, MORPH-II, or IMDB-WIKI | More diverse training data |
| **Face Alignment** | Apply facial landmark-based alignment before cropping | More consistent inputs to the model |
| **Better Face Detection** | Replace MTCNN with RetinaFace or YOLOv8-Face | Faster and more robust detection |
| **Temporal Smoothing** | Implement Kalman filtering for age tracking | Smoother transitions, especially during movement |
| **Multi-Person Tracking** | Add face tracking (e.g., DeepSORT) for per-identity smoothing | Independent predictions per person |
| **Age Distribution** | Apply class-balanced sampling or focal loss | Better performance on underrepresented age groups |
| **Model Compression** | Quantization (INT8) and pruning for edge deployment | Enable real-time on mobile/edge devices |

---

## 🙏 Credits

### Libraries & Frameworks

| Library | Purpose |
|---------|---------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | Deep learning framework for model training and inference |
| [OpenCV](https://opencv.org/) | Image processing and webcam capture |
| [MTCNN](https://github.com/ipazc/mtcnn) | Multi-task Cascaded CNN for face detection |
| [scikit-learn](https://scikit-learn.org/) | Train-test splitting, accuracy & MAE metrics |
| [NumPy](https://numpy.org/) | Array operations and numerical computing |
| [Pandas](https://pandas.pydata.org/) | DataFrame management for dataset handling |
| [Matplotlib](https://matplotlib.org/) | Training curve visualization |
| [Seaborn](https://seaborn.pydata.org/) | Statistical data visualization |
| [Pillow](https://pillow.readthedocs.io/) | Image loading and display |

### Dataset

- **UTKFace** — Large-scale face dataset with age, gender, and ethnicity annotations
  - Source: [UTKFace Cropped on Kaggle](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped)
  - Paper: [*Age Progression/Regression by Conditional Adversarial Autoencoder*](https://susanqq.github.io/UTKFace/)
  - Contains 23,000+ face images spanning 0–116 years

### Pretrained Models

- **VGG16** — *Very Deep Convolutional Networks for Large-Scale Image Recognition* (Simonyan & Zisserman, 2014)
- **ResNet50** — *Deep Residual Learning for Image Recognition* (He et al., 2015)
- Both models loaded with **ImageNet** pretrained weights via `keras.applications`

---

<p align="center">
  <i>Built with ❤️ using TensorFlow, OpenCV, and the UTKFace dataset</i>
</p>
