# Affect Recognition with Multi-Task CNNs  

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](#)  


---

## 📌 Overview  

This project implements **multi-task CNN models** for **facial affect recognition**.  
We study affect in two complementary spaces:  

- **Categorical** → 8 emotions (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)  
- **Dimensional** → **Valence** (−1 to +1; pleasantness) and **Arousal** (−1 to +1; activation)  

The architecture uses a **shared CNN backbone** (ResNet / DenseNet / EfficientNet) with two heads:  

1. **Classification Head** → 8-class softmax  
2. **Regression Head** → Valence & Arousal prediction  

---

## ✨ Features  

- Transfer learning with **freeze → unfreeze** fine-tuning  
- Strong **data augmentations**: RandAugment, ColorJitter, RandomErasing  
- **Balanced sampling** + label smoothing + optional MixUp  
- Full evaluation pipeline with:  
  - **Classification metrics**: Accuracy, F1, Cohen’s κ, Krippendorff’s α, ROC-AUC, PR-AUC  
  - **Regression metrics**: RMSE, Pearson r, SAGR, CCC  
- Auto-saved training curves, confusion matrices & qualitative analysis  
- Clean, modular **PyTorch implementation** (Colab-friendly)  

---

## 📊 Results (Quick Look)  

| Model            | Accuracy | F1-macro | Cohen’s κ |
|------------------|---------:|---------:|----------:|
| DenseNet121      | **0.469** | **0.469** | **0.393** |
| ResNet18         | 0.439    | 0.438    | 0.359     |
| EfficientNet-B0  | 0.427    | 0.424    | 0.346     |
| ResNet50         | 0.401    | 0.395    | 0.316     |

📁 Regression metrics (RMSE, r, SAGR, CCC) are saved in `results/*.json`  
📁 Training curves & confusion matrices → `figures/`  

---

## 🗂 Dataset  

Provided dataset includes:  

- Cropped & resized **224×224 RGB face images**  
- **Facial landmarks (68 points)** in `.npy` files  
- Labels:  
  - **Expression IDs** (0–7)  
  - **Valence** ∈ [−1, +1] (−2 if uncertain/no face)  
  - **Arousal** ∈ [−1, +1] (−2 if uncertain/no face)  

📌 Dataset link (FAST email required): [Google Drive](https://drive.google.com/file/d/1PdQBPUuHSBfEtK30cV2IIpRFBJ4-8B78/view?usp=sharing)  

---

## 🚀 Training & Evaluation  

- Train models with different CNN backbones (ResNet, DenseNet, EfficientNet).  
- Evaluate using both **categorical** and **dimensional** metrics.  
- Save training graphs, confusion matrices, and qualitative examples.  

Outputs include:  

- Training & validation loss/accuracy curves  
- Confusion matrices  
- Regression scatterplots  
- Saved metrics (`results_*.json`)  

---

## 🎨 Qualitative Analysis  

- Correct vs incorrect classification examples  
- Mixed expression cases  
- Predicted vs ground-truth valence/arousal scatter  

---

## 📑 Deliverables  

- **Code**: Modular, documented PyTorch notebook + scripts  
- **Report (PDF)**: Includes  
  - Architecture & parameters  
  - Dataset split details  
  - Training graphs  
  - Performance metrics (classification + regression)  
  - Comparison of multiple CNN baselines  
  - Example predictions (correct + incorrect)  

---

## 📈 Evaluation Metrics  

- **Classification**: Accuracy, F1, κ, α, ROC-AUC, PR-AUC  
- **Regression**: RMSE, Correlation (r), SAGR, CCC  

