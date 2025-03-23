# 🧠 NeuroArchitectura: Encoding and Decoding Abstract Engineering Concepts from Visual Brain Signals

**Author**: Guilherme Simioni Bonfim  
**Date**: March 21st, 2025

---

## 📚 Abstract

This project explores the neural basis of **abstract architectural and engineering concepts** — _Firmitas_ (structural robustness), _Utilitas_ (functional use), and _Venustas_ (aesthetic beauty) — through analysis of EEG signals recorded during visual thinking tasks.

Participants were shown **150 categorized images**, while their brain activity was recorded via EEG. They were asked to consciously reflect on each concept while viewing. The analysis was carried out in two core directions:

- 🧠 **Direct EEG Classification**
- 🖼️ **Image Classifier-to-EEG Regression**

---

## 🔍 Methodology

### 1. Direct EEG Classification
- Leveraged the **EEGNet architecture** to classify EEG signals by abstract class.
- **Channel activation analysis** revealed key involvement of:
  - 🟢 *Occipital Cortex* (early visual processing)
  - 🔵 *Anterior Prefrontal Cortex* (higher-order cognition)
- A **channel masking procedure** assessed the importance of specific regions.

### 2. Image Classifier-to-EEG Regression
- Trained an **Inception-V3** model on the image dataset.
- Performed **layer-wise regression** to predict EEG signals from model activations.
- Found strongest alignment with **early convolutional layers**, indicating lower-level visual representations play a significant role.

---

## 🧪 Key Findings

- EEG signals **encode meaningful distinctions** between _Firmitas_, _Utilitas_, and _Venustas_.
- Brain activity reflects **basic visual and higher cognitive processing**, rather than deeper motor or abstract visual discriminations.
- Regression results show **limited overlap** between EEG signals and deeper neural network layers.

---

## 🚀 Future Directions

- 🧠 Increase dataset size and diversity
- 🤝 Integrate **multimodal approaches** (e.g. fMRI, eye-tracking)
- 🔬 Explore **temporal dynamics** of abstract concept formation
- 🤖 Improve alignment between biological and artificial neural representations

---

## 📂 Repository Structure
 File | Description |
|------|-------------|
| `define.py` | Implementation of EEG Classifier|
| `extract_activations.py` | Extracts activations from Inception-V3 layers for use in EEG regression. |
| `masking.py` | Applies masking to EEG channels to evaluate their impact on classification performance. |
| `regression.py` | Performs regression from image model activations to EEG signals. |
| `train.py` | Trains the EEGNet model to classify EEG recordings into abstract concept classes. Using define.py |
| `verify.py` | Obtains Visualization and results/plots |

