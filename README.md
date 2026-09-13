# Multimodal Hand Gesture Recognition via sEMG and FMG Sensor Fusion: High-Dimensional Feature Engineering and Ensemble Benchmarks


This repository provides an open-source, reproducible research framework for multimodal hand gesture classification using surface electromyography (sEMG) and force myography (FMG) sensor fusion. The codebase includes clean, PEP 8-compliant implementations, a 12-feature **Golden Standard** biosignal extraction pipeline, and specialized machine learning architectures validated against two major open-access benchmark databases:
1. **Zenodo Multimodal Forearm Dataset (Young et al., 2025, *PLOS ONE*)**
2. **GREFTUD Multimodal Gesture Dataset (Rohr et al., 2025, *IEEE TNSRE*)**

---

## 📁 Repository Structure

```text
push_github/
├── README.md
├── requirements.txt
│
├── zenodo_benchmark/
│   ├── feature_extraction.py      # Extracts 192-D hybrid features (613,440 windows)
│   ├── lightgbm_training.py       # Full-scale LightGBM classifier evaluation
│   ├── svm_training.py            # Scalable RBF-Kernel SVM benchmark
│   └── robustness_test.py         # Load Invariance (0g -> 1000g) & Unseen Subjects tests
│
└── gref_benchmark/
    ├── feature_extraction_gref.py # Extracts 96-D hybrid features from dataset.hdf5
    ├── ensemble_training_gref.py  # Heterogeneous Soft Voting Ensemble (Intra-Subject)
    └── loocv_evaluation_gref.py   # 13-Fold Leave-One-Out Cross-Validation (Inter-Subject)
```

---

## 🔬 Biosignal Feature Engineering ("Golden Standard" 12-Feature Set)

Each 250 ms analysis window across all channels is transformed into a 12-parameter descriptive vector capturing amplitude, non-linear dynamics, complexity, and spectral power density:

$$\mathbf{f} = \left[ \text{MAV}, \text{RMS}, \text{WL}, \text{ZC}_{\text{dyn}}, \text{SSC}_{\text{dyn}}, \text{LogD}, \text{Act}, \text{Mob}, \text{Comp}, \text{MNF}, \text{MDF}, \text{SpecEnt} \right]$$

* **Amplitude & Energy:** Mean Absolute Value ($\text{MAV}$), Root Mean Square ($\text{RMS}$), and Waveform Length ($\text{WL}$).
* **Dynamic-Threshold Morphology:** Zero Crossing ($\text{ZC}$) and Slope Sign Change ($\text{SSC}$) computed with an adaptive noise rejection threshold $\text{Th} = 0.05 \cdot \sigma_x$ to eliminate low-frequency baseline drift in FMG and high-frequency stochastic noise in sEMG.
* **Complexity & Non-linear Force:** Log-Detector ($\text{LogD} = \exp(\frac{1}{N}\sum \ln(\vert{}x_i\vert{} + \epsilon))$) and Hjorth Parameters ($\text{Activity}$, $\text{Mobility}$, $\text{Complexity}$) to model motor unit chaotic recruitment and fine coordination.
* **Spectral Domain (Welch PSD):** Mean Frequency ($\text{MNF}$), Median Frequency ($\text{MDF}$), and Spectral Entropy ($\text{SpecEnt}$) quantifying signal disorder and frequency shifts under muscle contraction.

---

## 📊 Benchmark 1: Zenodo Dataset (Young et al., 2025)

* **Dataset Overview:** 27 able-bodied subjects, 8-channel concentric sEMG + 8-channel FSR-based FMG ($f_s = 2000\text{ Hz}$), 4 functional grasp classes (Key, Pinch, Power, Tripod), 8 limb positions, and 5 load levels (0 g to 1000 g).
* **Total Volume:** 613,440 temporal windows ($250\text{ ms}$ length, $50\%$ overlap), $192\text{-D}$ fusion feature matrix.

### Performance vs. Literature Baseline

| Evaluation Metric / Scenario | Young et al. (2025) Baseline (LDA) | This Study (LightGBM) | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **sEMG Only Accuracy** | ~88.00% – 91.00% | **93.75%** | **+2.75% to +5.75%** |
| **FMG Only Accuracy** | ~92.00% – 94.00% | **99.71%** | **+5.71% to +7.71%** |
| **Multimodal Fusion Accuracy** | 97.34% | **99.92%** | **+2.58% (Ceiling)** |
| **Load Robustness (0 g Train $\rightarrow$ 1000 g Test)** | 54.00% (Fusion Collapse) | **70.37% (Robust)** | **+16.37%** |
| **Inter-Subject Domain Shift (20 Train $\rightarrow$ 7 Test)**| *Not Evaluated* | **43.59%** | *Domain Shift Proof* |

---

## 📊 Benchmark 2: GREFTUD Dataset (Rohr et al., 2025)

* **Dataset Overview:** 13 healthy subjects, 4-channel sEMG + 4-channel piezoelectric ferroelectret FMG ($f_s = 1000\text{ Hz}$), 66 gesture classes across 3 physiological speeds.
* **Classification Pipeline:** Heterogeneous Soft Voting Ensemble (Random Forest + LightGBM + SVM / Extra Trees) with Subject-Wise Z-Score Standardization.
* **Total Volume:** 69,776 temporal windows ($250\text{ ms}$ length, $80\%$ overlap), $96\text{-D}$ fusion feature matrix.

### Comparative Validation (Task 2: Single Finger Movements - 5 Classes)

| Benchmark Protocol | Classifier Model | sEMG Only (%) | FMG Only (%) | Fusion (sEMG + FMG) (%) | Relative Fusion Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rohr et al. (2025) - LOOCV** | Support Vector Machine (SVM) | ~22.0 – 25.0% | ~20.0 – 22.0% | **31.00%** | +35.0% |
| **Rohr et al. (2025) - LOOCV** | Neural Network (NN) | ~22.0 – 25.0% | ~20.0 – 22.0% | **33.00%** | +42.0% |
| **Rohr et al. (2025) - LOOCV** | XGBoost | ~22.0 – 25.0% | ~20.0 – 22.0% | **33.00%** | +42.0% |
| **This Study - LOOCV (Inter-Subject)** | **RF + LGBM + ET Ensemble** | **31.35%** | **34.57%** | **36.83%** | **+17.5% (Outperformed)** |
| **This Study - Intra-Subject (80/20)** | **RF + LGBM + SVM Ensemble** | **76.31%** | **67.17%** | **91.63%** | **+20.1% (High Precision)** |



## 🚀 Execution & Reproducibility Guide

### 1. Zenodo Benchmark Suite
Download the raw Zenodo dataset files into `zenodo_benchmark/Data/`:

```bash
cd zenodo_benchmark

# Step 1: Feature Extraction (Outputs zenodo_features.npz)
python3 feature_extraction.py

# Step 2: Model Training & Evaluation
python3 lightgbm_training.py


# Step 3: Load Invariance and Subject Generalization Tests
python3 robustness_test.py
```

### 2. GREFTUD Benchmark Suite
Place `dataset.hdf5` inside the `gref_benchmark/` directory:

```bash
cd gref_benchmark

# Step 1: Feature Extraction (Outputs greftud_features.npz)
python3 feature_extraction_gref.py

# Step 2: Intra-Subject Soft Voting Ensemble Evaluation
python3 ensemble_training_gref.py

# Step 3: 13-Fold Inter-Subject Leave-One-Out Cross-Validation
python3 loocv_evaluation_gref.py
```

---

## 📚 References & Datasets

If you utilize this codebase or methodology, please reference the corresponding baseline publications and datasets:

1. **Zenodo Study:**
   > Young, P. R., Hong, K., Winslow, E. J., Sagastume, G. K., Battraw, M. A., Whittle, R. S., & Schofield, J. S. (2025). The effects of limb position and grasped load on hand gesture classification using electromyography, force myography, and their combination. *PLOS ONE*, 20(4), e0321319.  
   > Dataset: [Zenodo Repository]

2. **GREFTUD Study:**
   > Rohr, M., Haidamous, J., Schäfer, N., Schaumann, S., Latsch, B., Kupnik, M., & Antink, C. H. (2025). On the benefit of FMG and EMG sensor fusion for gesture recognition using cross-subject validation. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 33, 935–944.  
   > Dataset: [TU Darmstadt HDF5 Repository]

---
