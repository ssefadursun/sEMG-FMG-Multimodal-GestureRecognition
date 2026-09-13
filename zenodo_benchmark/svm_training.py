"""
Multimodal Hand Gesture Classification using Support Vector Machines (SVM).

This script evaluates gesture classification performance across individual modalities
(sEMG, FMG) and feature-level sensor fusion (sEMG + FMG) on the Zenodo dataset using
an RBF-kernel SVM with Z-score feature standardization.
"""

import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Configuration
NPZ_FILE = 'zenodo_features.npz'
SAMPLE_SIZE = 50000  # Stratified subsample size for scalable SVM training
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVM_C = 10.0
SVM_KERNEL = 'rbf'
SVM_GAMMA = 'scale'


def main():
    print("=" * 60)
    print("      Multimodal Gesture Classification: SVM Evaluation      ")
    print("=" * 60)

    # 1. Load Extracted Features
    print("\n[INFO] Loading pre-extracted features from disk...")
    start_load = time.time()
    data = np.load(NPZ_FILE, allow_pickle=True)

    X_emg = data['X_emg']
    X_fmg = data['X_fmg']
    y_gesture = data['y_gesture']

    print(f"[INFO] Features loaded in {time.time() - start_load:.2f} s")
    print(f"[INFO] Total Available Samples: {len(y_gesture):,}")
    print(f"[INFO] Feature Dimensions: sEMG ({X_emg.shape[1]}), FMG ({X_fmg.shape[1]})")

    # 2. Stratified Subsampling (Computational Feasibility for Kernel SVM)
    if len(y_gesture) > SAMPLE_SIZE:
        print(f"\n[INFO] Applying stratified subsampling to {SAMPLE_SIZE:,} instances for SVM optimization...")
        _, X_e_sub, _, X_f_sub, _, y_sub = train_test_split(
            X_emg, X_fmg, y_gesture,
            test_size=SAMPLE_SIZE,
            stratify=y_gesture,
            random_state=RANDOM_STATE
        )
    else:
        X_e_sub, X_f_sub, y_sub = X_emg, X_fmg, y_gesture

    # Construct Multimodal Fusion Matrix (sEMG + FMG)
    X_fusion_sub = np.hstack((X_e_sub, X_f_sub))

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    (
        X_e_train, X_e_test,
        X_f_train, X_f_test,
        X_fus_train, X_fus_test,
        y_train, y_test
    ) = train_test_split(
        X_e_sub, X_f_sub, X_fusion_sub, y_sub,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sub
    )

    # 4. Modality-Wise Z-Score Feature Standardization
    print("[INFO] Applying Z-score feature standardization...")
    scaler_e = StandardScaler()
    X_e_train_s = scaler_e.fit_transform(X_e_train)
    X_e_test_s = scaler_e.transform(X_e_test)

    scaler_f = StandardScaler()
    X_f_train_s = scaler_f.fit_transform(X_f_train)
    X_f_test_s = scaler_f.transform(X_f_test)

    scaler_fus = StandardScaler()
    X_fus_train_s = scaler_fus.fit_transform(X_fus_train)
    X_fus_test_s = scaler_fus.transform(X_fus_test)

    # 5. Model Definition & Benchmark
    svm_model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        random_state=RANDOM_STATE
    )

    print("\n[INFO] Training SVM models across modalities...")
    start_train = time.time()

    # --- Modality 1: sEMG Only ---
    svm_model.fit(X_e_train_s, y_train)
    acc_emg = accuracy_score(y_test, svm_model.predict(X_e_test_s))

    # --- Modality 2: FMG Only ---
    svm_model.fit(X_f_train_s, y_train)
    acc_fmg = accuracy_score(y_test, svm_model.predict(X_f_test_s))

    # --- Modality 3: Multimodal Fusion (sEMG + FMG) ---
    svm_model.fit(X_fus_train_s, y_train)
    acc_fus = accuracy_score(y_test, svm_model.predict(X_fus_test_s))

    elapsed_time = (time.time() - start_train) / 60

    # 6. Results Summary
    print("\n" + "=" * 45)
    print("          EXPERIMENTAL RESULTS (SVM)      ")
    print("=" * 45)
    print("  Modality / Setup         Accuracy (%)")
    print("-" * 45)
    print(f"  1. sEMG Only             : {acc_emg * 100:6.2f}%")
    print(f"  2. FMG Only              : {acc_fmg * 100:6.2f}%")
    print(f"  3. Fusion (sEMG + FMG)   : {acc_fus * 100:6.2f}%")
    print("=" * 45)
    print(f"[INFO] Total training time: {elapsed_time:.2f} min\n")


if __name__ == '__main__':
    main()