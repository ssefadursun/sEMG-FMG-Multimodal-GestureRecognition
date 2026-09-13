"""
Multimodal Hand Gesture Classification using LightGBM.

This script evaluates gesture recognition performance across individual modalities
(sEMG, FMG) and feature-level sensor fusion (sEMG + FMG) on the Zenodo dataset.
"""

import time
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configuration
NPZ_FILE = 'zenodo_features.npz'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    print("=" * 60)
    print("  Multimodal Gesture Classification: LightGBM Evaluation  ")
    print("=" * 60)

    # 1. Load Extracted Features
    print("\n[INFO] Loading pre-extracted features from disk...")
    start_load = time.time()
    data = np.load(NPZ_FILE, allow_pickle=True)

    X_emg = data['X_emg']
    X_fmg = data['X_fmg']
    y_gesture = data['y_gesture']

    X_fusion = np.hstack((X_emg, X_fmg))
    total_samples = len(y_gesture)

    print(f"[INFO] Data loaded in {time.time() - start_load:.2f} s")
    print(f"[INFO] Total Samples: {total_samples:,}")
    print(f"[INFO] Feature Dimensions: sEMG ({X_emg.shape[1]}), FMG ({X_fmg.shape[1]}), Fusion ({X_fusion.shape[1]})")

    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    (
        X_e_train, X_e_test,
        X_f_train, X_f_test,
        X_fus_train, X_fus_test,
        y_train, y_test
    ) = train_test_split(
        X_emg, X_fmg, X_fusion, y_gesture,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_gesture
    )

    # 3. Model Definition
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=127,
        max_depth=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    print("\n[INFO] Training LightGBM models across modalities...")
    start_train = time.time()

    # --- Modality 1: sEMG Only ---
    lgbm_model.fit(X_e_train, y_train)
    acc_emg = accuracy_score(y_test, lgbm_model.predict(X_e_test))

    # --- Modality 2: FMG Only ---
    lgbm_model.fit(X_f_train, y_train)
    acc_fmg = accuracy_score(y_test, lgbm_model.predict(X_f_test))

    # --- Modality 3: Multimodal Fusion (sEMG + FMG) ---
    lgbm_model.fit(X_fus_train, y_train)
    acc_fus = accuracy_score(y_test, lgbm_model.predict(X_fus_test))

    elapsed_time = (time.time() - start_train) / 60

    # 4. Results Summary
    print("\n" + "=" * 45)
    print("          EXPERIMENTAL RESULTS            ")
    print("=" * 45)
    print(f"  Modality / Setup         Accuracy (%)")
    print("-" * 45)
    print(f"  1. sEMG Only             : {acc_emg * 100:6.2f}%")
    print(f"  2. FMG Only              : {acc_fmg * 100:6.2f}%")
    print(f"  3. Fusion (sEMG + FMG)   : {acc_fus * 100:6.2f}%")
    print("=" * 45)
    print(f"[INFO] Total execution time: {elapsed_time:.2f} min\n")


if __name__ == '__main__':
    main()