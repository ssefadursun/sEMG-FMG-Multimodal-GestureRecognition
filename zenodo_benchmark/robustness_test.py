"""
Robustness and Generalization Benchmarks for Multimodal Gesture Recognition (Zenodo Dataset).

This script executes two rigorous stress tests aligned with Young et al. (2025):
1. External Load Invariance: Training on unloaded condition (0 g) and evaluating on maximum load (1000 g).
2. Inter-Subject Generalization: Training on 20 subjects and evaluating on 7 unseen subjects.
"""

import time
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# Configuration
NPZ_FILE = 'zenodo_features.npz'
RANDOM_STATE = 42


def build_robust_classifier() -> LGBMClassifier:
    """
    Constructs a regularized LightGBM classifier to prevent overfitting across domain shifts.
    """
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )


def evaluate_load_robustness(X_fus: np.ndarray, y_gest: np.ndarray, y_weight: np.ndarray):
    """
    Benchmark 1: External Load Invariance (0 g Train -> 1000 g Test).
    """
    print("\n" + "=" * 65)
    print("  BENCHMARK 1: EXTERNAL LOAD INVARIANCE (0 g Train -> 1000 g Test)  ")
    print("=" * 65)

    y_weight_clean = np.array([str(w).strip() for w in y_weight])

    train_mask = (y_weight_clean == '0')
    test_mask = (y_weight_clean == '1000')

    X_train, y_train = X_fus[train_mask], y_gest[train_mask]
    X_test, y_test = X_fus[test_mask], y_gest[test_mask]

    print(f"[INFO] Training Instances (Unloaded - 0 g)    : {len(X_train):,}")
    print(f"[INFO] Testing Instances  (Max Load - 1000 g) : {len(X_test):,}")

    model = build_robust_classifier()
    t_start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = (time.time() - t_start) / 60

    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Load Robustness Accuracy: {acc * 100:.2f}% (Execution Time: {elapsed:.2f} min)")
    print("\n[INFO] Detailed Classification Report (0 g -> 1000 g):")
    print(classification_report(y_test, y_pred, digits=4))


def evaluate_unseen_subjects(X_fus: np.ndarray, y_gest: np.ndarray, y_subj: np.ndarray):
    """
    Benchmark 2: Inter-Subject Domain Shift (20 Subjects Train -> 7 Unseen Subjects Test).
    """
    print("\n" + "=" * 65)
    print("  BENCHMARK 2: INTER-SUBJECT DOMAIN SHIFT (20 Train -> 7 Test)     ")
    print("=" * 65)

    y_subj_clean = np.array([str(s).replace(" ", "") for s in y_subj])
    train_subjects = [f"Par{i}" for i in range(1, 21)]

    train_mask = np.isin(y_subj_clean, train_subjects)
    test_mask = ~train_mask

    X_train, y_train = X_fus[train_mask], y_gest[train_mask]
    X_test, y_test = X_fus[test_mask], y_gest[test_mask]

    print(f"[INFO] Training Instances (Subjects 1-20) : {len(X_train):,}")
    print(f"[INFO] Testing Instances  (Subjects 21-27): {len(X_test):,}")

    model = build_robust_classifier()
    t_start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = (time.time() - t_start) / 60

    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Unseen Subjects Accuracy: {acc * 100:.2f}% (Execution Time: {elapsed:.2f} min)")
    print("\n[INFO] Detailed Classification Report (Unseen Subjects):")
    print(classification_report(y_test, y_pred, digits=4))


def main():
    print("=" * 65)
    print("       Multimodal Robustness and Stress Testing Suite        ")
    print("=" * 65)

    # 1. Load Extracted Features
    print(f"\n[INFO] Loading pre-extracted features from '{NPZ_FILE}'...")
    start_load = time.time()
    data = np.load(NPZ_FILE, allow_pickle=True)

    X_emg = data['X_emg']
    X_fmg = data['X_fmg']
    X_fus = np.hstack((X_emg, X_fmg))
    y_gest = data['y_gesture']
    y_weight = data['y_weight']
    y_subj = data['y_subject']

    print(f"[INFO] Dataset loaded in {time.time() - start_load:.2f} s")
    print(f"[INFO] Total Dataset Volume: {len(y_gest):,} windows (192 multimodal features)")

    # 2. Execute Robustness Benchmarks
    evaluate_load_robustness(X_fus, y_gest, y_weight)
    evaluate_unseen_subjects(X_fus, y_gest, y_subj)

    print("\n" + "=" * 65)
    print("           ALL ROBUSTNESS BENCHMARKS COMPLETED              ")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()