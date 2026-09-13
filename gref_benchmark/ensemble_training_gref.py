"""
Heterogeneous Soft Voting Ensemble Training (GREFTUD Dataset).

Evaluates Random Forest + LightGBM + SVM (RBF) Soft Voting Ensemble under
an intra-subject stratified evaluation protocol across three clinical tasks:
- Task 1: Flexion vs Extension
- Task 2: Single Finger Movements
- Task 3: All Active Movement Classes (66 Classes)
"""

import time
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

NPZ_FILE = 'greftud_features.npz'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_soft_voting_ensemble() -> VotingClassifier:
    """
    Constructs the 3-model Heterogeneous Soft Voting Ensemble.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )

    return VotingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm), ('svm', svm)],
        voting='soft',
        n_jobs=1
    )


def main():
    print("=" * 70)
    print("   Heterogeneous Soft Voting Ensemble Benchmark (Intra-Subject)   ")
    print("=" * 70)

    print(f"\n[INFO] Loading pre-extracted features from '{NPZ_FILE}'...")
    data = np.load(NPZ_FILE, allow_pickle=True)
    X_emg = data['X_emg']
    X_fmg = data['X_fmg']
    y_all = data['y']

    print(f"[INFO] Total Instances: {len(y_all):,} | Features: sEMG ({X_emg.shape[1]}), FMG ({X_fmg.shape[1]})")

    # Task Definitions
    mask_task1 = np.array([('f' in lbl or 'e' in lbl) and not lbl.startswith('6.') for lbl in y_all])
    mask_task2 = np.array([lbl.startswith('1.') for lbl in y_all])
    mask_task3 = np.ones(len(y_all), dtype=bool)

    tasks = {
        "Task 1: Flexion vs Extension": (mask_task1, X_emg, X_fmg, y_all),
        "Task 2: Single Finger Movements": (mask_task2, X_emg, X_fmg, y_all),
        "Task 3: All Active Classes (66 Classes)": (mask_task3, X_emg, X_fmg, y_all)
    }

    results = {}

    for task_name, (mask, emg_data, fmg_data, labels) in tasks.items():
        print("\n" + "-" * 70)
        print(f"Executing: {task_name}")
        print("-" * 70)

        X_e_task = emg_data[mask]
        X_f_task = fmg_data[mask]
        X_fus_task = np.hstack((X_e_task, X_f_task))
        y_task = labels[mask]

        # Stratified Train-Test Split (80% Train, 20% Test)
        idx = np.arange(len(y_task))
        idx_train, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_task
        )
        y_train, y_test = y_task[idx_train], y_task[idx_test]

        # Z-Score Standardization
        sc_e, sc_f, sc_fus = StandardScaler(), StandardScaler(), StandardScaler()
        X_e_tr = sc_e.fit_transform(X_e_task[idx_train])
        X_e_te = sc_e.transform(X_e_task[idx_test])

        X_f_tr = sc_f.fit_transform(X_f_task[idx_train])
        X_f_te = sc_f.transform(X_f_task[idx_test])

        X_fus_tr = sc_fus.fit_transform(X_fus_task[idx_train])
        X_fus_te = sc_fus.transform(X_fus_task[idx_test])

        t0 = time.time()

        # 1. sEMG Only
        model_emg = build_soft_voting_ensemble()
        model_emg.fit(X_e_tr, y_train)
        acc_emg = accuracy_score(y_test, model_emg.predict(X_e_te))

        # 2. FMG Only
        model_fmg = build_soft_voting_ensemble()
        model_fmg.fit(X_f_tr, y_train)
        acc_fmg = accuracy_score(y_test, model_fmg.predict(X_f_te))

        # 3. Multimodal Fusion (sEMG + FMG)
        model_fus = build_soft_voting_ensemble()
        model_fus.fit(X_fus_tr, y_train)
        acc_fus = accuracy_score(y_test, model_fus.predict(X_fus_te))

        elapsed = (time.time() - t0) / 60
        print(f"[RESULT] {task_name} | Fusion Accuracy: {acc_fus * 100:.2f}% (Time: {elapsed:.2f} min)")

        results[task_name] = {
            "sEMG Only (%)": round(acc_emg * 100, 2),
            "FMG Only (%)": round(acc_fmg * 100, 2),
            "Fusion (sEMG + FMG) (%)": round(acc_fus * 100, 2)
        }

    # Summary Report
    df_results = pd.DataFrame(results).T
    print("\n" + "=" * 70)
    print("       INTRA-SUBJECT HETEROGENEOUS ENSEMBLE RESULTS SUMMARY       ")
    print("=" * 70)
    print(df_results.to_string())
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()