"""
Leave-One-Out Cross-Validation (LOOCV) Benchmark (GREFTUD Dataset).

Performs 13-fold inter-subject cross-validation with subject-wise Z-score
normalization using a regularized ensemble (RF + LightGBM + Extra Trees) to evaluate
cross-user generalization performance across all 3 benchmark tasks.
"""

import time
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configuration
NPZ_FILE = 'greftud_features.npz'
CSV_DETAIL_PATH = 'loocv_fold_details.csv'
CSV_SUMMARY_PATH = 'loocv_summary_report.csv'
RANDOM_STATE = 42


def build_regularized_loocv_model() -> VotingClassifier:
    """
    Constructs a regularized ensemble to minimize inter-subject variance and overfitting.
    """
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    lgbm = LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    )

    return VotingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm), ('et', et)],
        voting='soft'
    )


def prepare_tasks(
    X_emg: np.ndarray, X_fmg: np.ndarray, y_all: np.ndarray, subjects_all: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Standardizes labels and partitions feature tensors for the 3 benchmark tasks.
    """
    # 1. Subject-wise Z-Score Standardization (Inter-subject domain alignment)
    unique_subjects = np.unique(subjects_all)
    X_e_norm = np.zeros_like(X_emg)
    X_f_norm = np.zeros_like(X_fmg)

    for subj in unique_subjects:
        s_mask = (subjects_all == subj)
        X_e_norm[s_mask] = StandardScaler().fit_transform(X_emg[s_mask])
        X_f_norm[s_mask] = StandardScaler().fit_transform(X_fmg[s_mask])

    # Task 1: Flexion vs Extension (Binary: 2 Classes)
    m1 = np.array([('f' in lbl or 'e' in lbl) and not lbl.startswith('6.') for lbl in y_all])
    y_t1 = np.array(['Flexion' if 'f' in lbl else 'Extension' for lbl in y_all[m1]])

    # Task 2: Single Finger Movements (5 Classes)
    m2 = np.array([lbl.startswith('1.') and len(lbl) >= 4 and lbl[2] in ['1', '2', '3', '4', '5'] for lbl in y_all])
    y_t2 = np.array([f"Finger_{lbl[2]}" for lbl in y_all[m2]])

    # Task 3: All Active Classes (66 Classes)
    m3 = np.ones(len(y_all), dtype=bool)
    y_t3 = np.array([lbl[:-1] if lbl[-1] in ['s', 'n', 'q'] else lbl for lbl in y_all])

    return {
        'Task 1: Flexion vs Extension (2 Classes)': (X_e_norm[m1], X_f_norm[m1], y_t1, subjects_all[m1]),
        'Task 2: Single Finger Movements (5 Classes)': (X_e_norm[m2], X_f_norm[m2], y_t2, subjects_all[m2]),
        'Task 3: All Active Classes (66 Classes)': (X_e_norm[m3], X_f_norm[m3], y_t3, subjects_all[m3])
    }


def main():
    print("=" * 75)
    print("   Inter-Subject Leave-One-Out Cross-Validation (LOOCV) Benchmark   ")
    print("=" * 75)

    print(f"\n[INFO] Loading pre-extracted features from '{NPZ_FILE}'...")
    data = np.load(NPZ_FILE, allow_pickle=True)
    X_emg = data['X_emg']
    X_fmg = data['X_fmg']
    y_all = data['y']
    subjects_all = data['subjects']

    tasks = prepare_tasks(X_emg, X_fmg, y_all, subjects_all)
    unique_subjects = np.unique(subjects_all)

    fold_results = []
    summary_results = {}

    for task_name, (X_e, X_f, y, subjs) in tasks.items():
        print("\n" + "=" * 75)
        print(f">>> LOOCV BENCHMARK: {task_name} (Samples: {len(y):,}, Classes: {len(np.unique(y))}) <<<")
        print("=" * 75)

        X_fus = np.hstack((X_e, X_f))
        acc_e_list, acc_f_list, acc_fus_list = [], [], []

        for fold_idx, test_subj in enumerate(unique_subjects):
            t_start = time.time()
            train_idx = (subjs != test_subj)
            test_idx = (subjs == test_subj)

            y_train, y_test = y[train_idx], y[test_idx]

            # 1. sEMG Only
            m_e = build_regularized_loocv_model()
            m_e.fit(X_e[train_idx], y_train)
            acc_e = accuracy_score(y_test, m_e.predict(X_e[test_idx]))

            # 2. FMG Only
            m_f = build_regularized_loocv_model()
            m_f.fit(X_f[train_idx], y_train)
            acc_f = accuracy_score(y_test, m_f.predict(X_f[test_idx]))

            # 3. Multimodal Fusion
            m_fus = build_regularized_loocv_model()
            m_fus.fit(X_fus[train_idx], y_train)
            acc_fus = accuracy_score(y_test, m_fus.predict(X_fus[test_idx]))

            acc_e_list.append(acc_e)
            acc_f_list.append(acc_f)
            acc_fus_list.append(acc_fus)

            duration = time.time() - t_start
            fold_results.append({
                'Task': task_name,
                'Fold': fold_idx + 1,
                'Subject': test_subj,
                'sEMG_Accuracy (%)': round(acc_e * 100, 2),
                'FMG_Accuracy (%)': round(acc_f * 100, 2),
                'Fusion_Accuracy (%)': round(acc_fus * 100, 2),
                'Duration (s)': round(duration, 1)
            })
            pd.DataFrame(fold_results).to_csv(CSV_DETAIL_PATH, index=False)

            print(
                f"[{fold_idx + 1:02d}/13] Subject: {test_subj:8s} | "
                f"sEMG: {acc_e * 100:5.2f}% | FMG: {acc_f * 100:5.2f}% | "
                f"Fusion: {acc_fus * 100:5.2f}% ({duration:.1f} s)"
            )

        summary_results[task_name] = {
            'sEMG Mean (%)': round(np.mean(acc_e_list) * 100, 2),
            'sEMG Std (%)': round(np.std(acc_e_list) * 100, 2),
            'FMG Mean (%)': round(np.mean(acc_f_list) * 100, 2),
            'FMG Std (%)': round(np.std(acc_f_list) * 100, 2),
            'Fusion Mean (%)': round(np.mean(acc_fus_list) * 100, 2),
            'Fusion Std (%)': round(np.std(acc_fus_list) * 100, 2)
        }

    # Summary Report
    df_sum = pd.DataFrame(summary_results).T
    df_sum.to_csv(CSV_SUMMARY_PATH)

    print("\n" + "=" * 75)
    print("          LOOCV BENCHMARK SUMMARY REPORT (13 FOLDS)          ")
    print("=" * 75)
    print(df_sum.to_string())
    print("=" * 75 + "\n")
    print(f"[INFO] Fold-level details saved to : '{CSV_DETAIL_PATH}'")
    print(f"[INFO] Summary statistics saved to : '{CSV_SUMMARY_PATH}'\n")


if __name__ == '__main__':
    main()