"""
Feature Extraction Pipeline for Multimodal Hand Gesture Recognition (Zenodo Dataset).

This script processes raw sEMG and FMG signals recorded at 2000 Hz, applies sliding
window segmentation, computes a 12-feature 'Golden Standard' feature set per channel,
and saves the compressed feature tensors and metadata into an NPZ archive.
"""

import os
import time
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# Dataset & Signal Configuration
DATASET_DIR = './Data'
OUTPUT_FILE = 'zenodo_features.npz'
FS = 2000  # Sampling frequency in Hz
WIN_SIZE = 500  # Window length: 250 ms (500 samples @ 2000 Hz)
WIN_STEP = 250  # Step size: 125 ms (50% overlap)


def extract_golden_features(window: np.ndarray, fs: int = 2000) -> List[float]:
    """
    Extracts a 12-parameter hybrid feature set from a single-channel signal window.

    Parameters
    ----------
    window : np.ndarray
        1D signal array representing the temporal window.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    List[float]
        Extracted feature vector containing amplitude, complexity, and spectral metrics.
    """
    if len(window) < 32:
        return [0.0] * 12

    x = np.ascontiguousarray(window, dtype=np.float64)
    N = len(x)

    # 1. Amplitude & Energy Metrics
    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    wl = float(np.sum(np.abs(np.diff(x))))

    # 2. Dynamic-Threshold Morphology (Zero Crossing & Slope Sign Change)
    std_x = np.std(x)
    thresh = 0.05 * std_x
    sign_x = np.sign(x)
    sign_x[np.abs(x) < thresh] = 0
    sign_x = sign_x[sign_x != 0]
    zc = float(np.sum(np.abs(np.diff(sign_x)) == 2)) if len(sign_x) > 1 else 0.0

    dx = np.diff(x)
    std_dx = np.std(dx)
    thresh_dx = 0.05 * std_dx
    sign_dx = np.sign(dx)
    sign_dx[np.abs(dx) < thresh_dx] = 0
    sign_dx = sign_dx[sign_dx != 0]
    ssc = float(np.sum(np.abs(np.diff(sign_dx)) == 2)) if len(sign_dx) > 1 else 0.0

    # 3. Complexity, Fatigue & Hjorth Parameters
    logd = float(np.exp(np.mean(np.log(np.abs(x) + 1e-12))))
    var_x = float(np.var(x))
    act = var_x
    var_dx = float(np.var(dx)) if len(dx) > 0 else 0.0
    mob = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0

    ddx = np.diff(dx)
    var_ddx = float(np.var(ddx)) if len(ddx) > 0 else 0.0
    mob_dx = float(np.sqrt(var_ddx / var_dx)) if var_dx > 0 else 0.0
    comp = float(mob_dx / mob) if mob > 0 else 0.0

    # 4. Spectral Domain Metrics (Welch PSD)
    freqs, psd = welch(x, fs=fs, nperseg=min(N, 256))
    sum_psd = np.sum(psd)

    if sum_psd > 0:
        psd_norm = psd / sum_psd
        mnf = float(np.sum(freqs * psd_norm))
        cum_psd = np.cumsum(psd_norm)
        mdf_idx = np.where(cum_psd >= 0.5)[0]
        mdf = float(freqs[mdf_idx[0]]) if len(mdf_idx) > 0 else 0.0
        spec_ent = float(entropy(psd_norm + 1e-12))
    else:
        mnf, mdf, spec_ent = 0.0, 0.0, 0.0

    return [mav, rms, wl, zc, ssc, logd, act, mob, comp, mnf, mdf, spec_ent]


def parse_metadata(file_path: str) -> Tuple[str, str, str, str]:
    """
    Parses subject ID, gesture label, load condition, and arm position from file path.
    """
    parts = file_path.replace('\\', '/').split('/')
    subject = [p for p in parts if p.startswith('Par')][0]
    gesture = parts[-3]
    weight = parts[-2]
    filename = os.path.basename(file_path)
    position = filename.split(' ')[2].replace('.csv', '')
    return subject, gesture, weight, position


def main():
    print("=" * 70)
    print("      Multimodal Feature Extraction Pipeline (Zenodo Dataset)       ")
    print("=" * 70)
    print(f"[INFO] Scanning dataset directory: {DATASET_DIR}")
    start_time = time.time()

    X_emg_list, X_fmg_list = [], []
    y_gesture_list, y_weight_list, y_pos_list, y_subject_list = [], [], [], []
    processed_subjects = set()

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(root, file)

            try:
                subject, gesture, weight, position = parse_metadata(file_path)
            except (IndexError, ValueError):
                continue

            if subject not in processed_subjects:
                processed_subjects.add(subject)
                print(f"[{time.strftime('%H:%M:%S')}] Processing {subject}... ({len(processed_subjects)}/27 subjects)")

            try:
                df = pd.read_csv(file_path)
                data_vals = df.values
            except Exception:
                continue

            if len(data_vals) < WIN_SIZE:
                continue

            # Sliding Window Feature Extraction
            for start in range(0, len(data_vals) - WIN_SIZE, WIN_STEP):
                end = start + WIN_SIZE
                window_data = data_vals[start:end, :]

                # Channels 0-7: FMG (8 channels), Channels 8-15: sEMG (8 channels)
                fmg_features = []
                for ch in range(8):
                    fmg_features.extend(extract_golden_features(window_data[:, ch], fs=FS))

                emg_features = []
                for ch in range(8, 16):
                    emg_features.extend(extract_golden_features(window_data[:, ch], fs=FS))

                X_emg_list.append(emg_features)
                X_fmg_list.append(fmg_features)
                y_gesture_list.append(gesture)
                y_weight_list.append(weight)
                y_pos_list.append(position)
                y_subject_list.append(subject)

    print(f"\n[{time.strftime('%H:%M:%S')}] Feature extraction completed. Structuring arrays...")

    X_emg_all = np.array(X_emg_list, dtype=np.float32)
    X_fmg_all = np.array(X_fmg_list, dtype=np.float32)
    y_gesture_all = np.array(y_gesture_list)
    y_weight_all = np.array(y_weight_list)
    y_pos_all = np.array(y_pos_list)
    y_subject_all = np.array(y_subject_list)

    elapsed_time = (time.time() - start_time) / 60

    print(f"[INFO] Total Windows Extracted: {len(y_gesture_all):,}")
    print(f"[INFO] sEMG Feature Matrix Shape: {X_emg_all.shape}")
    print(f"[INFO] FMG Feature Matrix Shape : {X_fmg_all.shape}")
    print(f"[INFO] Total Processing Time    : {elapsed_time:.2f} min")

    print(f"[INFO] Saving compressed feature archive to '{OUTPUT_FILE}'...")
    np.savez_compressed(
        OUTPUT_FILE,
        X_emg=X_emg_all,
        X_fmg=X_fmg_all,
        y_gesture=y_gesture_all,
        y_weight=y_weight_all,
        y_pos=y_pos_all,
        y_subject=y_subject_all
    )
    print(f"[SUCCESS] Dataset successfully saved to '{OUTPUT_FILE}'.\n")


if __name__ == '__main__':
    main()