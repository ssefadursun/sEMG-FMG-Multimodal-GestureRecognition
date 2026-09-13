"""
Feature Extraction Pipeline for Multimodal Gesture Recognition (GREFTUD Dataset).

This script reads continuous sEMG and piezoelectric FMG signals from 'dataset.hdf5',
segments active gesture intervals using a 250 ms sliding window with 50 ms step size,
extracts a 12-parameter hybrid feature set per channel, and saves the compressed
dataset to 'greftud_features.npz'.
"""

import os
import time
import warnings
from typing import List
import h5py
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# Dataset & Signal Configuration
HDF5_FILE = './dataset.hdf5'
OUTPUT_FILE = 'greftud_features.npz'
FS = 1000  # Sampling frequency in Hz
WIN_SIZE = 250  # Window length: 250 ms (250 samples @ 1000 Hz)
WIN_STEP = 50  # Step size: 50 ms (80% overlap)


def extract_golden_features(window: np.ndarray, fs: int = 1000) -> List[float]:
    """
    Extracts a 12-parameter hybrid feature set from a single-channel signal window.

    Parameters
    ----------
    window : np.ndarray
        1D array representing the temporal window.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    List[float]
        Extracted feature vector (amplitude, morphology, complexity, spectral metrics).
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

    # 3. Log-Detector & Hjorth Parameters
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


def main():
    print("=" * 70)
    print("       Multimodal Feature Extraction Pipeline (GREFTUD Dataset)       ")
    print("=" * 70)

    if not os.path.exists(HDF5_FILE):
        raise FileNotFoundError(f"HDF5 dataset not found at '{HDF5_FILE}'.")

    print(f"[INFO] Reading HDF5 database: {HDF5_FILE}")
    start_time = time.time()

    X_emg_list, X_fmg_list, y_list, subjects_list = [], [], [], []

    with h5py.File(HDF5_FILE, 'r') as f:
        subject_keys = [k for k in f.keys() if k != 'labels_to_fingers_moved']

        for idx, subj in enumerate(subject_keys):
            grp = f[subj]
            time_ms = grp['time (ms)'][:]
            labels = grp['label'][:]
            timestamps = grp['label_timestamps (s)'][:]

            emg_channels = [grp[f'EMG_Ch{i} (mV)'][:] for i in range(1, 5)]
            fmg_channels = [grp[f'FMG_Ch{i} (mV)'][:] for i in range(1, 5)]

            for i in range(len(labels)):
                if labels[i] in [b'0', 0, b'0.0']:
                    continue

                try:
                    start_idx = np.argwhere(time_ms >= timestamps[i] * 1000).flatten()[0]
                    end_idx = (
                        np.argwhere(time_ms >= timestamps[i + 1] * 1000).flatten()[0]
                        if (i + 1) < len(timestamps)
                        else len(time_ms)
                    )
                except IndexError:
                    continue

                label_str = labels[i].decode('utf-8') if isinstance(labels[i], bytes) else str(labels[i])

                for st in range(start_idx, end_idx - WIN_SIZE, WIN_STEP):
                    en = st + WIN_SIZE

                    emg_feat, fmg_feat = [], []
                    for ch in range(4):
                        emg_feat.extend(extract_golden_features(emg_channels[ch][st:en], fs=FS))
                    for ch in range(4):
                        fmg_feat.extend(extract_golden_features(fmg_channels[ch][st:en], fs=FS))

                    X_emg_list.append(emg_feat)
                    X_fmg_list.append(fmg_feat)
                    y_list.append(label_str)
                    subjects_list.append(subj)

            print(f"[{idx + 1:02d}/{len(subject_keys):02d}] Subject '{subj}' processed.")

    X_emg_all = np.array(X_emg_list, dtype=np.float32)
    X_fmg_all = np.array(X_fmg_list, dtype=np.float32)
    y_all = np.array(y_list)
    subjects_all = np.array(subjects_list)

    elapsed = (time.time() - start_time) / 60

    print(f"\n[INFO] Feature extraction completed in {elapsed:.2f} min.")
    print(f"[INFO] Total Windows Extracted: {len(y_all):,}")
    print(f"[INFO] sEMG Shape: {X_emg_all.shape} | FMG Shape: {X_fmg_all.shape}")

    print(f"[INFO] Saving compressed arrays to '{OUTPUT_FILE}'...")
    np.savez_compressed(
        OUTPUT_FILE,
        X_emg=X_emg_all,
        X_fmg=X_fmg_all,
        y=y_all,
        subjects=subjects_all
    )
    print(f"[SUCCESS] Dataset successfully saved to '{OUTPUT_FILE}'.\n")


if __name__ == '__main__':
    main()