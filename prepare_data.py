"""
Data preparation script for EEG2Text project.

Downloads and preprocesses BCIC-IV-2a motor imagery dataset into LMDB format.

Usage:
    # Download + process:
    python prepare_data.py

    # If .mat files already exist in data/BCICIV2a/raw/:
    python prepare_data.py --skip_download
"""

import os
import ssl
import pickle
import argparse
import urllib.request
import numpy as np
import scipy.io
import lmdb
from scipy.signal import butter, lfilter, resample

# ─── Constants ────────────────────────────────────────────────────────────────

SUBJECTS = [f'A0{i}' for i in range(1, 10)]   # A01 – A09
SESSIONS = ['T', 'E']                           # Train, Eval

BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"

EEG_CHANNELS = list(range(22))   # 22 EEG channels (excludes EOG)

SFREQ = 250
TARGET_SFREQ = 200
LOW_CUT = 0.3
HIGH_CUT = 50.0
FILTER_ORDER = 5

MI_START_SEC = 2.0
MI_END_SEC = 6.0
MI_SAMPLES_ORIG = int((MI_END_SEC - MI_START_SEC) * SFREQ)          # 1000
MI_SAMPLES_RESAMPLED = int((MI_END_SEC - MI_START_SEC) * TARGET_SFREQ)  # 800

N_PATCHES = 4
PATCH_SIZE = 200   # 800 / 4
NORM_SCALE = 100.0

TRAIN_SUBJECTS = ['A01', 'A02', 'A03', 'A04', 'A05']
VAL_SUBJECTS   = ['A06', 'A07']
TEST_SUBJECTS  = ['A08', 'A09']

# ─── Download ─────────────────────────────────────────────────────────────────

def download_data(raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    for subject in SUBJECTS:
        for session in SESSIONS:
            filename = f"{subject}{session}.mat"
            filepath = os.path.join(raw_dir, filename)
            if os.path.exists(filepath):
                print(f"  Already exists: {filename}")
                continue
            url = BASE_URL + filename
            print(f"  Downloading {filename} ...")
            with urllib.request.urlopen(url, context=ssl_ctx) as r:
                data = r.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            print(f"  Saved {filename} ({len(data)/1e6:.1f} MB)")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data, axis=-1)


def preprocess_subject(mat_path: str):
    """Returns list of (eeg_array, label) where eeg_array: float32 (22, 4, 200)."""
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data_struct = mat['data']

    samples = []
    for run in data_struct:
        try:
            X = run.X
            trial = run.trial
            y = run.y
            fs = run.fs
        except AttributeError:
            continue

        X = X[:, EEG_CHANNELS].T   # (22, n_samples)
        X = X - X.mean(axis=1, keepdims=True)
        X = bandpass_filter(X, LOW_CUT, HIGH_CUT, fs)

        for t_idx, t_start in enumerate(trial):
            t_start = int(t_start)
            cue_offset = int(MI_START_SEC * fs)
            seg_start = t_start + cue_offset
            seg_end = seg_start + MI_SAMPLES_ORIG
            if seg_end > X.shape[1]:
                continue

            seg = X[:, seg_start:seg_end]
            seg = resample(seg, MI_SAMPLES_RESAMPLED, axis=1)
            seg = seg.reshape(22, N_PATCHES, PATCH_SIZE)
            seg = seg / NORM_SCALE

            label = int(y[t_idx]) - 1   # 0-indexed: {0,1,2,3}
            samples.append((seg.astype(np.float32), label))

    return samples


# ─── LMDB Writing ─────────────────────────────────────────────────────────────

def write_lmdb(lmdb_path: str, split_data: dict):
    """
    Write all splits into a single LMDB file.

    Format (matches bciciv2a_dataset.py and faced_llm_dataset.py):
      '__keys__'   -> pickle({'train': [...], 'val': [...], 'test': [...]})
      '<key>'      -> pickle({'sample': np.ndarray, 'label': int})
    """
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(2e10))

    keys_dict = {}
    with env.begin(write=True) as txn:
        for split_name, samples in split_data.items():
            split_keys = []
            for i, (eeg, label) in enumerate(samples):
                key = f"{split_name}_{i:06d}"
                val = pickle.dumps({'sample': eeg, 'label': label})
                txn.put(key.encode(), val)
                split_keys.append(key)
            keys_dict[split_name] = split_keys
            print(f"  {split_name}: {len(split_keys)} samples")

        txn.put('__keys__'.encode(), pickle.dumps(keys_dict))

    env.close()
    print(f"LMDB written to: {lmdb_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/BCICIV2a/raw')
    parser.add_argument('--lmdb_dir', type=str, default='data/BCICIV2a/processed_lmdb')
    parser.add_argument('--skip_download', action='store_true')
    args = parser.parse_args()

    if not args.skip_download:
        print("Downloading BCIC-IV-2a raw data...")
        download_data(args.raw_dir)
    else:
        print("Skipping download (using existing .mat files).")

    splits = {
        'train': TRAIN_SUBJECTS,
        'val':   VAL_SUBJECTS,
        'test':  TEST_SUBJECTS,
    }

    split_data = {}
    for split_name, subject_list in splits.items():
        print(f"\nProcessing {split_name} split: {subject_list}")
        all_samples = []
        for subject in subject_list:
            for session in SESSIONS:
                mat_path = os.path.join(args.raw_dir, f"{subject}{session}.mat")
                if not os.path.exists(mat_path):
                    print(f"  Warning: {mat_path} not found, skipping")
                    continue
                print(f"  Processing {subject}{session}.mat ...")
                samples = preprocess_subject(mat_path)
                print(f"    → {len(samples)} trials")
                all_samples.extend(samples)
        split_data[split_name] = all_samples

    print(f"\nWriting LMDB...")
    write_lmdb(args.lmdb_dir, split_data)
    print("\nData preparation complete.")


if __name__ == '__main__':
    main()
