"""
Data preparation script for EEG Visual Imagery dataset.

Extracts, preprocesses, and stores the Figshare Visual Imagery EEG dataset
(doi:10.6084/m9.figshare.30227503) into LMDB format compatible with the
existing BCICIV2A pipeline.

Dataset structure (inside 30227503.zip):
  sub-01.zip ... sub-22.zip
  Each sub-XX.zip: sub-XX/ses-01/eeg/ and sub-XX/ses-02/eeg/
  Three BDF tasks per session:
    task-AVI (Animals): dog=1, bird=2, fish=3
    task-FVI (Figures): pentagram=1, square=2, circle=3
    task-OVI (Objects): scissor=1, watch=2, cup=3, chair=4

Global label mapping (0-9):
  0: dog     1: bird    2: fish
  3: pentagram  4: square   5: circle
  6: scissor  7: watch   8: cup   9: chair

Usage:
    # Extract zip + preprocess:
    python prepare_data_vi.py --zip_path /path/to/30227503.zip

    # If already extracted to data/VisualImagery/raw/:
    python prepare_data_vi.py --skip_extract
"""

import os
import io
import csv
import pickle
import argparse
import zipfile
import numpy as np
import lmdb
import mne
from scipy.signal import butter, sosfiltfilt, resample

mne.set_log_level('WARNING')

# ─── Constants ────────────────────────────────────────────────────────────────

# 22 subjects (sub-01 to sub-22)
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 23)]
SESSIONS = ['ses-01', 'ses-02']
TASKS    = ['AVI', 'FVI', 'OVI']

# Global label offsets per task (event values within each task are 1-indexed)
TASK_LABEL_OFFSET = {'AVI': 0, 'FVI': 3, 'OVI': 6}

# Maximum valid event value per task (filters spurious triggers)
TASK_MAX_VALUE = {'AVI': 3, 'FVI': 3, 'OVI': 4}

# 10 global class names
VI_CLASS_NAMES = [
    'dog', 'bird', 'fish',            # AVI: 0, 1, 2
    'pentagram', 'square', 'circle',  # FVI: 3, 4, 5
    'scissor', 'watch', 'cup', 'chair',  # OVI: 6, 7, 8, 9
]

# 32 channel names in order (from electrodes.tsv)
VI_CHANNEL_NAMES = [
    'Fpz', 'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',    # 0-7  Frontal
    'FCz', 'FC3', 'FC4', 'FT7', 'FT8',                      # 8-12 Fronto-central
    'Cz', 'C3', 'C4', 'T7', 'T8',                           # 13-17 Central
    'CP3', 'CP4', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P7', 'P8',  # 18-26 Parietal
    'PO3', 'PO4', 'Oz', 'O1', 'O2',                         # 27-31 Occipital
]
N_CHANNELS = 32

# Preprocessing parameters
SFREQ = 1000           # Original sampling frequency
TARGET_SFREQ = 200     # Target sampling frequency
LOW_CUT = 0.3
HIGH_CUT = 50.0
FILTER_ORDER = 5

# Trial window: use 4-second imagery epoch from event onset
TRIAL_DURATION_SEC = 4.0
TRIAL_SAMPLES_ORIG = int(TRIAL_DURATION_SEC * SFREQ)           # 4000
TRIAL_SAMPLES_RESAMPLED = int(TRIAL_DURATION_SEC * TARGET_SFREQ)  # 800

N_PATCHES = 4
PATCH_SIZE = 200   # 800 / 4
NORM_SCALE = 100.0

# Subject splits (by subject index, 1-based)
TRAIN_SUBJECTS = [f'sub-{i:02d}' for i in range(1,  17)]  # sub-01 to sub-16
VAL_SUBJECTS   = [f'sub-{i:02d}' for i in range(17, 20)]  # sub-17 to sub-19
TEST_SUBJECTS  = [f'sub-{i:02d}' for i in range(20, 23)]  # sub-20 to sub-22


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_dataset(zip_path: str, raw_dir: str):
    """Extract 30227503.zip -> raw_dir/sub-XX/ directory tree."""
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Extracting {zip_path} -> {raw_dir} ...")
    with zipfile.ZipFile(zip_path) as outer:
        for name in outer.namelist():
            if name.startswith('sub-') and name.endswith('.zip'):
                sub_id = name.replace('.zip', '')
                sub_dir = os.path.join(raw_dir, sub_id)
                if os.path.exists(sub_dir):
                    print(f"  Already extracted: {sub_id}")
                    continue
                print(f"  Extracting {name} ...")
                sub_zip_bytes = io.BytesIO(outer.read(name))
                with zipfile.ZipFile(sub_zip_bytes) as inner:
                    inner.extractall(raw_dir)
    print("Extraction complete.")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Zero-phase SOS Butterworth filter (numerically stable for long recordings)."""
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1)


def read_events_tsv(tsv_path: str):
    """Read events TSV, return list of (latency_samples, global_label) tuples."""
    task = None
    for t in TASKS:
        if f'task-{t}_' in tsv_path:
            task = t
            break
    if task is None:
        raise ValueError(f"Cannot determine task from path: {tsv_path}")

    offset = TASK_LABEL_OFFSET[task]
    events = []
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                latency = int(row['latency'])
                value   = int(row['value'])
                if value < 1 or value > TASK_MAX_VALUE[task]:
                    continue   # skip spurious trigger codes
                label   = offset + (value - 1)   # 0-indexed global label
                events.append((latency, label))
            except (ValueError, KeyError):
                continue
    return events


def preprocess_bdf(bdf_path: str, events_tsv_path: str):
    """
    Loads BDF, applies bandpass filter, extracts 4-second trial windows,
    resamples, reshapes and normalises.

    Returns list of (eeg_array, label) where eeg_array: float32 (32, 4, 200).
    """
    # Load BDF (32 EEG channels only)
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

    # Pick the 32 EEG channels in the standard order
    available_ch = [ch.upper() for ch in raw.ch_names]
    picks = []
    for ch in VI_CHANNEL_NAMES:
        upper = ch.upper()
        if upper in available_ch:
            picks.append(raw.ch_names[available_ch.index(upper)])
        else:
            raise RuntimeError(f"Channel '{ch}' not found in BDF. Available: {raw.ch_names}")
    raw.pick(picks)

    # Get continuous data: (32, total_samples)
    data = raw.get_data()                    # volts
    data = data * 1e6                        # convert to µV
    data = data - data.mean(axis=1, keepdims=True)   # zero-mean
    data = bandpass_filter(data, LOW_CUT, HIGH_CUT, SFREQ)

    # Read events from TSV
    events = read_events_tsv(events_tsv_path)

    samples = []
    for latency, label in events:
        seg_end = latency + TRIAL_SAMPLES_ORIG
        if seg_end > data.shape[1]:
            continue
        seg = data[:, latency:seg_end]           # (32, 4000)
        seg = resample(seg, TRIAL_SAMPLES_RESAMPLED, axis=1)  # (32, 800)
        seg = seg.reshape(N_CHANNELS, N_PATCHES, PATCH_SIZE)   # (32, 4, 200)
        seg = seg / NORM_SCALE
        samples.append((seg.astype(np.float32), label))

    return samples


def preprocess_subject(subject: str, raw_dir: str):
    """Process all sessions and tasks for one subject."""
    all_samples = []
    for session in SESSIONS:
        for task in TASKS:
            bdf_path = os.path.join(
                raw_dir, subject, session, 'eeg',
                f'{subject}_{session}_task-{task}_eeg.bdf'
            )
            tsv_path = os.path.join(
                raw_dir, subject, session, 'eeg',
                f'{subject}_{session}_task-{task}_events.tsv'
            )
            if not os.path.exists(bdf_path):
                print(f"    Warning: {bdf_path} not found, skipping")
                continue
            if not os.path.exists(tsv_path):
                print(f"    Warning: {tsv_path} not found, skipping")
                continue
            print(f"    Processing {subject}/{session}/task-{task} ...")
            samples = preprocess_bdf(bdf_path, tsv_path)
            print(f"      -> {len(samples)} trials")
            all_samples.extend(samples)
    return all_samples


# ─── LMDB Writing ─────────────────────────────────────────────────────────────

def write_lmdb(lmdb_path: str, split_data: dict):
    """Same format as BCICIV2A LMDB for compatibility."""
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(5e9))    # 5 GB max

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
    parser = argparse.ArgumentParser(
        description='Prepare EEG Visual Imagery dataset for CSBrain training')
    parser.add_argument('--zip_path',    type=str,
                        default='/c/Users/manoj/Downloads/30227503.zip',
                        help='Path to downloaded 30227503.zip')
    parser.add_argument('--raw_dir',     type=str,
                        default='data/VisualImagery/raw',
                        help='Where to extract raw BDF files')
    parser.add_argument('--lmdb_dir',    type=str,
                        default='data/VisualImagery/processed_lmdb',
                        help='Where to write the LMDB database')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip extraction if BDFs already in raw_dir')
    args = parser.parse_args()

    # Step 1: Extract
    if not args.skip_extract:
        extract_dataset(args.zip_path, args.raw_dir)
    else:
        print("Skipping extraction.")

    # Step 2: Preprocess by split
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
            print(f"  Subject {subject} ...")
            samples = preprocess_subject(subject, args.raw_dir)
            print(f"  -> {len(samples)} total trials for {subject}")
            all_samples.extend(samples)
        split_data[split_name] = all_samples

    # Step 3: Write LMDB
    print(f"\nWriting LMDB to {args.lmdb_dir} ...")
    write_lmdb(args.lmdb_dir, split_data)
    print("\nData preparation complete.")
    print(f"Sample shape: (32, 4, 200) | Classes: {VI_CLASS_NAMES}")


if __name__ == '__main__':
    main()
