"""
prepare_data_vi_stimview.py — Stimulus-view variant of prepare_data_vi.py.

Extracts the 4-second window DURING which the image is shown (the period
immediately before the event trigger), rather than the 4-second imagery
window that follows the trigger.

Trial structure:
  |--- 3s fixation ---|--- 4s image shown ---|--- 4s imagery ---|--- 4s rest ---|
                                              ↑ trigger (latency)
                       ← this 4s window →

Window extracted: data[:, latency - 4000 : latency]

Output: data/VisualImagery/processed_lmdb_stimview/
        Same LMDB format as processed_lmdb — fully compatible with all
        training and generation scripts via --datasets_dir override.

Usage:
    python prepare_data_vi_stimview.py --zip_path /path/to/30227503.zip
    python prepare_data_vi_stimview.py --skip_extract
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

# ─── Constants (identical to prepare_data_vi.py) ──────────────────────────────

SUBJECTS = [f'sub-{i:02d}' for i in range(1, 23)]
SESSIONS = ['ses-01', 'ses-02']
TASKS    = ['AVI', 'FVI', 'OVI']

TASK_LABEL_OFFSET = {'AVI': 0, 'FVI': 3, 'OVI': 6}
TASK_MAX_VALUE    = {'AVI': 3, 'FVI': 3, 'OVI': 4}

VI_CLASS_NAMES = [
    'dog', 'bird', 'fish',
    'pentagram', 'square', 'circle',
    'scissor', 'watch', 'cup', 'chair',
]

VI_CHANNEL_NAMES = [
    'Fpz', 'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
    'FCz', 'FC3', 'FC4', 'FT7', 'FT8',
    'Cz', 'C3', 'C4', 'T7', 'T8',
    'CP3', 'CP4', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P7', 'P8',
    'PO3', 'PO4', 'Oz', 'O1', 'O2',
]
N_CHANNELS = 32

SFREQ         = 1000
TARGET_SFREQ  = 200
LOW_CUT       = 0.3
HIGH_CUT      = 50.0
FILTER_ORDER  = 5

TRIAL_DURATION_SEC      = 4.0
TRIAL_SAMPLES_ORIG      = int(TRIAL_DURATION_SEC * SFREQ)        # 4000
TRIAL_SAMPLES_RESAMPLED = int(TRIAL_DURATION_SEC * TARGET_SFREQ) # 800

N_PATCHES  = 4
PATCH_SIZE = 200   # 800 / 4
NORM_SCALE = 100.0

TRAIN_SUBJECTS = [f'sub-{i:02d}' for i in range(1,  17)]
VAL_SUBJECTS   = [f'sub-{i:02d}' for i in range(17, 20)]
TEST_SUBJECTS  = [f'sub-{i:02d}' for i in range(20, 23)]


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_dataset(zip_path: str, raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Extracting {zip_path} -> {raw_dir} ...")
    with zipfile.ZipFile(zip_path) as outer:
        for name in outer.namelist():
            if name.startswith('sub-') and name.endswith('.zip'):
                sub_id  = name.replace('.zip', '')
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
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1)


def read_events_tsv(tsv_path: str):
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
                    continue
                label = offset + (value - 1)
                events.append((latency, label))
            except (ValueError, KeyError):
                continue
    return events


def preprocess_bdf(bdf_path: str, events_tsv_path: str):
    """
    Extract the 4-second window BEFORE each event trigger (image-shown window).

    The trigger marks onset of imagery; stepping back 4000 samples gives the
    window during which the stimulus image was displayed.
    """
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

    available_ch = [ch.upper() for ch in raw.ch_names]
    picks = []
    for ch in VI_CHANNEL_NAMES:
        upper = ch.upper()
        if upper in available_ch:
            picks.append(raw.ch_names[available_ch.index(upper)])
        else:
            raise RuntimeError(f"Channel '{ch}' not found in BDF. Available: {raw.ch_names}")
    raw.pick(picks)

    data = raw.get_data()
    data = data * 1e6
    data = data - data.mean(axis=1, keepdims=True)
    data = bandpass_filter(data, LOW_CUT, HIGH_CUT, SFREQ)

    events = read_events_tsv(events_tsv_path)

    samples = []
    for latency, label in events:
        seg_start = latency - TRIAL_SAMPLES_ORIG   # 4s before trigger
        if seg_start < 0:
            # Trigger too early in the recording to have a full 4s before it
            continue
        seg = data[:, seg_start:latency]             # (32, 4000)
        seg = resample(seg, TRIAL_SAMPLES_RESAMPLED, axis=1)  # (32, 800)
        seg = seg.reshape(N_CHANNELS, N_PATCHES, PATCH_SIZE)   # (32, 4, 200)
        seg = seg / NORM_SCALE
        samples.append((seg.astype(np.float32), label))

    return samples


def preprocess_subject(subject: str, raw_dir: str):
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
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(5e9))

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
        description='Prepare EEG Visual Imagery dataset — stimulus-view window (4s image shown)')
    parser.add_argument('--zip_path',     type=str,
                        default='/c/Users/manoj/Downloads/30227503.zip')
    parser.add_argument('--raw_dir',      type=str,
                        default='data/VisualImagery/raw')
    parser.add_argument('--lmdb_dir',     type=str,
                        default='data/VisualImagery/processed_lmdb_stimview')
    parser.add_argument('--skip_extract', action='store_true')
    args = parser.parse_args()

    if not args.skip_extract:
        extract_dataset(args.zip_path, args.raw_dir)
    else:
        print("Skipping extraction.")

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

    print(f"\nWriting LMDB to {args.lmdb_dir} ...")
    write_lmdb(args.lmdb_dir, split_data)
    print("\nData preparation complete.")
    print(f"Window: 4s image-shown (latency-4000 : latency)")
    print(f"Sample shape: (32, 4, 200) | Classes: {VI_CLASS_NAMES}")


if __name__ == '__main__':
    main()
