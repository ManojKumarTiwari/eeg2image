#!/bin/bash
# Prepare BCIC-IV-2a data for EEG2Text
# If raw .mat files are already in data/BCICIV2a/raw/, use --skip_download

python prepare_data.py \
    --raw_dir data/BCICIV2a/raw \
    --lmdb_dir data/BCICIV2a/processed_lmdb
