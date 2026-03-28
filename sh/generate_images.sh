#!/usr/bin/env bash
# generate_images.sh — Full EEG → Text → Image pipeline
#
# Requires a trained model in pth_downtasks/eeg_llm_bcic_new/
# Run from the project root directory.

EPOCH=6   # change to match your best checkpoint epoch

python generate.py \
    --foundation_dir   pth/CSBrain.pth \
    --projection_dir   pth_downtasks/eeg_llm_bcic_new/projection_epoch${EPOCH}.pth \
    --lora_dir         pth_downtasks/eeg_llm_bcic_new/lora_epoch${EPOCH} \
    --datasets_dir     data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples      8 \
    --max_new_tokens   64 \
    --generate_images \
    --image_model      stabilityai/stable-diffusion-2-1 \
    --num_inference_steps 25 \
    --guidance_scale   7.5 \
    --image_height     512 \
    --image_width      512 \
    --output_dir       outputs/eeg2image \
    --seed             42 \
    --cuda             0
