#!/bin/bash
# Fine-tune EEG-LLM on FACED emotion recognition (EEG → text generation)
# Run from the EEG2Text project root

python finetune_eeg_llm.py \
    --downstream_dataset FACED \
    --datasets_dir data/FACED/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_faced \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --llm_model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --llm_dim 2048 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --clip_value 1.0 \
    --dropout 0.1 \
    --max_target_len 128 \
    --temporal_pool_stride 2 \
    --num_of_classes 9 \
    --seed 42 \
    --cuda 0
