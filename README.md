# EEG2Image — EEG-to-Image Generation via Language Decoding

A two-stage pipeline that decodes raw EEG brain signals into natural language
descriptions and then synthesises corresponding visual images using an open-source
diffusion model — no proprietary models or closed licences required.

```
EEG Signal  →  [Stage 1] CSBrain + TinyLlama  →  Text Description
                                                        │
                                               [Stage 2] Stable Diffusion 2.1
                                                        │
                                                    PNG Image
```

---

## Licence

All models used are fully open source:

| Model | Licence |
|-------|---------|
| CSBrain encoder | Research (NeurIPS 2025) |
| TinyLlama-1.1B-Chat | Apache 2.0 |
| Stable Diffusion 2.1 | **Apache 2.0** |

---

## Architecture

### Stage 1 — EEG → Text

```
EEG Signal  (batch, channels, patches, patch_size)
        │
        ▼
┌─────────────────────────────────────────┐
│  CSBrain Encoder  [FROZEN]              │
│  12-layer transformer, d_model=200      │
│  Cross-scale temporal + region spatial  │
│  attention                              │
└─────────────────────────────────────────┘
        │  (batch, n_channels, n_patches, 200)
        ▼
┌─────────────────────────────────────────┐
│  EEGTokenReducer                        │
│  Pools channels within brain regions    │
│  BCIC: 3 regions × 4 patches = 12 tok  │
│  FACED: 5 regions × patches = ~75 tok  │
└─────────────────────────────────────────┘
        │  (batch, n_tokens, 200)
        ▼
┌─────────────────────────────────────────┐
│  EEGProjection  [TRAINABLE]             │
│  2-layer MLP: 200 → 2048 → 2048        │
└─────────────────────────────────────────┘
        │  (batch, n_tokens, 2048)
        ▼
┌─────────────────────────────────────────┐
│  TinyLlama-1.1B-Chat  [LoRA ADAPTER]   │
│  4-bit NF4 quantisation                 │
│  LoRA r=8, α=16 on q_proj & v_proj     │
│  ~1.1M trainable params (0.10%)        │
└─────────────────────────────────────────┘
        │
        ▼
  Neuroscience text description
  e.g. "EEG shows motor imagery consistent with left hand movement.
        Contralateral right hemisphere activation observed…"
```

### Stage 2 — Text → Image

```
Text Description
        │
        ▼
┌─────────────────────────────────────────┐
│  Prompt Builder                         │
│  Maps class label + key terms to a      │
│  structured visual prompt               │
└─────────────────────────────────────────┘
        │  Visual prompt string
        ▼
┌─────────────────────────────────────────┐
│  Stable Diffusion 2.1  [FROZEN]         │
│  stabilityai/stable-diffusion-2-1       │
│  Apache 2.0 licence                     │
│  DPMSolver++ scheduler (25 steps)       │
│  CFG scale = 7.5                        │
│  Output: 512 × 512 PNG                  │
└─────────────────────────────────────────┘
        │
        ▼
  PNG image in outputs/eeg2image/
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA GPU
  - Stage 1 (training): 8 GB+ VRAM (tested on RTX 4060 Laptop 8 GB)
  - Stage 2 (inference, SD 2.1 fp16): ~4 GB additional VRAM
    Both stages fit on an 8 GB GPU when run sequentially (LLM is freed before SD loads)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

```bash
python prepare_data.py --skip_download   # if .mat files are already in data/BCICIV2a/raw/
# or
python prepare_data.py                   # downloads ~1.4 GB automatically
```

This creates `data/BCICIV2a/processed_lmdb/` with 2,784 train / 1,152 val / 1,152 test
samples.

### 3. Train Stage 1 (EEG → Text)

```bash
bash sh/finetune_eeg_llm_bcic.sh
```

Or run directly:

```bash
python finetune_eeg_llm.py \
    --downstream_dataset BCICIV2a \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_bcic_new \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --cuda 0
```

Training runs in two phases (~30–60 min on RTX 4060):

```
Phase 1: Projection warmup (5 epochs)
  Epoch 1 [warmup]: Loss=2.3412, LR=0.001000, Time=4.2min
  Epoch 1 Val Accuracy: 0.2769 (319/1152)
  …
Phase 2: Joint projection + LoRA training (15 epochs)
  Epoch 6 [joint]: Loss=1.8734, LR=0.000180, Time=5.1min
  Epoch 6 Val Accuracy: 0.3681 (424/1152)  ← best, saving…
  …
Test Accuracy: 0.3134 (361/1152)
```

Weights are saved to `pth_downtasks/eeg_llm_bcic_new/`:
- `projection_epoch6.pth` — EEGProjection + EEGTokenReducer
- `lora_epoch6/` — LoRA adapter (HuggingFace PEFT format)

### 4. Run the full EEG → Text → Image pipeline

```bash
bash sh/generate_images.sh
```

Or run directly:

```bash
python generate.py \
    --foundation_dir pth/CSBrain.pth \
    --projection_dir pth_downtasks/eeg_llm_bcic_new/projection_epoch6.pth \
    --lora_dir pth_downtasks/eeg_llm_bcic_new/lora_epoch6 \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples 8 \
    --generate_images \
    --image_model stabilityai/stable-diffusion-2-1 \
    --num_inference_steps 25 \
    --output_dir outputs/eeg2image
```

Expected output:

```
============================================================
Stage 1: EEG → Text
============================================================
Sample 1:
  True class : 0 — left hand
  Generated  : The EEG shows motor imagery consistent with left hand movement.
               Contralateral right hemisphere ERD observed in sensorimotor cortex…

Sample 2:
  True class : 2 — feet
  Generated  : Bilateral central midline activation consistent with feet motor imagery.
               Strong CZ and CPZ involvement, supplementary motor area active…

============================================================
Stage 2: Text → Image  (Stable Diffusion 2.1, Apache 2.0)
============================================================
Loading image generation model: stabilityai/stable-diffusion-2-1 …
Image generator ready on cuda.
Generating 8 image(s) …
  Saved: outputs/eeg2image/eeg2image_0000_label0.png
  Saved: outputs/eeg2image/eeg2image_0001_label2.png
  …
  Prompt log: outputs/eeg2image/prompts.txt

Generated 8 image(s) → outputs/eeg2image
```

### 4a. Text only (no image generation)

```bash
python generate.py \
    --foundation_dir pth/CSBrain.pth \
    --projection_dir pth_downtasks/eeg_llm_bcic_new/projection_epoch6.pth \
    --lora_dir pth_downtasks/eeg_llm_bcic_new/lora_epoch6 \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples 5
```

---

## Project Structure

```
EEG2Image/
├── models/
│   ├── CSBrain.py                  # Pretrained EEG foundation encoder
│   ├── CSBrain_transformer.py      # Transformer building blocks
│   ├── CSBrain_transformerlayer.py # Custom transformer layer
│   ├── eeg_llm.py                  # EEGLanguageModel, EEGTokenReducer, EEGProjection
│   └── image_generator.py          # EEGImageGenerator (Stable Diffusion 2.1)
├── datasets/
│   ├── bciciv2a_llm_dataset.py     # BCIC-IV-2a motor imagery with text labels + collator
│   ├── bciciv2a_dataset.py         # BCIC-IV-2a classification loader (reference)
│   └── faced_llm_dataset.py        # FACED emotion dataset with text labels
├── utils/
│   ├── signaltools.py              # FFT-based EEG resampling (PyTorch)
│   └── util.py                     # General utilities
├── data/
│   └── BCICIV2a/
│       ├── raw/                    # .mat files (A01T.mat … A09E.mat)
│       └── processed_lmdb/        # LMDB database (train/val/test splits)
├── outputs/
│   └── eeg2image/                  # Generated PNG images + prompts.txt log
├── pth/
│   └── CSBrain.pth                 # Pretrained CSBrain encoder weights
├── pth_downtasks/
│   └── eeg_llm_bcic_new/
│       ├── projection_epoch6.pth   # Trained EEGProjection weights
│       └── lora_epoch6/            # Trained LoRA adapter (HuggingFace PEFT format)
├── sh/
│   ├── prepare_data.sh             # Data download + preprocessing
│   ├── finetune_eeg_llm_bcic.sh   # Train Stage 1 on BCIC-IV-2a
│   ├── finetune_eeg_llm_faced.sh  # Train Stage 1 on FACED
│   └── generate_images.sh          # Run full EEG → Text → Image pipeline
├── prepare_data.py                 # Data download + LMDB preprocessing
├── finetune_eeg_llm.py             # Stage 1 training entry point
├── finetune_eeg_llm_trainer.py     # EEGLLMTrainer (2-phase training)
├── generate.py                     # Full inference pipeline (Stage 1 + Stage 2)
├── EEG_LLM_Architecture.md        # Detailed architecture notes
└── requirements.txt
```

---

## Training Strategy (Stage 1)

| Phase | Epochs | Trainable | LR |
|-------|--------|-----------|----|
| Warmup | 1–5 | EEGProjection only | 5e-4 |
| Joint | 6–20 | EEGProjection + LoRA | 2e-4 |

- Optimiser: AdamW (`weight_decay=0.01`)
- LR schedule: Cosine annealing (`eta_min=1e-6`)
- Effective batch size: 4 × 8 grad accum = 32
- Mixed precision: float16 autocast + GradScaler
- Gradient clipping: `max_norm=1.0`

---

## Results (BCIC-IV-2a, 4-class Motor Imagery)

| Metric | Value |
|--------|-------|
| Test accuracy (keyword matching) | 31.34% |
| Chance level | 25.00% |
| Best val epoch | Epoch 6 (36.81%) |

---

## Datasets

### BCIC-IV-2a (Motor Imagery)

- 9 subjects, 4 classes: left hand / right hand / feet / tongue
- 22 EEG channels, 250 Hz → preprocessed to 200 Hz
- Window: 2–6 s post-cue → 800 samples → reshaped to `(22, 4, 200)`
- Splits: train (A01–A05), val (A06–A07), test (A08–A09)

### FACED (Emotion Recognition, optional)

- 9 emotion classes: Amusement, Inspiration, Joy, Tenderness, Anger, Disgust, Fear,
  Sadness, Neutral
- 30 EEG channels, 250 Hz
- Requires separate FACED dataset download and LMDB preparation

---

## Key Dependencies

| Package | Purpose | Licence |
|---------|---------|---------|
| `torch>=2.0` | Core deep learning | BSD |
| `transformers>=4.36` | TinyLlama model + tokeniser | Apache 2.0 |
| `peft>=0.7` | LoRA fine-tuning | Apache 2.0 |
| `bitsandbytes>=0.41` | 4-bit NF4 quantisation | MIT |
| `diffusers>=0.24` | Stable Diffusion pipeline | Apache 2.0 |
| `safetensors>=0.4` | Safe model weight loading | Apache 2.0 |
| `Pillow>=9.0` | Image saving | HPND (open) |
| `lmdb` | Fast dataset storage | OpenLDAP |
| `einops` | Tensor reshaping | MIT |
| `scipy` | Signal filtering & resampling | BSD |

---

## Citation

If you use the CSBrain encoder, please cite:

```
CSBrain: Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding
NeurIPS 2025 Spotlight
```

If you use Stable Diffusion 2.1, please credit:

```
Robin Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models",
CVPR 2022. Model: stabilityai/stable-diffusion-2-1 (Apache 2.0).
```
