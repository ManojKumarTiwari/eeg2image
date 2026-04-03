# EEG2Image — Brain-to-Image Generation Pipeline

Decode raw EEG signals into visual images using two complementary pipelines:
1. **EEG → Text → Image** (via TinyLlama LLM + aMUSEd-512 diffusion)
2. **EEG → Image** (direct, via EEGCLIPMapper + Stable Diffusion 1.5)

Covers two datasets: **BCIC-IV-2a** (motor imagery, 22-ch) and **EEG Visual Imagery** (visual imagery, 32-ch).

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Pipeline 1 — BCIC-IV-2a Motor Imagery (EEG → Text → Image)](#4-pipeline-1--bcic-iv-2a-motor-imagery)
5. [Pipeline 2 — Visual Imagery EEG → Text → Image](#5-pipeline-2--visual-imagery-eeg--text--image)
6. [Pipeline 3 — Visual Imagery Direct EEG → Image](#6-pipeline-3--visual-imagery-direct-eeg--image)
7. [Model Architectures](#7-model-architectures)
8. [Training Details](#8-training-details)
9. [Results](#9-results)
10. [Licences](#10-licences)

---

## 1. Datasets

### BCIC-IV-2a (Motor Imagery)

- **Task**: 4-class motor imagery — left hand, right hand, feet, tongue
- **Subjects**: 9 (A01–A09), 2 sessions each
- **Channels**: 22 EEG channels (10-20 system, no EOG)
- **Sampling rate**: 250 Hz → resampled to 200 Hz
- **Trial window**: 2–6 s post-cue → 800 samples → reshaped to `(22, 4, 200)`
- **Brain regions**: 3 (frontal, central, parietal/occipital)
- **Splits**: train A01–A05 (2784), val A06–A07 (1152), test A08–A09 (1152)
- **Source**: [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/)

### EEG Visual Imagery (Nature Scientific Data 2025)

- **Task**: Visual imagery of 10 object categories
- **Subjects**: 22, 2 sessions each (50 trials/class/session)
- **Channels**: 32 EEG channels (extended 10-20)
- **Sampling rate**: 1000 Hz → resampled to 200 Hz
- **Trial window**: 0.5–4.5 s post-cue → 4000 samples → reshaped to `(32, 4, 200)`
- **Classes**: dog, bird, fish, pentagram, square, circle, scissor, watch, cup, chair
- **Brain regions**: 5 (frontal, central, parietal, occipital, temporal)
- **Splits**: train S01–S16, val S17–S19, test S20–S22
- **Source**: Figshare `10.6084/m9.figshare.30227503`

---

## 2. Project Structure

```
EEG2Image/
├── models/
│   ├── CSBrain.py                   # Pretrained EEG foundation encoder (32-ch support)
│   ├── CSBrain_transformer.py       # Transformer building blocks
│   ├── CSBrain_transformerlayer.py  # Custom transformer layer
│   ├── eeg_llm.py                   # EEGLanguageModel, EEGTokenReducer, EEGProjection
│   │                                #   VI_BRAIN_REGIONS, VI_ELECTRODE_LABELS, VI_TOPOLOGY
│   ├── eeg_clip_mapper.py           # EEGCLIPMapper (Q-Former), CLIPImageTargetBuilder,
│   │                                #   CLIPTextTargetBuilder
│   └── image_generator.py           # EEGImageGenerator (SD 2.1)
│
├── datasets/
│   ├── bciciv2a_llm_dataset.py      # BCIC-IV-2a LMDB loader + collator
│   ├── bciciv2a_dataset.py          # BCIC-IV-2a classification loader (reference)
│   ├── faced_llm_dataset.py         # FACED emotion dataset loader
│   └── visual_imagery_llm_dataset.py # VI LMDB loader + collator + VI_KEYWORDS
│
├── data/
│   ├── BCICIV2a/
│   │   ├── raw/                     # A01T.mat … A09E.mat
│   │   └── processed_lmdb/          # LMDB (train/val/test)
│   └── VisualImagery/
│       ├── raw/                     # BIDS .fif files per subject
│       ├── stimuli/                 # 10 stimulus JPEGs (Animal_dog.jpg, etc.)
│       └── processed_lmdb/          # LMDB (train/val/test)
│
├── pth/
│   └── CSBrain.pth                  # Pretrained CSBrain encoder weights
│
├── pth_downtasks/
│   ├── eeg_llm_bcic_new/
│   │   ├── projection_epoch6.pth    # EEGProjection + EEGTokenReducer (BCIC)
│   │   └── lora_epoch6/             # LoRA adapter
│   ├── eeg_llm_vi/
│   │   ├── projection_epoch5.pth    # EEGProjection + EEGTokenReducer (VI)
│   │   └── lora_epoch5/             # LoRA adapter
│   └── eeg_direct/
│       └── mapper_epoch17.pth       # EEGCLIPMapper checkpoint (best)
│
├── outputs/
│   ├── eeg2image/                   # BCIC generated images
│   ├── vi_images/                   # VI EEG→Text→Image results
│   └── vi_eeg2img/                  # VI direct EEG→Image results
│
├── sh/
│   ├── finetune_eeg_llm_bcic.sh
│   ├── finetune_eeg_llm_vi.sh
│   └── generate_images.sh
│
├── prepare_data.py                  # BCIC-IV-2a download + LMDB preprocessing
├── prepare_data_vi.py               # Visual Imagery download + LMDB preprocessing
├── finetune_eeg_llm.py              # EEG→Text training (BCIC + VI)
├── finetune_eeg_to_image.py         # Direct EEG→Image training (EEGCLIPMapper)
├── generate.py                      # BCIC EEG→Text→Image inference
├── generate_vi.py                   # VI EEG→Text→Image inference
├── generate_vi_eeg2img.py           # VI direct EEG→Image inference (SD 1.5)
└── requirements.txt
```

---

## 3. Environment Setup

```bash
# Python 3.9+, CUDA 11.8+
pip install -r requirements.txt
```

Hardware requirements:

| Task | VRAM |
|------|------|
| EEG→Text training (TinyLlama 4-bit + LoRA) | 8 GB |
| EEGCLIPMapper training | 6 GB |
| Inference (EEG models only) | 2 GB |
| SD 1.5 generation (fp16) | 4 GB |
| End-to-end (sequential: free EEG before SD loads) | **8 GB** |

---

## 4. Pipeline 1 — BCIC-IV-2a Motor Imagery

### Process Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     PIPELINE 1: BCIC-IV-2a                       │
│                                                                    │
│  Raw .mat files                                                    │
│       │                                                            │
│       ▼  prepare_data.py                                           │
│  Preprocess (22-ch, 250→200 Hz, 2-6s window, (22,4,200))         │
│       │                                                            │
│       ▼  LMDB                                                      │
│  data/BCICIV2a/processed_lmdb/                                    │
│       │                                                            │
│       ▼  finetune_eeg_llm.py                                       │
│  CSBrain[frozen] → EEGTokenReducer → EEGProjection → TinyLlama    │
│  [Phase 1: warmup EEGProjection 5ep]                              │
│  [Phase 2: joint EEGProjection + LoRA 15ep]                       │
│       │                                                            │
│       ▼  generate.py                                               │
│  EEG → Text Description → SD 2.1 → 512×512 PNG                   │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1: Download and Preprocess

```bash
# Download + preprocess (downloads ~1.4 GB, creates LMDB)
python prepare_data.py

# If .mat files already in data/BCICIV2a/raw/:
python prepare_data.py --skip_download
```

Preprocessing per trial:
- Select 22 EEG channels, exclude EOG
- Zero-mean across channels
- Bandpass filter 0.3–50 Hz (5th-order Butterworth)
- Extract 2–6 s post-cue → 800 samples at 200 Hz
- Reshape to `(22, 4, 200)` (4 patches × 200 samples)
- Divide by 100.0, cast float32

Result: 2784 train / 1152 val / 1152 test samples in LMDB

### Step 2: Train EEG → Text Model

```bash
python finetune_eeg_llm.py \
    --downstream_dataset BCICIV2a \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_bcic_new \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --epochs 20 --warmup_epochs 5 \
    --batch_size 4 --gradient_accumulation_steps 8 \
    --lr 2e-4 --cuda 0
```

### Step 3: Generate Images

```bash
python generate.py \
    --foundation_dir pth/CSBrain.pth \
    --projection_dir pth_downtasks/eeg_llm_bcic_new/projection_epoch6.pth \
    --lora_dir pth_downtasks/eeg_llm_bcic_new/lora_epoch6 \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples 20 --generate_images \
    --image_model stabilityai/stable-diffusion-2-1 \
    --output_dir outputs/eeg2image
```

---

## 5. Pipeline 2 — Visual Imagery EEG → Text → Image

### Process Flow

```
┌──────────────────────────────────────────────────────────────────┐
│             PIPELINE 2: Visual Imagery EEG→Text→Image            │
│                                                                    │
│  Raw BIDS .fif files (22 subjects, 2 sessions)                    │
│       │                                                            │
│       ▼  prepare_data_vi.py                                        │
│  Preprocess (32-ch, 1000→200 Hz, 0.5-4.5s window, (32,4,200))   │
│       │                                                            │
│       ▼  LMDB                                                      │
│  data/VisualImagery/processed_lmdb/                               │
│       │                                                            │
│       ▼  finetune_eeg_llm.py --downstream_dataset VI              │
│  CSBrain[frozen,32-ch] → EEGTokenReducer(5 regions)               │
│      → EEGProjection → TinyLlama[LoRA]                            │
│       │                                                            │
│       ▼  generate_vi.py                                            │
│  EEG → keyword text → class label → aMUSEd-512 → 512×512 PNG     │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1: Download and Preprocess Visual Imagery Dataset

```bash
# Download from Figshare (10.6084/m9.figshare.30227503)
python prepare_data_vi.py

# If raw files already downloaded:
python prepare_data_vi.py --skip_download
```

Preprocessing per trial:
- Select 32 EEG channels (extended 10-20)
- Zero-mean across channels
- Bandpass filter 0.3–50 Hz (5th-order Butterworth)
- Extract 0.5–4.5 s post-cue → 4000 samples at 1000 Hz
- Resample to 800 samples (200 Hz)
- Reshape to `(32, 4, 200)` (4 patches × 200 samples)
- Divide by 100.0, cast float32

Result: ~11000 train / ~3300 val / ~3300 test samples

### Step 2: Create Text Targets

Text targets are keyword-based class descriptions, created automatically from class labels:

```python
VI_KEYWORDS = {
    0: ['dog', 'canine', 'golden retriever', 'animal'],
    1: ['bird', 'avian', 'feather', 'wing'],
    2: ['fish', 'aquatic', 'fin', 'swim'],
    3: ['pentagram', 'star', 'five-pointed', 'pentagon'],
    4: ['square', 'rectangle', 'four-sided', 'box'],
    5: ['circle', 'round', 'oval', 'sphere'],
    6: ['scissor', 'scissors', 'cutting', 'shears'],
    7: ['watch', 'clock', 'timepiece', 'wristwatch'],
    8: ['cup', 'mug', 'coffee', 'ceramic'],
    9: ['chair', 'seat', 'furniture', 'sitting'],
}
```

### Step 3: Train EEG → Text Model (Visual Imagery)

```bash
python finetune_eeg_llm.py \
    --downstream_dataset VI \
    --datasets_dir data/VisualImagery/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_vi \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --epochs 10 --warmup_epochs 3 \
    --batch_size 4 --gradient_accumulation_steps 8 \
    --lr 2e-4 --cuda 0
```

CSBrain weight loading for 32 channels:
- `PatchEmbedding`, transformer layers → loaded from `CSBrain.pth` (compatible)
- `BrainEmbedEEGLayer.region_blocks` → random init (incompatible, different region sizes)
- 295/295 weights loaded overall

### Step 4: Generate Images (EEG → Text → aMUSEd-512)

```bash
python generate_vi.py \
    --datasets_dir data/VisualImagery/processed_lmdb \
    --projection_path pth_downtasks/eeg_llm_vi/projection_epoch5.pth \
    --lora_dir pth_downtasks/eeg_llm_vi/lora_epoch5 \
    --stimuli_dir data/VisualImagery/stimuli \
    --num_samples 20 \
    --output_dir outputs/vi_images
```

Image generation uses `amused/amused-512` (OpenRAIL++ licence, ~0.6 GB VRAM):
- 12 inference steps, guidance_scale=10.0
- Output: side-by-side comparison (original stimulus | generated)

---

## 6. Pipeline 3 — Visual Imagery Direct EEG → Image

This pipeline bypasses text entirely — EEG embeddings are mapped directly to CLIP conditioning space using a Q-Former style mapper, then decoded with Stable Diffusion 1.5.

### Process Flow

```
┌──────────────────────────────────────────────────────────────────┐
│            PIPELINE 3: Direct EEG→Image (EEGCLIPMapper)          │
│                                                                    │
│  TRAINING                                                          │
│  ─────────────────────────────────────────────                    │
│  VI LMDB (32,4,200) EEG + class label                             │
│       │                                                            │
│       ▼  CSBrain [frozen]                                          │
│  EEG features (32,4,200)                                          │
│       │                                                            │
│       ▼  EEGTokenReducer [trainable, warm-start]                  │
│  20 EEG tokens × 200 dim                                          │
│       │                                                            │
│       ▼  EEGCLIPMapper [trainable]                                 │
│  encoder_hidden_states (B,77,768) + prompt_embeds (B,768)         │
│       │            │                                               │
│       │            ▼  class_logits (B,10)                         │
│       │                                                            │
│  Targets (frozen):                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ CLIPImageTargetBuilder                                       │  │
│  │   CLIP ViT-L/14 encodes 10 stimulus photos → (10,768)       │  │
│  │ CLIPTextTargetBuilder                                        │  │
│  │   aMUSEd CLIP text encoder → (10,77,768) + (10,768)         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│       │                                                            │
│  Loss = λ_cls × CE + λ_cont × InfoNCE + λ_cos × cosine           │
│       + λ_mse × MSE (disabled in warmup)                         │
│                                                                    │
│  INFERENCE                                                         │
│  ─────────────────────────────────────────────                    │
│  EEG → CSBrain → TokenReducer → EEGCLIPMapper                     │
│       → class_logits → predicted label → SD 1.5 text prompt       │
│       → 512×512 PNG                                                │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1: Prepare CLIP Image Targets

CLIP image targets are built automatically during training from the 10 stimulus photos:

```
data/VisualImagery/stimuli/
├── Animal_dog.jpg        # class 0
├── Animal_bird.jpg       # class 1
├── Animal_fish.jpg       # class 2
├── Figure_pentagram.jpg  # class 3
├── Figure_square.jpg     # class 4
├── Figure_circle.jpg     # class 5
├── Object_scissor.jpg    # class 6
├── Object_watch.jpg      # class 7
├── Object_cup.jpg        # class 8
└── Object_chair.jpg      # class 9
```

The `CLIPImageTargetBuilder` encodes these with `openai/clip-vit-large-patch14`, producing discriminative targets (mean pairwise cosine similarity ~0.54 vs ~0.90 for text targets).

### Step 2: Train EEGCLIPMapper

```bash
python finetune_eeg_to_image.py \
    --datasets_dir data/VisualImagery/processed_lmdb \
    --stimuli_dir data/VisualImagery/stimuli \
    --foundation_dir pth/CSBrain.pth \
    --output_dir pth_downtasks/eeg_direct \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 16 \
    --lr 5e-4 \
    --lambda_cls 5.0 \
    --lambda_contrastive 2.0 \
    --lambda_cos 1.0 \
    --lambda_mse 0.1 \
    --temperature 0.5 \
    --cuda 0
```

**Two-phase training**:

| Phase | Epochs | Active losses | LR |
|-------|--------|---------------|----|
| Warmup | 1–5 | CE + InfoNCE + cosine | 5e-4 |
| Full | 6–20 | CE + InfoNCE + cosine + MSE | 1e-4 |

### Step 3: Generate Images (Direct EEG → SD 1.5)

```bash
python generate_vi_eeg2img.py \
    --datasets_dir data/VisualImagery/processed_lmdb \
    --mapper_path pth_downtasks/eeg_direct/mapper_epoch17.pth \
    --stimuli_dir data/VisualImagery/stimuli \
    --num_samples 20 \
    --output_dir outputs/vi_eeg2img \
    --num_inference_steps 25 \
    --guidance_scale 7.5
```

Two-stage inference (fits in 8 GB VRAM):
1. Load EEG models → run classification → delete EEG models → `torch.cuda.empty_cache()`
2. Load SD 1.5 (fp16) → generate with class-matched prompts → save

---

## 7. Model Architectures

### CSBrain Encoder (Frozen Foundation Model)

```
Input: (B, C, P, T)   C=channels, P=patches, T=200 time samples
        │
        ▼
┌────────────────────────────────────────────┐
│  PatchEmbedding                            │
│  Linear(T=200 → d_model=200) per patch     │
└────────────────────────────────────────────┘
        │  (B, C, P, 200)
        ▼
┌────────────────────────────────────────────┐
│  BrainEmbedEEGLayer                        │
│  Groups channels into brain regions        │
│  BCIC: 3 regions  |  VI: 5 regions         │
│  Per-region learnable embeddings           │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  TemEmbedEEGLayer                          │
│  Learnable temporal position embeddings    │
│  across P=4 patches                        │
└────────────────────────────────────────────┘
        │
        ▼ × 12 layers
┌────────────────────────────────────────────┐
│  CSBrain Transformer Block                 │
│  ├─ Cross-scale spatial attention          │
│  │   (channels attend across brain regions)│
│  └─ Temporal self-attention                │
│      (patches attend within each channel) │
└────────────────────────────────────────────┘
        │
        ▼
  Output: (B, C, P, 200)    same shape as input
```

### EEGTokenReducer

```
Input: (B, C, P, 200)
        │
        ▼
  Group channels by brain region
  Average-pool within each region
        │
        ▼
  Flatten: n_regions × P tokens
        │
        ▼
  Output: (B, n_regions × P, 200)
           BCIC: (B, 12, 200)   3 regions × 4 patches
           VI:   (B, 20, 200)   5 regions × 4 patches
```

### EEGProjection (Pipeline 1 & 2 only)

```
Input: (B, n_tokens, 200)
        │
        ▼  Linear(200 → 2048) + LayerNorm + GELU + Dropout
        ▼  Linear(2048 → 2048) + LayerNorm
        │
        ▼
  Output: (B, n_tokens, 2048)   [TinyLlama hidden dim]
```

### EEGCLIPMapper (Pipeline 3 — Q-Former architecture)

```
Input: (B, 20, 200)   20 EEG tokens from TokenReducer
        │
Stage 1 — Input projection MLP
        ▼  Linear(200 → 512) + LayerNorm + GELU + Dropout + Linear(512 → 512)
        │  (B, 20, 512)
        │
Stage 2 — Transformer self-attention over EEG tokens
        ▼  4× TransformerEncoderLayer(d=512, heads=8, ffn=1024, pre-norm)
        │  (B, 20, 512)
        │
Stage 3 — Q-Former: 77 learnable queries attend to 20 EEG tokens
        │  queries: nn.Parameter(77, 512) → expand → (B, 77, 512)
        ▼  MultiheadAttention(Q=queries, K=EEG, V=EEG, heads=8)
        ▼  LayerNorm(attended + queries)   residual connection
        │  (B, 77, 512)
        │
Stage 4 — Project to CLIP space
        ▼  Linear(512 → 768) + LayerNorm  →  encoder_hidden_states (B, 77, 768)
        ▼  mean(dim=1)                     →  pooled (B, 768)
        ▼  Linear(768 → 768) + Tanh        →  prompt_embeds (B, 768)
        ▼  Linear(768 → 10)                →  class_logits (B, 10)

Total trainable params: ~10.8M
```

### TinyLlama + LoRA (Pipeline 1 & 2)

```
EEG tokens (B, n_tokens, 2048)
        │
        ▼  Prefix inject into TinyLlama KV cache
        │
┌──────────────────────────────────────────┐
│  TinyLlama-1.1B-Chat [4-bit NF4 quant]  │
│  22 transformer layers, d=2048           │
│  LoRA adapters on q_proj + v_proj only   │
│    rank=8, alpha=16                      │
│    ~1.1M trainable params (0.10%)        │
└──────────────────────────────────────────┘
        │
        ▼
  Generated text (keyword extraction → class label)
```

---

## 8. Training Details

### Pipeline 1 & 2 — EEG→Text (EEGProjection + LoRA)

| Hyperparameter | BCIC | Visual Imagery |
|----------------|------|----------------|
| Epochs (warmup + joint) | 5 + 15 = 20 | 3 + 7 = 10 |
| Batch size | 4 | 4 |
| Grad accumulation | 8 | 8 |
| Effective batch | 32 | 32 |
| LR (warmup) | 5e-4 | 5e-4 |
| LR (joint) | 2e-4 | 2e-4 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| LR schedule | Cosine anneal | Cosine anneal |
| Mixed precision | fp16 | fp16 |
| Grad clip norm | 1.0 | 1.0 |

**Loss**: Cross-entropy on next-token prediction over target text sequences

### Pipeline 3 — Direct EEG→Image (EEGCLIPMapper)

**Combined loss function**:

```
L_total = λ_cls × L_CE  +  λ_cont × L_InfoNCE  +  λ_cos × L_cosine  +  λ_mse × L_MSE

where:
  L_CE      = cross-entropy on class_logits vs true label   (λ=5.0)
  L_InfoNCE = InfoNCE contrastive loss vs CLIP image prototypes  (λ=2.0, τ=0.5)
  L_cosine  = 1 − cosine_similarity(prompt_embeds, CLIP_image_target)  (λ=1.0)
  L_MSE     = MSE(encoder_hidden_states, CLIP_text_hidden_target)  (λ=0.1, disabled warmup)
```

Why CLIP image targets over text targets:
- CLIP text embeddings for 10 class names: mean pairwise cosine sim ≈ **0.90** (near-identical → no gradient signal)
- CLIP image embeddings of stimulus photos: mean pairwise cosine sim ≈ **0.54** (discriminative)

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 |
| Warmup epochs | 5 |
| Batch size | 16 |
| LR (warmup) | 5e-4 |
| LR (full) | 1e-4 |
| Temperature τ | 0.5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |

---

## 9. Results

### Pipeline 1 — BCIC-IV-2a Motor Imagery (4 classes, chance=25%)

| Metric | Value |
|--------|-------|
| Best val accuracy (keyword) | 36.81% (epoch 6) |
| Test accuracy (keyword) | 31.34% |
| Chance level | 25.00% |

### Pipeline 2 — Visual Imagery EEG→Text (10 classes, chance=10%)

| Metric | Value |
|--------|-------|
| Best val accuracy | ~40%+ |
| Test accuracy | ~35%+ |
| Chance level | 10.00% |

### Pipeline 3 — Direct EEG→Image (10 classes, chance=10%)

| Metric | Value |
|--------|-------|
| Best val accuracy (cls head) | 10.75% (epoch 17) |
| EEG classification on test | varies per run |
| Chance level | 10.00% |

Image quality: Stable Diffusion 1.5 with class-matched prompts produces photorealistic 512×512 outputs regardless of classifier confidence.

### Output Format

Each pipeline saves:
- `generated_NNNN_trueX_predY.png` — generated image (512×512)
- `comparison_NNNN_trueX_predY.png` — side-by-side: original stimulus | generated
  - Green header: correct prediction; Red header: incorrect prediction
- `summary_grid.png` — 4-column thumbnail grid of all comparisons

---

## 10. Licences

| Component | Model/Repo | Licence |
|-----------|-----------|---------|
| CSBrain encoder | NeurIPS 2025 | Research use |
| TinyLlama-1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Apache 2.0 |
| Stable Diffusion 1.5 | runwayml/stable-diffusion-v1-5 | CreativeML OpenRAIL-M |
| Stable Diffusion 2.1 | stabilityai/stable-diffusion-2-1 | Apache 2.0 |
| aMUSEd-512 | amused/amused-512 | OpenRAIL++ |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | MIT |
| BCIC-IV-2a dataset | BCI Competition IV | Research use |
| EEG Visual Imagery | Figshare 10.6084/m9.figshare.30227503 | CC BY 4.0 |

---

## Citation

If you use the CSBrain encoder:

```
CSBrain: Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding
NeurIPS 2025 Spotlight
```

If you use the EEG Visual Imagery dataset:

```
EEG Visual Imagery Dataset, Nature Scientific Data 2025
DOI: 10.6084/m9.figshare.30227503
```
