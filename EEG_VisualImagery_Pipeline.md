# EEG Visual Imagery → CSBrain Embeddings → Text Generation Pipeline

## Overview

This document describes the end-to-end pipeline for decoding **visual imagery** from EEG signals using the CSBrain encoder and TinyLlama language model. The pipeline is adapted from the existing BCICIV2A motor imagery pipeline to support a 10-class visual imagery dataset.

---

## 1. Dataset

### Source
- **Name**: EEG Dataset for Visual Imagery-Based Brain–Computer Interface
- **Published**: Nature Scientific Data, 2025
- **DOI**: [10.6084/m9.figshare.30227503](https://figshare.com/articles/dataset/EEG_Dataset_for_Visual_Imagery/30227503)
- **Downloaded as**: `30227503.zip`

### Structure inside the zip
```
30227503.zip
├── sub-01.zip ... sub-22.zip      ← 22 subjects, each a nested zip
├── stimuli/                        ← 10 stimulus images (.jpg)
│   ├── Animal_bird.jpg
│   ├── Animal_dog.jpg
│   ├── Animal_fish.jpg
│   ├── Figure_circle.jpg
│   ├── Figure_pentagram.jpg
│   ├── Figure_square.jpg
│   ├── Object_chair.jpg
│   ├── Object_cup.jpg
│   ├── Object_scissor.jpg
│   └── Object_watch.jpg
├── electrodes.tsv                  ← 32-channel 3D positions
├── task-AVI_events.json            ← Animal Visual Imagery event schema
├── task-FVI_events.json            ← Figure Visual Imagery event schema
├── task-OVI_events.json            ← Object Visual Imagery event schema
└── code/Preprocess.py              ← Official preprocessing reference
```

Each subject zip contains:
```
sub-XX/
├── ses-01/eeg/
│   ├── sub-XX_ses-01_task-AVI_eeg.bdf       ← Raw EEG (Animal imagery)
│   ├── sub-XX_ses-01_task-AVI_events.tsv     ← Trial onset/labels
│   ├── sub-XX_ses-01_task-FVI_eeg.bdf        ← Raw EEG (Figure imagery)
│   ├── sub-XX_ses-01_task-FVI_events.tsv
│   ├── sub-XX_ses-01_task-OVI_eeg.bdf        ← Raw EEG (Object imagery)
│   └── sub-XX_ses-01_task-OVI_events.tsv
└── ses-02/eeg/                                ← Second session (same structure)
```

### EEG Recording Specs
| Property | Value |
|---|---|
| Subjects | 22 |
| Sessions per subject | 2 |
| Tasks per session | 3 (AVI, FVI, OVI) |
| Channels | 32 (extended 10-20 system) |
| Sampling rate | 1000 Hz |
| File format | BDF (BioSemi Data Format) |
| Reference | CPz (hardware), re-referenced in preprocessing |
| Ground | AFz |

### 10 Visual Imagery Classes

| Global Label | Class | Task | Event Value |
|---|---|---|---|
| 0 | dog | AVI | 1 |
| 1 | bird | AVI | 2 |
| 2 | fish | AVI | 3 |
| 3 | pentagram | FVI | 1 |
| 4 | square | FVI | 2 |
| 5 | circle | FVI | 3 |
| 6 | scissor | OVI | 1 |
| 7 | watch | OVI | 2 |
| 8 | cup | OVI | 3 |
| 9 | chair | OVI | 4 |

### Trial Structure (per task)
```
|--- 3s fixation cross ---|--- 4s image shown ---|--- imagery window (4s) ---|--- 4s rest ---|
                                                   ↑ event trigger here
                                                   ← we extract this 4s window →
```
- **Trials per session**: AVI=120, FVI=120, OVI=160 → **400 trials/session**
- **Trials per subject**: 400 × 2 sessions = **800 trials**

---

## 2. Data Preparation (`prepare_data_vi.py`)

### Step 1: Extraction
The outer zip contains per-subject nested zips. The script extracts all of them:
```
30227503.zip → data/VisualImagery/raw/sub-01/ ... sub-22/
```

```bash
python prepare_data_vi.py --zip_path /path/to/30227503.zip
# If already extracted:
python prepare_data_vi.py --skip_extract
```

### Step 2: Preprocessing Pipeline (per BDF file)

```
Raw BDF (1000 Hz, 33 channels including Status)
    │
    ├─ Pick 32 EEG channels (drop Status channel)
    │   Order: Fpz, Fp1, Fp2, Fz, F3, F4, F7, F8,
    │          FCz, FC3, FC4, FT7, FT8,
    │          Cz, C3, C4, T7, T8,
    │          CP3, CP4, TP7, TP8, Pz, P3, P4, P7, P8,
    │          PO3, PO4, Oz, O1, O2
    │
    ├─ Convert volts → microvolts (× 1e6)
    │
    ├─ Zero-mean normalisation (subtract channel mean)
    │
    ├─ Bandpass filter: 0.3 – 50 Hz
    │   (5th-order Butterworth, zero-phase sosfiltfilt)
    │
    ├─ Read events.tsv → extract (latency_samples, global_label)
    │   Filter out spurious trigger codes (e.g. sub-13 ses-02 had value=5)
    │
    ├─ Extract 4-second window from each trial onset
    │   (4000 samples at 1000 Hz)
    │
    ├─ Resample: 1000 Hz → 200 Hz
    │   (800 samples via scipy.signal.resample)
    │
    ├─ Reshape: (32, 800) → (32, 4, 200)
    │   (32 channels × 4 temporal patches × 200 samples/patch)
    │
    └─ Normalise: divide by 100.0 → float32
```

**Output per trial**: `np.ndarray(32, 4, 200), dtype=float32`

### Step 3: Subject Splits

| Split | Subjects | Trials |
|---|---|---|
| Train | sub-01 → sub-16 (16 subjects) | 11,840 |
| Val | sub-17 → sub-19 (3 subjects) | 2,400 |
| Test | sub-20 → sub-22 (3 subjects) | 1,600 |

### Step 4: LMDB Storage

Stored in `data/VisualImagery/processed_lmdb/` using the same format as BCICIV2A for compatibility:

```python
# LMDB key-value structure:
'__keys__'     → pickle({'train': [...], 'val': [...], 'test': [...]})
'train_000000' → pickle({'sample': np.ndarray(32, 4, 200), 'label': int})
'train_000001' → pickle({'sample': np.ndarray(32, 4, 200), 'label': int})
...
```

---

## 3. Brain Region Configuration (`models/eeg_llm.py`)

CSBrain is a **channel-agnostic** framework — it accepts arbitrary EEG layouts via `brain_regions` and `sorted_indices` parameters. For the 32-channel VI dataset, 5 anatomical regions are defined:

```python
VI_BRAIN_REGIONS = [
    0, 0, 0, 0, 0, 0, 0, 0,      # Frontal      : Fpz, Fp1, Fp2, Fz, F3, F4, F7, F8
    1, 1, 1, 1, 1,                # Fronto-central: FCz, FC3, FC4, FT7, FT8
    2, 2, 2, 2, 2,                # Central       : Cz, C3, C4, T7, T8
    3, 3, 3, 3, 3, 3, 3, 3, 3,   # Parietal      : CP3, CP4, TP7, TP8, Pz, P3, P4, P7, P8
    4, 4, 4, 4, 4,                # Occipital     : PO3, PO4, Oz, O1, O2
]
```

These regions are motivated by visual imagery neuroscience:
- **Frontal** — top-down attention and imagery generation
- **Fronto-central** — working memory during imagery
- **Central** — sensorimotor (minimal involvement in VI)
- **Parietal** — spatial representation and object reconstruction
- **Occipital** — primary visual cortex activation during mental imagery

---

## 4. CSBrain Encoder

**File**: `models/CSBrain.py`

### Role
Feature extractor that transforms raw EEG patches into rich spatiotemporal representations. **Frozen during fine-tuning** — pretrained on a large EEG dataset.

### Input / Output
```
Input:  (batch, 32, 4, 200)   ← 32 channels, 4 patches, 200 samples
Output: (batch, 32, 4, 200)   ← same shape, enriched features
```

### Architecture
```
Input EEG (batch, 32, 4, 200)
    │
    ├─ Channel reordering (sort by brain region)
    │
    ├─ PatchEmbedding
    │   ├─ Conv2D temporal projection (kernel 1×49, stride 1×25)
    │   ├─ Spectral embedding (FFT-based)
    │   └─ Positional embedding (2D Conv)
    │
    ├─ × 12 Transformer layers
    │   ├─ TemEmbedEEGLayer  (multi-scale temporal: kernels 1, 3, 5)
    │   ├─ BrainEmbedEEGLayer (region-aware spatial convolutions)
    │   └─ Inter-window + inter-region attention
    │
    └─ Output projection
```

### Pretrained Weights
- **File**: `pth/CSBrain.pth` (35.7 MB)
- **Loaded with** `strict=False` — PatchEmbedding and transformer weights transfer fully (295/295 keys matched)
- `BrainEmbedEEGLayer.region_blocks` reinitialises randomly (different region topology vs pretrained config) and learns during training

---

## 5. EEG Token Reduction

**Class**: `EEGTokenReducer` in `models/eeg_llm.py`

Reduces the 32-channel output to a compact token sequence by **averaging channels within each brain region**:

```
CSBrain output: (batch, 32, 4, 200)
    │
    ├─ Region 0 (8 frontal channels)      → mean → (batch, 4, 200)
    ├─ Region 1 (5 fronto-central ch.)    → mean → (batch, 4, 200)
    ├─ Region 2 (5 central channels)      → mean → (batch, 4, 200)
    ├─ Region 3 (9 parietal channels)     → mean → (batch, 4, 200)
    └─ Region 4 (5 occipital channels)    → mean → (batch, 4, 200)
    │
    Stack + reshape → (batch, 20, 200)
                       ↑
                       5 regions × 4 temporal patches = 20 EEG tokens
```

---

## 6. EEG Projection

**Class**: `EEGProjection` in `models/eeg_llm.py`

2-layer MLP that maps EEG tokens from CSBrain's 200-dim space into TinyLlama's 2048-dim embedding space:

```
(batch, 20, 200) → Linear(200→2048) → GELU → Dropout → Linear(2048→2048) → (batch, 20, 2048)
```

This is the **primary trainable component in Phase 1** (warmup).

---

## 7. Language Model: TinyLlama

**Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

| Property | Value |
|---|---|
| Parameters | 1.1 billion |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| VRAM usage | ~700 MB |
| Fine-tuning method | LoRA (r=8, α=16) |
| LoRA target modules | q_proj, v_proj |
| Trainable LoRA params | 1,126,400 |

---

## 8. EEG-Language Model (`models/eeg_llm.py`)

### Full Forward Pass
```
EEG Input (batch, 32, 4, 200)
    │
    ├─ CSBrain encoder (frozen)        → (batch, 32, 4, 200)
    ├─ EEGTokenReducer (5 regions)     → (batch, 20, 200)
    └─ EEGProjection MLP (trainable)   → (batch, 20, 2048)
                                               ↓
                                    Concatenate with prompt embeddings
                                               ↓
                            [prompt_embeds | eeg_embeds]  (batch, prompt_len+20, 2048)
                                               ↓
                                    TinyLlama (4-bit + LoRA)
                                               ↓
                                    Generated text description
```

### Chat Template
```
<|system|>
You are an expert EEG analyst specializing in visual imagery decoding from brain signals.
Analyze the provided EEG recording and describe the visual imagery being imagined.
</s>
<|user|>
[EEG_TOKENS]
Analyze this EEG recording and describe which visual image the subject is imagining,
including the observed neural patterns and activated brain regions.
</s>
<|assistant|>
[Generated description]
```

---

## 9. Training (`finetune_eeg_llm.py` + `finetune_eeg_llm_trainer.py`)

### Two-Phase Strategy

#### Phase 1 — Projection Warmup
| Property | Value |
|---|---|
| Epochs | 1–5 |
| Trainable | EEGProjection only |
| Learning rate | 1e-3 (5× base LR) |
| Frozen | CSBrain + LoRA adapters |
| Goal | Align EEG embedding space to TinyLlama input |

#### Phase 2 — Joint Training
| Property | Value |
|---|---|
| Epochs | 6–20 |
| Trainable | EEGProjection + LoRA adapters |
| Learning rate | 2e-4 |
| Frozen | CSBrain encoder |
| Goal | End-to-end EEG → text optimisation |

### Training Hyperparameters
| Parameter | Value |
|---|---|
| Batch size | 4 |
| Gradient accumulation | 8 steps (effective batch = 32) |
| Optimizer | AdamW |
| LR schedule | Cosine annealing (η_min = 1e-6) |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | float16 autocast + GradScaler |
| Max target length | 128 tokens |

### Run Command
```bash
.venv/Scripts/python.exe finetune_eeg_llm.py \
    --downstream_dataset VI \
    --datasets_dir data/VisualImagery/processed_lmdb \
    --num_of_classes 10 \
    --model_dir pth_downtasks/eeg_llm_vi \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --temporal_pool_stride 1 \
    --epochs 20 \
    --warmup_epochs 5
```

### Saved Checkpoints (`pth_downtasks/eeg_llm_vi/`)
```
pth_downtasks/eeg_llm_vi/
├── projection_epochN.pth      ← EEGProjection + EEGTokenReducer weights
└── lora_epochN/               ← HuggingFace PEFT LoRA adapter
    ├── adapter_config.json
    └── adapter_model.bin
```
Only the **best validation accuracy** checkpoint is kept.

---

## 10. Evaluation

### Metric: Keyword Extraction Accuracy
Generated text is matched against class-specific keywords:

| Label | Class | Keywords |
|---|---|---|
| 0 | dog | dog, canine, four-legged, mammal |
| 1 | bird | bird, avian, flight, feather |
| 2 | fish | fish, aquatic, underwater, fins |
| 3 | pentagram | pentagram, five-pointed, star |
| 4 | square | square, four-sided, rectangle |
| 5 | circle | circle, round, circular, curved |
| 6 | scissor | scissor, scissors, cutting, tool |
| 7 | watch | watch, wristwatch, timepiece, clock |
| 8 | cup | cup, mug, vessel, drinking |
| 9 | chair | chair, seat, furniture, sitting |

**Chance level**: 10% (random 10-class)

---

## 11. Hardware & Performance

| Property | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| VRAM used during training | ~2.1 GB (26%) |
| Training speed | ~2.9 it/s |
| Time per epoch (train only) | ~17 min |
| Time per epoch (train + val) | ~28 min |
| Total training time (20 epochs) | ~9.5 hours |

---

## 12. File Structure

```
EEG2Image/
├── prepare_data_vi.py                        ← Download + preprocess VI dataset
├── finetune_eeg_llm.py                       ← Training entry point
├── finetune_eeg_llm_trainer.py               ← Two-phase training loop
├── models/
│   ├── CSBrain.py                            ← EEG encoder (channel-agnostic)
│   ├── CSBrain_transformer.py                ← TemEmbed + BrainEmbed layers
│   ├── CSBrain_transformerlayer.py           ← Transformer block
│   └── eeg_llm.py                            ← EEGLanguageModel (full model)
│       ├── BCIC_BRAIN_REGIONS                ← 22-ch motor imagery config
│       ├── FACED_BRAIN_REGIONS               ← 30-ch emotion config
│       └── VI_BRAIN_REGIONS                  ← 32-ch visual imagery config ← NEW
├── datasets/
│   ├── bciciv2a_llm_dataset.py               ← Motor imagery LMDB loader
│   ├── faced_llm_dataset.py                  ← Emotion LMDB loader
│   └── visual_imagery_llm_dataset.py         ← Visual imagery LMDB loader ← NEW
├── data/
│   ├── BCICIV2a/processed_lmdb/              ← Motor imagery LMDB (19 GB)
│   └── VisualImagery/
│       ├── raw/sub-01/ ... sub-22/           ← Extracted BDF files
│       └── processed_lmdb/                   ← VI LMDB (~3 GB)
└── pth/
    └── CSBrain.pth                           ← Pretrained encoder (35.7 MB)
```

---

## 13. Key Differences vs BCICIV2A Pipeline

| Aspect | BCICIV2A | Visual Imagery |
|---|---|---|
| Channels | 22 | 32 |
| Sampling rate | 250 Hz | 1000 Hz |
| Classes | 4 (motor) | 10 (visual) |
| Brain regions | 3 (sensorimotor) | 5 (frontal–occipital) |
| EEG tokens | 12 (3×4) | 20 (5×4) |
| File format | `.mat` | `.bdf` + `.tsv` (BIDS) |
| Subjects | 9 | 22 |
| Total samples | 5,088 | 15,840 |
| Filter method | `lfilter` (IIR) | `sosfiltfilt` (SOS, stable) |
