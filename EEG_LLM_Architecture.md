# EEG-to-Language Model: Architecture Documentation

**Notebook**: `eeg_llm_notebook.ipynb`
**Task**: Motor Imagery Decoding via Natural Language Generation
**Dataset**: BCI Competition IV 2a (BCIC-IV-2a) — 4-class, 9 subjects

---

## Overview

This system is a multimodal EEG-to-Language architecture inspired by **LLaVA** (Large Language and Vision Assistant), adapted for brain signals. It encodes raw EEG into semantic tokens, projects them into an LLM's embedding space, and generates free-form neuroscience descriptions of the motor imagery task being performed.

```
EEG (batch, 22, 4, 200)
  ↓
[CSBrain Encoder — frozen]          (batch, 22, 4, 200)
  ↓
[EEGTokenReducer — region pooling]  (batch, 3 regions × 4 patches = 12, 200)
  ↓
[EEGProjection — 2-layer MLP]       (batch, 12, 2048)
  ↓
[Concatenate with prompt embeddings]
  prompt_embeds   (batch, 101, 2048)
  eeg_embeds      (batch,  12, 2048)
  target_embeds   (batch,  61, 2048)
  ↓
[TinyLlama-1.1B-Chat — 4-bit NF4 + LoRA]
  → Generated neuroscience description text
```

---

## Components

### 1. CSBrain Encoder (Frozen)

**File**: [models/CSBrain.py](models/CSBrain.py)

The pretrained EEG foundation model. All parameters are **frozen** during LLM training — it acts as a feature extractor only.

| Sub-module | Description |
|---|---|
| `PatchEmbedding` | Conv2D patch projection + positional + spectral (FFT) embeddings |
| `TemEmbedEEGLayer` | Cross-scale temporal embedding with kernel sizes `(1,), (3,), (5,)` |
| `BrainEmbedEEGLayer` | Region-aware spatial embedding |
| `CSBrain_TransformerEncoder` | 12-layer transformer with inter-window temporal + inter-region spatial attention |
| `proj_out` | **Replaced with `nn.Identity()`** — raw d_model features are passed through |

**Input/Output**:
- Input: `(batch, 22, 4, 200)` — 22 channels, 4 temporal patches, 200 time points per patch
- Output: `(batch, 22, 4, 200)` — per-channel-patch feature vectors in d_model=200 space

**BCIC-IV-2a Channel Layout** (22 channels):

| Region ID | Electrodes |
|---|---|
| 0 (Frontal) | Fz |
| 4 (Central/FC) | FC3, FC1, FCZ, FC2, FC4, C5, C3, C1, CZ, C2, C4, C6, CP3, CP1, CPZ, CP2, CP4 |
| 1 (Parietal) | P1, PZ, P2, POZ |

Channels are **re-sorted** by brain region before encoding: `sorted_indices = [0, 18, 19, 20, 21, 1..17]`

---

### 2. EEGTokenReducer

Pools channels within each brain region to reduce the token sequence length before feeding into the LLM.

**Algorithm**:
1. For each brain region, average all channel embeddings in that region → one token per patch per region
2. With 3 regions × 4 temporal patches = **12 EEG tokens total**

**Input**: `(batch, 22, 4, 200)` — from CSBrain
**Output**: `(batch, 12, 200)` — 12 tokens, each 200-dim

This is critical for fitting within the LLM's context window without excessive compute.

---

### 3. EEGProjection (MLP)

A 2-layer MLP that maps EEG feature space (d_model=200) into the LLM's embedding space (2048-dim for TinyLlama).

```
Linear(200 → 2048) → GELU → Dropout(0.1) → Linear(2048 → 2048)
```

**Input**: `(batch, 12, 200)`
**Output**: `(batch, 12, 2048)`

This module is **trainable throughout** — it is the alignment bridge between the EEG encoder and the LLM.

---

### 4. TinyLlama-1.1B-Chat Decoder

**Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
**Quantization**: 4-bit NF4 (BitsAndBytes `load_in_4bit`) with double quantization
**Compute dtype**: float16
**Fine-tuning**: LoRA adapters on `q_proj` and `v_proj` in all attention layers

| LoRA Config | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Trainable params | 1,126,400 (~0.10% of 1.1B) |

**Chat template** (TinyLlama Zephyr format):
```
<|system|>
You are an expert EEG analyst specializing in brain-computer interfaces and motor imagery decoding...
<|user|>
[EEG_TOKENS]
Analyze this EEG recording and describe the motor imagery task...
<|assistant|>
[Generated text / training target]
```

---

## Input Sequence Construction

During a forward pass, three embedding segments are concatenated into a single sequence fed to TinyLlama:

```
[prompt_embeds (101 tokens)] + [eeg_embeds (12 tokens)] + [target_embeds (61 tokens)]
     ↑ text context                 ↑ EEG signal              ↑ supervised output
```

**Loss masking**: Only the `target_embeds` region contributes to cross-entropy loss. The prompt and EEG tokens are masked with label `−100`.

**Attention mask**: All three segments are attended to (full causal attention).

---

## Data Pipeline

### Dataset: BCIC-IV-2a

- **Source**: BNCI Horizon 2020 (9 subjects, `.mat` files)
- **Classes**: Left Hand (0), Right Hand (1), Both Feet (2), Tongue (3)
- **Preprocessing**:
  1. Select 22 EEG channels
  2. Zero-mean per sample
  3. Bandpass filter: 0.3–50 Hz (5th-order Butterworth)
  4. Crop motor imagery window: 2s–6s post-cue
  5. Resample to 800 samples → reshape to `(22, 4, 200)`
  6. Normalize: divide by 100
- **Storage**: LMDB for fast random access
- **Splits**: Train (A01–A05), Val (A06–A07), Test (A08–A09)
- **Sizes**: Train 2,784 | Val 1,152 | Test 1,152

### Label-to-Text Mapping

Each class maps to 3 paraphrases of a neuroscience description (used as training augmentation):

| Class | Key Phrases |
|---|---|
| Left Hand | ERD over right sensorimotor cortex (C4), contralateral mu/beta suppression |
| Right Hand | ERD over left sensorimotor cortex (C3), contralateral mu/beta suppression |
| Both Feet | Bilateral ERD over midline (Cz), supplementary motor area activation |
| Tongue | Bilateral lateral ERD, orofacial motor cortex, tongue homunculus |

Training uses random paraphrase selection per sample; evaluation always uses paraphrase 0.

---

## Training Strategy

### Phase 1 — Projection Warmup (5 epochs)

- **Trainable**: `EEGProjection` only
- **LR**: `5e-4` (= `LR × 2.5`)
- **Goal**: Align EEG embedding space to TinyLlama's input space before introducing LoRA
- LoRA parameters are **frozen** in this phase

### Phase 2 — Joint Training (15 epochs)

- **Trainable**: `EEGProjection` + LoRA adapters
- **LR**: `2e-4`
- **Goal**: End-to-end optimization of the EEG→text generation pipeline

### Common Settings

| Hyperparameter | Value |
|---|---|
| Batch size | 4 (physical) |
| Gradient accumulation | 8 steps → effective batch = 32 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| LR schedule | Cosine annealing (eta_min=1e-6) |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | `torch.amp.autocast` (float16) + GradScaler |
| Max target length | 128 tokens |
| Total epochs | 20 (5 warmup + 15 joint) |

---

## Evaluation

### Keyword Extraction Accuracy

Since the LLM generates free-form text, class prediction is done via **keyword matching**:

| Class | Matching Keywords |
|---|---|
| Left Hand | "left hand", "left-lateralized", "right sensorimotor", "right central", "contralateral" |
| Right Hand | "right hand", "right-lateralized", "left sensorimotor", "left central", "contralateral" |
| Both Feet | "feet", "foot", "bilateral", "midline", "cz", "vertex", "lower limb", "supplementary motor" |
| Tongue | "tongue", "orofacial", "face", "lateral portions" |

The class with the most keyword hits in the generated text is the predicted class.

### Results

| Phase | Val Accuracy |
|---|---|
| Warmup Epoch 1 | 27.69% |
| Warmup Epoch 5 (best warmup) | 28.30% |
| Joint Epoch 6 (**best overall**) | **36.81%** |
| Joint Epoch 15 (final) | 25.61% |
| **Test (best checkpoint)** | **31.34%** |

> Chance level for 4-class classification: **25.00%**

---

## Model Parameter Summary

| Module | Parameters | Trainable |
|---|---|---|
| CSBrain encoder | ~8M | No (frozen) |
| EEGProjection MLP | 4,608,000 | Yes (both phases) |
| TinyLlama base | ~1.1B (4-bit) | No |
| LoRA adapters | 1,126,400 | Yes (phase 2 only) |
| **Total trainable** | **~5.7M** | |

---

## Saved Artifacts

| Path | Contents |
|---|---|
| `pth_downtasks/eeg_llm_bcic/best_projection.pth` | Best `EEGProjection` state dict |
| `pth_downtasks/eeg_llm_bcic/best_lora/` | Best LoRA adapter weights (HuggingFace PEFT format) |

---

## Key Design Decisions

1. **Frozen CSBrain**: Avoids catastrophic forgetting of pretrained EEG representations; keeps GPU memory low
2. **EEGTokenReducer**: Reduces 22×4=88 tokens to 12 — essential for fitting within TinyLlama's context on 8GB VRAM
3. **4-bit quantization**: Reduces TinyLlama from ~4.4GB to ~0.7GB VRAM, leaving room for activations and gradients
4. **Two-phase training**: Stabilizes training by first aligning the projection before introducing LoRA updates
5. **Keyword accuracy metric**: Enables classification evaluation without requiring exact text match
6. **`max_new_tokens=32` at eval**: Keywords appear in early generated tokens; short generation avoids OOM during validation
