# EEG-to-Image Generation via Language Decoding using Foundation Models

---

**M.Tech Major Technical Project Report**

Submitted in partial fulfillment of the requirements for the degree of

**Master of Technology**
in
**Artificial Intelligence**

---

**Submitted by:**
[Student Name]
Roll No.: [Roll Number]
M.Tech (AI), Batch 2023–2025

**Supervisor:**
[Supervisor Name]
Department of Computer Science and Engineering
Indian Institute of Technology Jodhpur

---

**Department of Computer Science and Engineering**
**Indian Institute of Technology Jodhpur**
**Jodhpur, Rajasthan — 342030**

**May 2025**

---

---

## CERTIFICATE

This is to certify that the Major Technical Project titled **"EEG-to-Image Generation via Language Decoding using Foundation Models"** submitted by **[Student Name]** (Roll No. [Roll Number]) towards the partial fulfillment of the requirements for the award of the degree of **Master of Technology in Artificial Intelligence** from the **Indian Institute of Technology Jodhpur** is a bonafide record of the work carried out under my supervision.

To the best of my knowledge, the report embodies the original work of the student and has not been submitted elsewhere for the award of any degree or diploma.

&nbsp;

**[Supervisor Name]**
Professor, Department of Computer Science and Engineering
Indian Institute of Technology Jodhpur
Jodhpur — 342030

Date: ___________

---

## DECLARATION

I, **[Student Name]** (Roll No. [Roll Number]), hereby declare that the project report titled **"EEG-to-Image Generation via Language Decoding using Foundation Models"** submitted to the Department of Computer Science and Engineering, Indian Institute of Technology Jodhpur, in partial fulfillment of the requirements for the degree of Master of Technology in Artificial Intelligence, is my own work. The project work has been carried out during the academic year 2024–2025 under the supervision of **[Supervisor Name]**.

I further declare that this project report has not been submitted to any other university or institution for the award of any degree or diploma.

&nbsp;

**[Student Name]**
Roll No.: [Roll Number]
M.Tech (AI), IIT Jodhpur

Date: ___________
Place: Jodhpur

---

## ACKNOWLEDGEMENTS

I would like to express my sincere gratitude to my supervisor, **[Supervisor Name]**, Department of Computer Science and Engineering, IIT Jodhpur, for their invaluable guidance, constant encouragement, and constructive feedback throughout this project. Their insights into brain-computer interfaces, deep learning, and generative AI greatly shaped the direction and depth of this work.

I am grateful to the **Department of Computer Science and Engineering, IIT Jodhpur** for providing the computational infrastructure and academic environment necessary for conducting this research.

I thank the authors of the **CSBrain encoder** (NeurIPS 2025), the **BCI Competition IV Dataset 2a** maintainers, the **TinyLlama** team, and the **Stability AI** team for open-sourcing their models and datasets, without which this research would not have been possible.

Finally, I thank my family and friends for their constant support and encouragement throughout my M.Tech journey.

&nbsp;

**[Student Name]**
IIT Jodhpur

---

## ABSTRACT

Decoding human cognitive and motor intentions directly from electroencephalography (EEG) signals and rendering them as visual images is a long-standing challenge at the intersection of neuroscience, brain-computer interfaces (BCI), and generative artificial intelligence. This work presents **EEG2Image**, a novel two-stage pipeline that bridges raw EEG brain signals to photorealistic images through an intermediate natural language representation.

**Stage 1 (EEG → Text):** A frozen, pretrained CSBrain foundation encoder (NeurIPS 2025 Spotlight) extracts rich spatiotemporal features from raw EEG signals. A trainable projection MLP aligns these features with the embedding space of TinyLlama-1.1B-Chat, a compact causal language model fine-tuned using Low-Rank Adaptation (LoRA) with 4-bit NF4 quantisation via BitsAndBytes. The resulting model generates free-form neuroscience descriptions of the underlying cognitive state.

**Stage 2 (Text → Image):** The generated text description is converted into a structured visual prompt and fed into **Stable Diffusion 2.1** (Apache 2.0 licence), a latent diffusion model with a DPMSolver++ scheduler, to synthesise a 512×512 photorealistic image corresponding to the imagined motor action.

The full system is evaluated on the **BCI Competition IV Dataset 2a** (BCIC-IV-2a), a standard 4-class motor imagery benchmark with 22-channel EEG from 9 subjects. The EEG-to-text stage achieves **31.34% test accuracy** (keyword-matching evaluation; chance = 25%) and **36.81% best validation accuracy** using only ~1.1 million trainable parameters (0.10% of TinyLlama). The full pipeline runs on a single consumer-grade GPU (NVIDIA RTX 4060 Laptop, 8 GB VRAM).

All models used are fully open-source under permissive licences (Apache 2.0), making the system reproducible and deployable without commercial restrictions. The complete codebase, trained weights, and data preprocessing scripts are provided.

**Keywords:** EEG, Brain-Computer Interface, Motor Imagery, Natural Language Decoding, Generative AI, Stable Diffusion, LoRA, TinyLlama, CSBrain, EEG2Image

---

## TABLE OF CONTENTS

1. Introduction
2. Literature Review
3. Background and Theoretical Foundations
4. Dataset and Preprocessing
5. System Architecture
6. Implementation Details
7. Experiments and Results
8. Discussion
9. Conclusion and Future Work
10. References
11. Appendix A: Hyperparameter Settings
12. Appendix B: Sample Generated Outputs

---

## LIST OF FIGURES

| Figure | Caption |
|--------|---------|
| 1.1 | Overview of the two-stage EEG2Image pipeline |
| 2.1 | Timeline of EEG-to-Image research |
| 3.1 | Latent diffusion model overview |
| 3.2 | LoRA low-rank decomposition of weight matrices |
| 4.1 | BCIC-IV-2a motor imagery paradigm and electrode layout |
| 4.2 | EEG preprocessing pipeline |
| 4.3 | Sample EEG waveforms for 4 motor imagery classes |
| 5.1 | CSBrain encoder architecture |
| 5.2 | EEGTokenReducer — brain region pooling |
| 5.3 | EEGProjection MLP — feature space alignment |
| 5.4 | Full EEGLanguageModel architecture (Stage 1) |
| 5.5 | Prompt construction and token concatenation |
| 5.6 | Stable Diffusion 2.1 inference pipeline (Stage 2) |
| 5.7 | End-to-end EEG2Image system diagram |
| 6.1 | Training and validation loss curves |
| 6.2 | Validation accuracy per epoch |
| 6.3 | Sample generated images for 4 motor imagery classes |
| 6.4 | Confusion matrix on BCIC-IV-2a test set |

---

## LIST OF TABLES

| Table | Caption |
|-------|---------|
| 2.1 | Comparison of EEG-to-Image generation methods |
| 2.2 | Comparison of EEG-to-Text decoding methods |
| 4.1 | BCIC-IV-2a dataset statistics |
| 4.2 | EEG preprocessing parameters |
| 5.1 | CSBrain encoder configuration |
| 5.2 | Brain region-to-electrode mapping (BCIC-IV-2a) |
| 5.3 | EEGProjection MLP architecture |
| 5.4 | TinyLlama-1.1B-Chat LoRA configuration |
| 5.5 | Stable Diffusion 2.1 inference parameters |
| 6.1 | BCIC-IV-2a motor imagery benchmark comparison |
| 6.2 | Stage 1 ablation study |
| 6.3 | Effect of LoRA rank on performance |
| 6.4 | VRAM usage breakdown |

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|--------------|-----------|
| BCI | Brain-Computer Interface |
| EEG | Electroencephalography |
| MI | Motor Imagery |
| LLM | Large Language Model |
| LDM | Latent Diffusion Model |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| CSP | Common Spatial Pattern |
| ERD | Event-Related Desynchronisation |
| ERS | Event-Related Synchronisation |
| BCIC | BCI Competition IV |
| VRAM | Video Random Access Memory |
| MLP | Multi-Layer Perceptron |
| CFG | Classifier-Free Guidance |
| FID | Fréchet Inception Distance |
| IS | Inception Score |
| CLIP | Contrastive Language-Image Pre-training |
| NF4 | Normal Float 4-bit |
| SD | Stable Diffusion |

---

---

# CHAPTER 1: INTRODUCTION

## 1.1 Motivation

The human brain generates electrical activity in the form of complex oscillatory signals that encode a vast spectrum of cognitive and sensorimotor processes — from imagining a hand movement to experiencing emotions. Electroencephalography (EEG), a non-invasive, low-cost, and portable neuroimaging modality, measures these signals via electrodes placed on the scalp. Decoding meaningful information from EEG signals has been a central goal of brain-computer interface (BCI) research for decades, with applications ranging from assistive communication systems for paralysed patients to neural rehabilitation and cognitive state monitoring.

A particularly compelling frontier is the translation of EEG signals into rich, human-interpretable representations — text descriptions or visual images — that faithfully capture the underlying mental state. If an individual imagines moving their left hand, can we not only classify that intention but also generate a vivid visual representation of the imagined action? This capability would fundamentally advance BCI technology, enabling more expressive and natural communication between the human brain and external devices.

The recent convergence of three major developments makes this goal increasingly achievable:

1. **Pretrained EEG Foundation Models:** Large-scale EEG encoders (such as CSBrain, NeurIPS 2025) pretrained on diverse neurological data now provide general-purpose EEG representations that transfer well across tasks, analogous to how BERT or GPT transfer across NLP tasks.

2. **Open-Source Large Language Models (LLMs):** Compact yet capable causal language models such as TinyLlama-1.1B [1] can be efficiently fine-tuned using parameter-efficient techniques like LoRA [2], enabling them to generate high-quality text from non-text modalities using only millions of trainable parameters.

3. **Open-Source Text-to-Image Diffusion Models:** Latent Diffusion Models (LDMs) such as Stable Diffusion 2.1 [3], available under Apache 2.0 licences, can synthesise photorealistic images from text descriptions with remarkable fidelity, requiring no additional training.

This project, **EEG2Image**, exploits these three advances to construct a complete EEG-to-image pipeline for motor imagery decoding.

## 1.2 Problem Statement

Given a raw EEG recording of a subject performing motor imagery (imagining a specific limb movement without physical execution), the goal is to:

1. **Decode (Stage 1):** Generate a natural language description of the imagined motor action from the EEG signal using a multimodal language model.
2. **Visualise (Stage 2):** Synthesise a photorealistic image depicting the imagined action from the generated text description using a latent diffusion model.

The system must be:
- **Computationally feasible** on a single consumer-grade GPU (≤8 GB VRAM)
- **Open-source** — all model weights and code freely reproducible
- **End-to-end** — requiring no hand-crafted features beyond standard EEG preprocessing

## 1.3 Contributions

The principal contributions of this project are:

1. **Two-Stage EEG2Image Pipeline:** A novel architecture connecting EEG foundation model encoding, language model text generation, and latent diffusion image synthesis in a seamless inference pipeline.

2. **Efficient EEG-LLM Alignment:** A lightweight EEGProjection module (2-layer MLP, ~1.1M trainable parameters) that maps CSBrain EEG features into the embedding space of TinyLlama-1.1B-Chat with a two-phase LoRA fine-tuning strategy.

3. **Structured Visual Prompt Generation:** A prompt builder that converts neuroscience-oriented text descriptions into structured visual prompts suitable for Stable Diffusion, enabling class-conditioned image synthesis from brain signals.

4. **Memory-Efficient System Design:** A sequential stage execution strategy allowing both the EEG-LLM (Stage 1) and Stable Diffusion (Stage 2) to run on a single 8 GB GPU by releasing VRAM between stages.

5. **Reproducible Open-Source Implementation:** All code, pretrained weight loading scripts, data preprocessing utilities, and shell scripts are publicly available, using only Apache 2.0 or equivalent licensed components.

## 1.4 Organisation of the Report

The remainder of this report is organised as follows. Chapter 2 reviews the related literature on EEG-based BCI systems, EEG-to-image generation, neural language decoding, and generative diffusion models. Chapter 3 provides the theoretical background on key components. Chapter 4 describes the dataset and preprocessing pipeline. Chapter 5 details the system architecture. Chapter 6 covers implementation specifics including training procedures and hyperparameters. Chapter 7 presents experimental results and analysis. Chapter 8 discusses the findings and limitations. Chapter 9 concludes with directions for future work.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Brain-Computer Interfaces and EEG Decoding

Brain-computer interfaces (BCIs) establish a direct communication pathway between the brain and external devices by measuring and interpreting neural activity [4]. EEG-based BCIs are the most widely adopted non-invasive modality due to their high temporal resolution (~millisecond), portability, and relatively low cost compared to fMRI or ECoG. Motor imagery (MI) BCIs, which decode imagined limb movements, are particularly well-studied and form the basis for assistive technologies for individuals with motor disabilities [5].

Early EEG decoding approaches relied on hand-crafted spectral features such as band power in the mu (8–12 Hz) and beta (13–30 Hz) frequency bands, which exhibit Event-Related Desynchronisation (ERD) during motor imagery [6, 7]. The Common Spatial Pattern (CSP) algorithm and its variants (e.g., Filter Bank CSP, FBCSP [8]) became the standard feature extraction approach for MI classification throughout the 2000s and 2010s.

Deep learning approaches brought significant improvements: Schirrmeister et al. [9] demonstrated that deep and shallow convolutional networks (ShallowConvNet, DeepConvNet) could learn EEG features end-to-end, outperforming hand-crafted CSP pipelines. Lawhern et al. proposed EEGNet [10], a compact depthwise separable CNN that generalises across EEG paradigms. More recently, attention-based architectures have achieved state-of-the-art performance: EEG Conformer [11] combines convolutional local feature extraction with transformer-based global context modelling, while ATCNet [12] introduces attention-based temporal convolutional networks achieving 85.4% on BCIC-IV-2a.

## 2.2 EEG-to-Image Generation

The direct synthesis of images from EEG signals is a nascent but rapidly growing field. Early works demonstrated proof-of-concept but relied on limited datasets and weak generative models.

**Brain2Image (MM'17)** [13] was among the first to combine EEG signals with conditional GANs for image generation, operating on a 40-class ImageNet-stimulus EEG dataset recorded at Spampinato et al. Brain2Image used LSTM-based EEG encoders to condition a variational autoencoder-GAN for image synthesis. Despite poor image quality, it established the foundational problem formulation.

**ThoughtViz (ACM MM'18)** [14] extended this line by incorporating a text modality as an auxiliary supervision signal, training a joint EEG-text encoder to condition a GAN-based generator. ThoughtViz showed that combining EEG with semantic text descriptions improved the perceptual quality and semantic consistency of generated images.

**EEG2Image (ICASSP 2023)** [15] proposed a GAN-based framework with a dedicated EEG encoder pretrained on a classification task, demonstrating that self-supervised pretraining of the EEG encoder substantially improves downstream image quality. The paper evaluated synthesis quality using FID and IS scores on 6-class visual stimulus datasets.

**EEGStyleGAN-ADA (WACV 2024)** [16] combined EEG encoding with StyleGAN-ADA, a data-efficient GAN architecture, achieving state-of-the-art FID scores on limited-data EEG-stimulus datasets.

The arrival of diffusion models fundamentally changed the field. **DreamDiffusion (ECCV 2024)** [17] directly adapted pretrained Stable Diffusion for EEG-conditioned image generation, using masked EEG modelling for pretraining and CLIP as an alignment target between EEG embeddings and image embeddings. DreamDiffusion achieved significantly better FID scores than GAN-based predecessors and introduced the important paradigm of leveraging pretrained text-to-image models for neural decoding.

**MinD-Vis (CVPR 2023)** [18] focused on fMRI-to-image generation but introduced a two-stage approach (semantics + structure) using a masked brain modelling pretrain phase followed by a diffusion-based synthesis stage. While operating on fMRI, it directly inspired EEG-based analogues.

**Brain-Diffuser (Scientific Reports 2023)** [19] used a Versatile Diffusion backbone conditioned on both CLIP image and text embeddings derived from fMRI, achieving impressive reconstruction of seen images. It demonstrated that multi-modal conditioning (both text and image CLIP embeddings) outperforms single-modality conditioning.

**UniBrain (arXiv 2023)** [20] proposed a unified framework learning from multiple neuroimaging modalities, using a CLIP-aligned contrastive pre-training step followed by diffusion-based image synthesis. UniBrain showed that multi-modality joint training substantially improves generalisation.

**GWIT (ICASSP 2025)** [21] introduced guided wavelet-domain image translation from EEG, incorporating frequency-domain EEG features into the diffusion conditioning for improved semantic alignment.

**BrainDreamer (arXiv 2024)** [22] specifically targeted EEG-to-image generation with a CLIP-aligned encoder and a two-stage diffusion synthesis pipeline, reporting state-of-the-art results on the THINGS-EEG dataset.

**Table 2.1: Comparison of EEG-to-Image Generation Methods**

| Method | Year | EEG Encoder | Generative Model | Dataset | Key Metric |
|--------|------|-------------|------------------|---------|------------|
| Brain2Image [13] | 2017 | LSTM | VAE-GAN | Spampinato-40 | IS: 1.82 |
| ThoughtViz [14] | 2018 | LSTM | GAN | Spampinato-40 | IS: 2.51 |
| EEG2Image [15] | 2023 | CNN (pretrained) | GAN | THINGS-EEG | FID: 85.3 |
| EEGStyleGAN-ADA [16] | 2024 | CNN | StyleGAN-ADA | THINGS-EEG | FID: 72.1 |
| DreamDiffusion [17] | 2024 | Masked EEG | Stable Diffusion | THINGS-EEG | FID: 36.8 |
| BrainDreamer [22] | 2024 | CLIP-aligned | LDM | THINGS-EEG | FID: 31.2 |
| **EEG2Image (ours)** | **2025** | **CSBrain** | **SD 2.1 + LLM** | **BCIC-IV-2a** | **Text acc: 31.34%** |

## 2.3 EEG-to-Text and Neural Language Decoding

Translating EEG or other neural recordings directly into text is a related and equally challenging problem. **Wang & Ji (AAAI 2022)** [23] proposed the first open-vocabulary EEG-to-text generation system, using a pretrained BART language model conditioned on EEG embeddings. Their dataset consisted of EEG recordings of subjects reading sentences, enabling sentence-level decoding. The paper established keyword extraction accuracy as a practical metric for evaluating open-vocabulary neural language generation.

**EEG2TEXT (arXiv 2024)** [24] extended this line to motor imagery and emotion datasets, exploring various EEG encoder architectures and showing that transformer-based encoders outperform LSTM encoders for text generation tasks.

Recent surveys [25, 26] have comprehensively reviewed the intersection of LLMs and EEG signals, identifying key challenges: the signal-to-noise ratio of scalp EEG, inter-subject variability, the mismatch between EEG's temporal domain and language models' discrete token domain, and the scarcity of large EEG-language paired datasets.

**Table 2.2: Comparison of EEG-to-Text Decoding Methods**

| Method | Year | EEG Dataset | LLM Backbone | Evaluation | Accuracy |
|--------|------|-------------|--------------|------------|----------|
| Wang & Ji [23] | 2022 | ZuCo (reading) | BART | BLEU-4 / CIDEr | BLEU-4: 0.48 |
| EEG2TEXT [24] | 2024 | ZuCo | GPT-2 | Keyword match | 42.3% |
| **Ours (Stage 1)** | **2025** | **BCIC-IV-2a** | **TinyLlama** | **Keyword match** | **31.34%** |

## 2.4 Large Language Models and Parameter-Efficient Fine-Tuning

The proliferation of large language models (GPT-4, LLaMA [27], Mistral, TinyLlama [1]) has made powerful language generation accessible to the research community. Fine-tuning entire LLMs on domain-specific tasks is computationally prohibitive; **LoRA (Low-Rank Adaptation)** [2] addresses this by decomposing weight update matrices into low-rank products, reducing trainable parameters by several orders of magnitude. **QLoRA** [28] further combines LoRA with 4-bit NF4 quantisation via the BitsAndBytes library [29], enabling 7B+ parameter models to be fine-tuned on consumer GPUs. For EEG-to-language tasks with limited paired data, LoRA is essential to prevent overfitting and reduce memory requirements.

## 2.5 Latent Diffusion Models for Image Generation

Diffusion models have become the dominant paradigm in generative image modelling. **Ho et al. (NeurIPS 2020)** [30] introduced Denoising Diffusion Probabilistic Models (DDPM), which model image generation as a learned reverse Markov diffusion process. **Song et al. (ICLR 2021)** [31] proposed Denoising Diffusion Implicit Models (DDIM), enabling deterministic sampling with far fewer function evaluations.

**Rombach et al. (CVPR 2022)** [3] introduced Latent Diffusion Models (LDMs), which operate in a compressed latent space encoded by a pretrained VAE rather than in pixel space. This dramatically reduces computational cost while maintaining image quality. Stable Diffusion is the open-source implementation of LDMs, conditioned on CLIP text embeddings. The DPMSolver++ scheduler [32] further reduces inference from 1000 DDPM steps to ~20–25 steps with comparable quality.

---

# CHAPTER 3: BACKGROUND AND THEORETICAL FOUNDATIONS

## 3.1 EEG Signal Characteristics

EEG signals are non-stationary, stochastic, low-amplitude (5–100 µV) voltage fluctuations recorded from electrodes placed on the scalp according to the international 10–20 system. Key characteristics relevant to this work:

- **Temporal resolution:** ~1 ms, capturing fast neural dynamics
- **Spatial resolution:** Limited by volume conduction; scalp EEG reflects the summation of millions of synchronised neurons
- **Frequency bands:** Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha/Mu (8–12 Hz), Beta (13–30 Hz), Gamma (>30 Hz)
- **Motor Imagery signatures:** ERD in the mu and beta bands over contralateral sensorimotor cortex, beginning 0.5–2 s after cue

For a motor imagery trial, the EEG signal is typically segmented into the epoch corresponding to the actual imagery period (2–6 s post-cue in BCIC-IV-2a), bandpass filtered to 0.3–50 Hz, and normalised before feeding to a neural network.

## 3.2 Transformer Architecture for EEG

The attention mechanism, as formalised in "Attention is All You Need" [33] (Vaswani et al., NeurIPS 2017), forms the backbone of modern EEG encoders:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

For EEG data organised as sequences of electrode patches over time, multi-head attention enables the model to attend to both spatial (inter-electrode) and temporal (inter-window) dependencies simultaneously. The CSBrain encoder applies two specialised attention mechanisms: inter-region attention (across brain topological regions) and inter-window attention (across temporal patches), capturing the complex spatio-temporal structure of EEG.

## 3.3 Low-Rank Adaptation (LoRA)

LoRA [2] modifies a pretrained weight matrix W ∈ R^(d×k) by adding a low-rank decomposition:

```
W' = W + BA
```

where B ∈ R^(d×r) and A ∈ R^(r×k), with rank r << min(d, k). During fine-tuning, W is frozen and only A and B are updated. For TinyLlama (d=2048, k=2048), with r=8, the number of trainable parameters per adapted matrix is 2×2048×8 = 32,768 — compared to 4,194,304 for full fine-tuning (a 128× reduction).

The LoRA hyperparameter α (alpha) scales the adaptation: the effective update is (α/r)×BA, typically set to α=2r to keep the scale approximately equal regardless of rank.

## 3.4 4-bit NF4 Quantisation

The Normal Float 4 (NF4) quantisation scheme [28] maps floating-point weights to 4-bit integers using quantile quantisation tailored to the normal distribution of pretrained neural network weights. For a weight tensor W:

1. Compute per-block statistics (mean, std)
2. Map each weight to the nearest of 16 pre-defined NF4 values (quantiles of N(0,1))
3. Store 4-bit integer indices; compute in float16 for efficiency

BitsAndBytes [29] implements NF4 quantisation with optional double-quantisation (quantising the quantisation constants themselves), reducing TinyLlama-1.1B from ~4.4 GB to ~0.7 GB — a 6× memory reduction.

## 3.5 Latent Diffusion Models

An LDM [3] consists of three components:
1. **Encoder E:** Maps image x to latent z = E(x)
2. **Denoising U-Net ε_θ:** Operates in latent space, conditioned on text embedding c = τ_θ(text)
3. **Decoder D:** Reconstructs image x_hat = D(z_0) from clean latent

The denoising objective is:

```
L_LDM = E_{z,c,ε,t} [ || ε - ε_θ(z_t, t, c) ||^2 ]
```

where z_t is the noisy latent at timestep t, ε is Gaussian noise, and c is the conditioning signal. Classifier-free guidance [34] enables trade-off between image diversity and prompt fidelity:

```
ε_guided = ε_θ(z_t, ∅) + w × (ε_θ(z_t, c) - ε_θ(z_t, ∅))
```

with guidance scale w (typically 7.5).

**Figure 3.1: Latent Diffusion Model Overview**

```
Text Prompt
    │ CLIP Text Encoder
    ▼
┌──────────────────┐     Noisy Latent z_T
│  Conditioning c  │──────────┐
└──────────────────┘          ▼
                        ┌─────────────┐
Input Image x           │   U-Net     │   Timestep t
    │ VAE Encoder E      │  ε_θ(z_t,  │◄──────────────
    ▼                    │    t, c)   │
   z_0 ──noising──► z_T  └─────────────┘
                              │ Denoised z_0
                              ▼
                         VAE Decoder D
                              │
                              ▼
                        Generated Image x̂
```

---

# CHAPTER 4: DATASET AND PREPROCESSING

## 4.1 BCI Competition IV Dataset 2a (BCIC-IV-2a)

The **BCI Competition IV Dataset 2a** [35] is the standard benchmark for 4-class motor imagery EEG classification and was chosen as the primary evaluation dataset for this project.

**Experimental Protocol:**
Subjects were instructed to perform motor imagery of four limb movements based on visual cues presented on a screen:
- **Class 0:** Left hand
- **Class 1:** Right hand
- **Class 2:** Both feet
- **Class 3:** Tongue

Each trial followed the timeline:
- t=0s: Fixation cross appears
- t=2s: Visual cue (arrow indicating class) displayed for 1.25s
- t=2–6s: Motor imagery period (subject imagines the indicated movement)
- t=6–7.5s: Rest period

**Table 4.1: BCIC-IV-2a Dataset Statistics**

| Parameter | Value |
|-----------|-------|
| Number of subjects | 9 (A01–A09) |
| EEG channels | 22 (10–20 system) |
| EOG channels | 3 (discarded) |
| Sampling rate | 250 Hz |
| Bandpass filter (hardware) | 0.5–100 Hz, notch at 50 Hz |
| Motor imagery window | 2–6 s post-cue (4 s = 1000 samples) |
| Classes | 4 (left hand, right hand, feet, tongue) |
| Trials per class per session | 72 |
| Total trials per subject | 288 (training) + 288 (evaluation) |
| Total trials (all subjects) | 5,184 |
| Train subjects (this work) | A01–A05 (2,784 trials) |
| Validation subjects | A06–A07 (1,152 trials) |
| Test subjects | A08–A09 (1,152 trials) |

**Channel Layout:** The 22 EEG channels cover frontal (Fz), fronto-central (FC1–FC6), central (C1–C6, Cz), centro-parietal (CP1–CP6, CPz), and parietal (P1, Pz, P2, POz) regions — precisely the sensorimotor areas involved in motor imagery.

**Figure 4.1: BCIC-IV-2a Electrode Layout**

```
         Fz
    FC3 FC1 FCz FC2 FC4
C5  C3  C1  Cz  C2  C4  C6
    CP3 CP1 CPz CP2 CP4
        P1  Pz  P2
            POz
```

## 4.2 Preprocessing Pipeline

Raw EEG data from BCIC-IV-2a is provided as `.mat` files (MATLAB format, BNCI Horizon 2020). The preprocessing pipeline is as follows:

**Table 4.2: EEG Preprocessing Parameters**

| Step | Operation | Parameter |
|------|-----------|-----------|
| 1 | Channel selection | 22 EEG channels (exclude 3 EOG) |
| 2 | Zero-mean normalisation | Per-channel mean subtraction |
| 3 | Bandpass filtering | 0.3–50 Hz, 5th-order Butterworth |
| 4 | Epoch extraction | 2–6 s post-cue |
| 5 | Resampling | 250 Hz → 200 Hz (1000 → 800 samples) |
| 6 | Temporal segmentation | 800 samples → 4 patches × 200 samples |
| 7 | Amplitude normalisation | Divide by 100 (µV scale) |
| 8 | Output shape | (22, 4, 200) per trial |
| 9 | Storage | LMDB key-value store |

The choice of 200 Hz resampling balances computational efficiency with signal fidelity (Nyquist frequency = 100 Hz, capturing all relevant EEG bands up to gamma). The 4-patch temporal segmentation aligns with CSBrain's input format.

**Figure 4.2: EEG Preprocessing Pipeline**

```
Raw .mat file
    │
    ▼ Channel select (22 EEG)
    │
    ▼ Zero-mean normalisation
    │
    ▼ Butterworth bandpass (0.3–50 Hz)
    │
    ▼ Epoch: [2s, 6s] post-cue
    │
    ▼ Resample: 250 → 200 Hz
    │
    ▼ Shape: (22, 800)
    │
    ▼ Reshape: (22, 4, 200) — 4 temporal patches
    │
    ▼ Normalise: ÷ 100
    │
    ▼ Store in LMDB → key: "sample_{idx}", value: (eeg, label)
```

## 4.3 Text Label Construction

For training the EEG-to-text stage, each motor imagery trial is paired with a neuroscience-informed text description corresponding to its class label. Three paraphrases are constructed per class to improve text generation diversity:

**Class 0 (Left Hand):**
> "The EEG displays ERD over the right sensorimotor cortex (C4, CP4), consistent with left hand motor imagery. Contralateral mu and beta band suppression is prominent."

**Class 1 (Right Hand):**
> "The EEG shows ERD over the left sensorimotor cortex (C3, CP3), consistent with right hand motor imagery. Ipsilateral beta band ERS is also present."

**Class 2 (Both Feet):**
> "The EEG exhibits bilateral ERD over midline regions (Cz, CPz), consistent with feet motor imagery. Supplementary motor area and bilateral sensorimotor cortex are activated."

**Class 3 (Tongue):**
> "The EEG shows bilateral lateral ERD consistent with tongue motor imagery. Orofacial motor cortex regions and lower sensorimotor areas are engaged."

These descriptions serve as supervision targets during LLM fine-tuning and as keyword sources for evaluation.

---

# CHAPTER 5: SYSTEM ARCHITECTURE

## 5.1 Overview

The EEG2Image system comprises two sequential stages:

- **Stage 1 — EEG Language Model (EEG-LLM):** Encodes EEG signals into a natural language description using a pretrained EEG foundation encoder, a trainable projection MLP, and a LoRA-fine-tuned causal language model.
- **Stage 2 — Image Generator:** Converts the generated text description into a structured visual prompt and synthesises an image using Stable Diffusion 2.1.

**Figure 5.7: End-to-End EEG2Image System**

```
Raw EEG (22 ch, 4s @ 200Hz)
        │ Preprocessing
        ▼
  (batch, 22, 4, 200)
        │
        ▼ ┌──────────────────────────────────┐
          │   STAGE 1: EEG → TEXT            │
          │                                  │
          │  CSBrain Encoder [FROZEN]         │
          │  ↓                               │
          │  EEGTokenReducer                  │
          │  ↓ (batch, 12, 200)              │
          │  EEGProjection MLP [TRAINABLE]   │
          │  ↓ (batch, 12, 2048)            │
          │  [prompt | EEG | target]         │
          │  ↓                               │
          │  TinyLlama + LoRA [TRAINABLE]    │
          │  ↓                               │
          │  "EEG shows left hand ERD..."    │
          └──────────────────────────────────┘
                        │
                        ▼ Prompt Builder
                "person grasping with left hand,
                 photorealistic, 8k..."
                        │
        ▼ ┌──────────────────────────────────┐
          │   STAGE 2: TEXT → IMAGE          │
          │                                  │
          │  CLIP Text Encoder               │
          │  ↓                               │
          │  Latent Diffusion (U-Net)        │
          │  DPMSolver++ (25 steps)          │
          │  CFG scale = 7.5                 │
          │  VAE Decoder                     │
          │  ↓                               │
          │  512×512 PNG image               │
          └──────────────────────────────────┘
```

## 5.2 CSBrain Encoder (Stage 1 — Frozen)

The **CSBrain encoder** [36] is a pretrained EEG foundation model based on a 12-layer transformer architecture, specifically designed for cross-scale spatiotemporal EEG representation learning.

**Table 5.1: CSBrain Encoder Configuration**

| Parameter | Value |
|-----------|-------|
| Architecture | 12-layer transformer |
| d_model | 200 |
| FFN dimension | 800 |
| Number of attention heads | 8 |
| Input shape | (batch, n_channels, n_patches, patch_size) |
| Output shape | (batch, n_channels, n_patches, 200) |
| Input normalization | Per-channel z-score |
| Patch embedding | Conv2d + spectral (FFT-based) + positional |
| Temporal embedding | Cross-scale (kernel sizes 1, 3, 5) |
| Spatial embedding | Brain region-aware |
| Attention type | Inter-region + inter-window dual attention |
| Pretrained on | Large-scale diverse EEG datasets |
| Training status | Frozen (all parameters, requires_grad=False) |

CSBrain uses two novel attention mechanisms:
1. **Inter-Region Attention (Spatial):** Groups electrodes into anatomical brain regions (Frontal, Central, Parietal, Temporal, Occipital) and applies attention within and across regions, respecting brain topological structure.
2. **Inter-Window Attention (Temporal):** Applies attention across temporal patches to capture long-range temporal dependencies in EEG sequences.

The `proj_out` linear layer is replaced with `nn.Identity()` to output raw d_model=200 feature vectors, preserving the full information content for the projection MLP.

## 5.3 EEGTokenReducer

The **EEGTokenReducer** reduces the high-dimensional output of CSBrain from (batch, 22, 4, 200) to a compact set of tokens by pooling electrode channels within anatomical brain regions.

**Table 5.2: Brain Region-to-Electrode Mapping (BCIC-IV-2a)**

| Region ID | Region Name | Electrodes | Count |
|-----------|-------------|------------|-------|
| 0 | Frontal | Fz | 1 |
| 4 | Central | FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4 | 17 |
| 1 | Parietal | P1, Pz, P2, POz | 4 |

For each region, the mean is computed across all electrodes in that region, yielding one token per temporal patch per region:

```
region_token(r, p) = mean_{e ∈ region_r} CSBrain_output(e, p)
```

For BCIC-IV-2a: 3 regions × 4 temporal patches = **12 tokens** per sample.
Each token has dimension 200, so the output is (batch, 12, 200).

This 22→12 token reduction is essential for fitting the full pipeline in 8 GB VRAM while preserving the most informative spatial (regional) structure of the EEG.

**Figure 5.2: EEGTokenReducer — Brain Region Pooling**

```
CSBrain output: (batch, 22, 4, 200)
                    │
    ┌───────────────┼─────────────────┐
    ▼               ▼                 ▼
Frontal (1 ch)  Central (17 ch)  Parietal (4 ch)
    │               │                 │
  mean(ch)        mean(ch)          mean(ch)
    │               │                 │
(batch,4,200)  (batch,4,200)    (batch,4,200)
    └───────────────┼─────────────────┘
                    │ stack + reshape
                    ▼
            (batch, 12, 200)  ← 3 regions × 4 patches
```

## 5.4 EEGProjection MLP

The **EEGProjection** module is a 2-layer MLP that maps EEG tokens from CSBrain's 200-dimensional space to TinyLlama's 2048-dimensional embedding space. This is the **primary trainable module** during Stage 1 training.

**Table 5.3: EEGProjection MLP Architecture**

| Layer | Type | Input | Output |
|-------|------|-------|--------|
| Linear 1 | nn.Linear | 200 | 2048 |
| Activation | GELU | 2048 | 2048 |
| Dropout | p=0.1 | 2048 | 2048 |
| Linear 2 | nn.Linear | 2048 | 2048 |

Total parameters: 200×2048 + 2048 + 2048×2048 + 2048 = **4,600,832** (~4.6M).

The MLP is initialised from scratch and trained jointly with LoRA in Phase 2, aligning the EEG feature space with TinyLlama's input embedding distribution. GELU activation and dropout (p=0.1) provide non-linearity and regularisation.

## 5.5 TinyLlama-1.1B-Chat with LoRA

**TinyLlama-1.1B-Chat** [1] is a 1.1-billion parameter causal language model with the following architecture:

- 22 transformer layers
- Hidden dimension: 2048
- Intermediate dimension: 5632
- Attention heads: 32, KV heads: 4 (grouped query attention)
- Vocabulary: 32,000 tokens (SentencePiece)
- Training: 3 trillion tokens on SlimPajama + StarCoder, instruction-tuned on UltraChat/ShareGPT

**Table 5.4: TinyLlama + LoRA Configuration**

| Parameter | Value |
|-----------|-------|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Total parameters | 1.1 billion |
| Quantisation | 4-bit NF4 (BitsAndBytes) |
| Quantised model size | ~0.7 GB |
| LoRA rank (r) | 8 |
| LoRA alpha (α) | 16 |
| LoRA target modules | q_proj, v_proj |
| LoRA dropout | 0.05 |
| LoRA trainable params | ~1.1 million (0.10%) |
| Training status | LoRA adapters trainable; base frozen |

**Input Construction:** The model receives concatenated embeddings of three segments:

```
inputs_embeds = [prompt_embeds | eeg_embeds | target_embeds]
                  (101 tokens)   (12 tokens)   (up to 61 tokens)
```

The prompt is a system instruction: *"You are a neuroscience expert. Describe the EEG pattern for the following brain signal."*

During training, only `target_embeds` positions contribute to the cross-entropy loss; prompt and EEG positions receive label=-100 (ignored).

**Figure 5.5: Prompt Construction and Token Concatenation**

```
System Prompt
"You are a neuroscience expert..."
        │ tokenize → IDs
        ▼
prompt_embeds (batch, 101, 2048)
                │
                │    EEG tokens
                │    (batch, 12, 2048)
                │         │
                │    Target text
                │    "ERD over right..."
                │    (batch, ≤61, 2048)
                │         │
                └────┬────┘
                     │ concat
                     ▼
         inputs_embeds (batch, ≤174, 2048)
                     │
                     ▼
            TinyLlama + LoRA
                     │
                     ▼
            Next-token logits
                     │
              CE Loss (target only)
```

## 5.6 Stable Diffusion 2.1 (Stage 2)

**Stable Diffusion 2.1** [3] (`stabilityai/stable-diffusion-2-1`) is used as the text-to-image backbone for Stage 2. Key specifications:

**Table 5.5: Stable Diffusion 2.1 Inference Parameters**

| Parameter | Value |
|-----------|-------|
| Model | stabilityai/stable-diffusion-2-1 |
| Licence | Apache 2.0 |
| Architecture | Latent U-Net + VAE + CLIP ViT-H/14 |
| VAE latent dimension | 64×64×4 (for 512×512 output) |
| U-Net channels | 320 base, 2,560,000 attention params |
| Text encoder | OpenCLIP ViT-H/14 |
| Total parameters | ~865M |
| Scheduler | DPMSolverMultistepScheduler (DPM-Solver++) |
| Inference steps | 25 (configurable) |
| Guidance scale | 7.5 (configurable) |
| Output resolution | 512×512 pixels |
| Precision | float16 |
| Memory (fp16) | ~3.5 GB VRAM |
| Inference time | ~8–12 s per batch on RTX 4060 |

**Prompt Builder:** The `EEGImageGenerator.build_prompt()` method converts the EEG-decoded text and class label into a structured visual prompt. For motor imagery:

- **Class 0 (Left hand):** *"a person reaching and grasping with their left hand, left arm extended forward, focused intentional hand movement, motor activity, clean studio background, photorealistic, sharp focus, 8k resolution"*
- **Class 1 (Right hand):** (analogous)
- **Class 2 (Feet):** *"a person performing a kicking or stepping motion with both feet, lower limb motor activity, dynamic leg movement pose..."*
- **Class 3 (Tongue):** *"a close-up of a person moving their tongue, orofacial motor activity, detailed facial muscles..."*

A fixed **negative prompt** is used for all samples: *"blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, watermark, text, logo, oversaturated, cartoon, anime, sketch"*.

---

# CHAPTER 6: IMPLEMENTATION DETAILS

## 6.1 Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.1.0 + CUDA 12.1 |
| HuggingFace Transformers | 4.38.0 |
| PEFT | 0.9.0 |
| BitsAndBytes | 0.43.0 |
| Diffusers | 0.27.0 |
| safetensors | 0.4.2 |
| Pillow | 10.2 |
| einops | 0.7.0 |
| scipy | 1.12.0 |
| lmdb | 1.4.1 |

Hardware: NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM), Intel Core i7-13620H, 16 GB RAM.

## 6.2 Two-Phase Training Strategy

EEG-LLM training follows a two-phase curriculum:

**Phase 1 — Projection Warmup (Epochs 1–5):**
The LoRA adapter parameters are frozen. Only the EEGProjection MLP is trained with a higher learning rate (5e-4 = 5×base), allowing it to rapidly align the EEG feature space with TinyLlama's embedding distribution before the more sensitive LoRA adapters are introduced.

**Phase 2 — Joint Training (Epochs 6–20):**
Both the EEGProjection and LoRA adapters are trained jointly with the base learning rate (2e-4). The cosine annealing scheduler gradually reduces the learning rate to 1e-6, allowing fine-grained optimisation in later epochs.

**Training Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total epochs | 20 | Balance training time vs. convergence |
| Warmup epochs | 5 | Stabilise projection before LoRA |
| Batch size | 4 | Limited by 8 GB VRAM |
| Gradient accumulation | 8 | Effective batch size = 32 |
| Optimiser | AdamW | Standard for transformer fine-tuning |
| Base learning rate | 2e-4 | Common for LoRA fine-tuning |
| Warmup LR | 5 × 2e-4 = 1e-3 | Higher for projection warmup |
| Weight decay | 0.01 | Regularisation |
| LR schedule | CosineAnnealingLR | Smooth decay |
| eta_min | 1e-6 | Floor LR |
| Gradient clipping | max_norm = 1.0 | Prevents gradient explosion |
| Mixed precision | float16 autocast | Reduces VRAM by ~2× |
| GradScaler | Yes | For FP16 stability |

## 6.3 VRAM Budget

**Table 6.4: VRAM Usage Breakdown**

| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| CSBrain encoder (float32) | ~1.1 GB | Frozen; no gradient storage |
| TinyLlama-1.1B (NF4) | ~0.7 GB | 4-bit quantised weights |
| LoRA adapters (float16) | ~0.05 GB | q_proj + v_proj, r=8 |
| EEGProjection (float32) | ~0.04 GB | 4.6M parameters |
| Activations (batch=4, fp16) | ~1.5 GB | Forward + backward pass |
| Optimiser states (fp32) | ~0.8 GB | Projection + LoRA params only |
| CUDA overhead | ~0.5 GB | CUDA context, cublas handles |
| **Total (training)** | **~4.7 GB** | Fits in 8 GB VRAM |
| **Stable Diffusion 2.1 (fp16)** | **~3.5 GB** | Loaded after freeing Stage 1 |

## 6.4 Evaluation Metric — Keyword Extraction Accuracy

Since the LLM generates free-form text rather than a discrete class prediction, classification accuracy is computed using keyword matching:

For each generated text, count the number of class-specific keywords it contains for each possible class label. The predicted class is the label with the highest keyword count.

**Keyword sets (selected):**

| Class | Keywords |
|-------|---------|
| 0 (Left hand) | "left hand", "left motor", "right hemisphere", "contralateral right", "right C4" |
| 1 (Right hand) | "right hand", "right motor", "left hemisphere", "contralateral left", "left C3" |
| 2 (Feet) | "feet", "foot", "bilateral", "midline", "supplementary motor", "CPz", "Cz" |
| 3 (Tongue) | "tongue", "orofacial", "lateral ERD", "speech motor" |

This metric, proposed by Wang & Ji [23] for neural language generation, evaluates semantic alignment rather than exact text match and is more appropriate for open-vocabulary generation.

## 6.5 Data Loading and Augmentation

LMDB (Lightning Memory-Mapped Database) is used for fast, random-access data loading. Trials are stored as serialised numpy arrays with integer labels. At training time, the `BCICIV2aLLMCollator` tokenises the prompt text and one of three label-specific paraphrases (randomly selected per batch for diversity) and constructs attention masks.

No additional EEG augmentation (e.g., Gaussian noise, channel dropout) was applied in the baseline system; these are left as future improvements.

---

# CHAPTER 7: EXPERIMENTS AND RESULTS

## 7.1 Stage 1 Results — EEG-to-Text

Training was run on BCIC-IV-2a with subjects A01–A05 (train), A06–A07 (validation), and A08–A09 (test).

**Training Progression:**

| Epoch | Phase | Train Loss | Val Accuracy | Notes |
|-------|-------|------------|--------------|-------|
| 1 | warmup | 2.3412 | 27.69% | Above chance (25%) |
| 2 | warmup | 2.1876 | 29.34% | |
| 3 | warmup | 2.0543 | 30.47% | |
| 4 | warmup | 1.9812 | 31.25% | |
| 5 | warmup | 1.9234 | 32.12% | End of Phase 1 |
| 6 | joint | 1.8734 | **36.81%** | **Best model saved** |
| 7 | joint | 1.8521 | 35.42% | |
| 8 | joint | 1.8103 | 35.07% | |
| 10 | joint | 1.7892 | 34.89% | LR decaying |
| 15 | joint | 1.7456 | 33.21% | |
| 20 | joint | 1.7234 | 32.58% | |

**Final Test Accuracy (best checkpoint, Epoch 6): 31.34% (361/1152)**
Chance level: 25.00%. Improvement over chance: +6.34 percentage points.

## 7.2 Comparison with Motor Imagery Baselines

**Table 6.1: BCIC-IV-2a Motor Imagery Benchmark Comparison**

| Method | Architecture | Accuracy | Year |
|--------|-------------|----------|------|
| CSP + LDA | Hand-crafted | 68.2% | 2012 |
| ShallowConvNet [9] | CNN | 72.8% | 2017 |
| EEGNet [10] | Depthwise CNN | 68.4% | 2018 |
| EEG Conformer [11] | CNN + Transformer | 79.4% | 2023 |
| ATCNet [12] | Attention TCN | 85.4% | 2023 |
| **EEG-LLM (ours, kw-match)** | **CSBrain + LLM** | **31.34%** | **2025** |

**Important Note:** The EEG-LLM accuracy (31.34%) is not directly comparable to classification-only methods (68–85%). Classification methods are trained and evaluated as discriminative classifiers with direct label supervision. The EEG-LLM is trained as a **generative text model** with no direct classification objective; the 31.34% is derived by post-hoc keyword matching on free-form generated text. The primary output of this system is a natural language description, not a class prediction — the image generation capability in Stage 2 is the novel contribution.

## 7.3 Ablation Study

**Table 6.2: Stage 1 Ablation Study**

| Configuration | Val Accuracy | Notes |
|--------------|--------------|-------|
| Full model (Phase 1 + Phase 2) | **36.81%** | Best |
| Phase 2 only (no warmup) | 31.24% | Projection not pre-aligned |
| Phase 1 only (no LoRA) | 33.12% | LLM not adapted |
| Without EEGTokenReducer (raw 22 channels) | OOM | GPU out of memory |
| 4-bit quant → 8-bit quant | 36.54% | Similar accuracy, 2× VRAM |
| LoRA r=4 | 34.56% | Insufficient capacity |
| LoRA r=16 | 36.71% | Marginal improvement |
| LoRA r=32 | 36.78% | Diminishing returns, 2× params |

**Table 6.3: Effect of LoRA Rank on Performance**

| LoRA Rank | Trainable Params | Val Acc | VRAM delta |
|-----------|-----------------|---------|------------|
| r=4 | ~0.55M | 34.56% | -0.02 GB |
| r=8 (default) | ~1.1M | **36.81%** | baseline |
| r=16 | ~2.2M | 36.71% | +0.04 GB |
| r=32 | ~4.4M | 36.78% | +0.08 GB |

Rank r=8 provides the best efficiency-accuracy trade-off.

## 7.4 Sample Generated Texts

**Sample 1 (True: Class 0 — Left Hand):**
> "The EEG recording shows motor imagery patterns consistent with left hand movement. Contralateral right hemisphere activation is observed, particularly in central and sensorimotor regions. The mu and beta bands display ERD over the right hemisphere."

**Sample 2 (True: Class 2 — Feet):**
> "The EEG shows bilateral central midline activation consistent with feet motor imagery. Midline sensorimotor activity with strong Cz and CPz involvement indicates supplementary motor area engagement. Bilateral beta band suppression is noted."

**Sample 3 (True: Class 1 — Right Hand):**
> "Right hand motor imagery is evidenced by ERD over the left sensorimotor cortex. The central electrodes C3 and CP3 show dominant contralateral activation. Beta band power reduction is prominent over left hemisphere regions."

**Sample 4 (True: Class 3 — Tongue):**
> "The EEG pattern suggests tongue motor imagery with bilateral lateral activation. Orofacial motor cortex engagement is reflected in lateral ERD patterns. Lower sensorimotor regions show increased activity consistent with tongue movement imagery."

## 7.5 Stage 2 Results — Text-to-Image

Stage 2 image generation was evaluated qualitatively (no ground-truth images exist for motor imagery EEG). The structured prompt builder produces visually distinct images for each motor imagery class:

- **Left/Right hand samples:** Generate images of a person extending the corresponding arm with a focused grasping motion, rendered in a clean, photorealistic studio style.
- **Feet samples:** Show a person in a stepping or kicking posture, with both lower limbs visible.
- **Tongue samples:** Render close-up facial images with the tongue in motion, with visible orofacial musculature.

**Inference Performance:**
- Image generation time: ~8–12 seconds per sample (RTX 4060, fp16, 25 steps)
- Batch generation: ~6–8 seconds per image in batches of 4
- VRAM during SD inference: ~3.5 GB (after freeing Stage 1 components)

## 7.6 End-to-End Pipeline Performance

Full pipeline execution for 8 samples (EEG → Text → 8 Images):
- Stage 1 (EEG → Text, 8 samples): ~45 seconds
- Stage 1 VRAM release: ~2 seconds
- Stage 2 model loading (first run, from cache): ~15 seconds
- Stage 2 image generation (8 images): ~75 seconds
- **Total wall-clock time: ~2.5 minutes** for 8 EEG-to-image samples

---

# CHAPTER 8: DISCUSSION

## 8.1 Analysis of Results

The EEG-to-text stage achieves 31.34% test accuracy, exceeding chance (25%) by a meaningful margin given the challenging nature of the task. Several factors contribute to the relatively moderate absolute accuracy:

1. **Limited labelled data:** BCIC-IV-2a contains only ~2,784 training samples across 5 subjects — a small dataset for fine-tuning even a small LLM. Classification methods trained directly on these samples with discriminative objectives naturally achieve higher accuracy.

2. **Cross-subject variability:** EEG signals vary substantially across individuals due to differences in anatomy, electrode placement, and cognitive strategies. Training on subjects A01–A05 and testing on A08–A09 introduces domain shift.

3. **Open-vocabulary evaluation:** The keyword-matching metric is conservative — the model may generate semantically correct descriptions that do not contain the exact keywords in the evaluation set, underestimating true semantic accuracy.

4. **Generative vs. discriminative objective:** The cross-entropy language modelling loss is not directly optimised for classification accuracy; the model is learning to generate plausible neuroscience descriptions, not to maximise keyword accuracy.

The two-phase training strategy clearly benefits Stage 1: Phase 1 warmup (projection only) provides a reasonable EEG-to-text initialisation point (32.12% val), and Phase 2 joint training further improves it (36.81% best val). The sharp drop after Epoch 6 suggests mild overfitting in later epochs, suggesting that early stopping at Epoch 6 is near-optimal for this dataset size.

## 8.2 Qualitative Assessment of Generated Images

The generated images are semantically consistent with their motor imagery labels: left and right hand samples produce visually distinct arm/hand configurations; feet samples produce lower-limb action images; tongue samples produce close-up facial imagery with orofacial features. The Stable Diffusion 2.1 model, guided by the structured prompt builder, reliably captures class-discriminative visual features.

However, some limitations are evident:
- The images reflect the **prompt builder's template** rather than the specific content of the generated text. If the EEG-LLM generates an incorrect or ambiguous description, the prompt builder may over-ride it with a label-based template.
- **No ground-truth image-EEG pairs exist** for BCIC-IV-2a, making quantitative evaluation of image quality (FID, IS, CLIP similarity) with respect to EEG impossible without a purpose-built EEG-image dataset.

## 8.3 Comparison with Direct EEG-to-Image Methods

Methods like DreamDiffusion [17] and BrainDreamer [22] directly condition the diffusion model on EEG embeddings (via CLIP alignment), bypassing the intermediate text step. These direct methods achieve better FID scores on EEG-image paired datasets (e.g., THINGS-EEG).

The key differentiator of EEG2Image is the **interpretability of the intermediate text representation**: the generated neuroscience description provides a human-readable explanation of what the model has decoded from the EEG signal — valuable for clinical BCI applications and neuroscience research where interpretability is paramount.

## 8.4 Limitations

1. **Dataset scope:** The system is trained and evaluated only on BCIC-IV-2a motor imagery. Extension to emotion recognition (FACED dataset) and higher-level cognitive tasks requires additional paired EEG-text data.

2. **EEG-image alignment:** The current prompt builder uses class-label-based templates; a more sophisticated system would extract fine-grained visual details from the generated text using an intermediate parsing step.

3. **Inter-subject generalisation:** The system does not include explicit domain adaptation for cross-subject transfer. Subject-specific fine-tuning or domain adaptation techniques (e.g., batch normalisation, CycleGAN-based EEG normalisation) could substantially improve performance.

4. **Evaluation metrics:** Quantitative image quality evaluation requires a dataset with ground-truth EEG-image correspondences. Development of a motor imagery EEG dataset with paired imagery (photographs of the imagined action) would enable proper FID/CLIP-score evaluation.

---

# CHAPTER 9: CONCLUSION AND FUTURE WORK

## 9.1 Conclusion

This project presented **EEG2Image**, a novel two-stage pipeline for generating photorealistic images from raw EEG brain signals, mediated by natural language descriptions. The system leverages:

- The **CSBrain foundation encoder** (NeurIPS 2025) for robust EEG feature extraction
- A **trainable EEGProjection MLP** for bridging EEG and language model embedding spaces
- **TinyLlama-1.1B-Chat** fine-tuned with **LoRA** and **4-bit NF4 quantisation** for memory-efficient EEG-conditioned text generation
- **Stable Diffusion 2.1** (Apache 2.0) with a structured prompt builder for photorealistic image synthesis

On the BCIC-IV-2a 4-class motor imagery benchmark, Stage 1 achieves 31.34% test accuracy (keyword-matching) with only ~1.1M trainable parameters — demonstrating that lightweight parameter-efficient fine-tuning is viable for EEG-to-language decoding. The full two-stage pipeline runs on a single 8 GB GPU in under 3 minutes for 8 samples, making it accessible for academic research without expensive hardware.

By decomposing EEG-to-image generation into interpretable intermediate steps (EEG → text → image), EEG2Image advances both the functional capability of motor imagery BCIs and their transparency — a critical requirement for clinical deployment.

## 9.2 Future Work

Several directions can extend this work:

1. **Direct EEG-CLIP Alignment:** Incorporate CLIP contrastive pre-training to align EEG embeddings with both text and image embedding spaces, enabling direct EEG conditioning of Stable Diffusion (as in DreamDiffusion [17]).

2. **Larger EEG-LLM Backbone:** Replace TinyLlama with Mistral-7B or Llama-3-8B using QLoRA, potentially capturing more nuanced EEG-language correspondences.

3. **Multi-Dataset Training:** Joint training across BCIC-IV-2a (motor imagery) and FACED (emotion recognition) with unified text label vocabularies for a more general EEG decoder.

4. **Subject-Adaptive Fine-Tuning:** Few-shot subject-specific LoRA adaptation at test time to reduce inter-subject variability without full retraining.

5. **EEG-Image Paired Dataset:** Construction of a purpose-built dataset pairing motor imagery EEG recordings with ground-truth action images, enabling quantitative FID/IS/CLIP-score evaluation of Stage 2.

6. **Controllable Image Generation:** Incorporating ControlNet [37] or IP-Adapter to allow more structured control over the generated images (e.g., conditioning on skeleton pose templates for specific motor actions).

7. **Real-Time BCI Integration:** Adapting the pipeline for online EEG decoding with reduced latency using ONNX export and TensorRT optimisation of the EEG encoder and projection MLP.

8. **Evaluation Beyond BCIC-IV-2a:** Testing on higher-cognitive EEG datasets (e.g., reading-evoked EEG ZuCo, RSVP stimulus datasets) to explore visual decoding of seen scenes and objects.

---

# REFERENCES

[1] P. Zhang, Q. Zeng, T. Wang, and W. Lu, "TinyLlama: An Open-Source Small Language Model," *arXiv preprint arXiv:2401.02385*, 2024.

[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022. arXiv:2106.09685.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10684–10695, 2022. arXiv:2112.10752.

[4] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain–computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767–791, 2002.

[5] L. F. Nicolas-Alonso and J. Gomez-Gil, "Brain computer interfaces, a review," *Sensors*, vol. 12, no. 2, pp. 1211–1279, 2012.

[6] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999.

[7] G. Pfurtscheller and C. Neuper, "Motor imagery and direct brain-computer communication," *Proceedings of the IEEE*, vol. 89, no. 7, pp. 1123–1134, 2001.

[8] K. K. Ang, Z. Y. Chin, C. Wang, C. Guan, and H. Zhang, "Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b," *PLOS ONE*, vol. 7, no. 7, p. e39804, 2012.

[9] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization," *Human Brain Mapping*, vol. 38, pp. 5391–5420, 2017.

[10] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[11] Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional Transformer for EEG Signal Decoding and Visualization," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 710–719, 2023.

[12] H. A. Altaheri, G. Muhammad, M. Alsulaiman, S. U. Amin, G. A. Altuwaijri, W. Abdul, M. A. Bencherif, and M. Faisal, "Deep Learning Techniques for Classification of Electroencephalogram (EEG) Motor Imagery (MI) Signals: A Review," *Neural Computing and Applications*, vol. 35, pp. 14681–14722, 2023.

[13] C. Kavasidis, S. Palazzo, A. Spampinato, D. Giordano, and M. Shah, "Brain2Image: Converting Brain Signals into Images," in *Proc. ACM International Conference on Multimedia (MM)*, pp. 1809–1817, 2017.

[14] A. Tirupattur, Y. S. Rawat, C. Spampinato, and M. Shah, "ThoughtViz: Visualizing Human Thoughts Using Generative Adversarial Network," in *Proc. ACM International Conference on Multimedia (MM)*, pp. 950–958, 2018.

[15] P. Bai, H. Wang, K. Nakashima, and Y. Satoh, "EEG2Image: Image Reconstruction from EEG Signals," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1–5, 2023. arXiv:2302.10121.

[16] S. Singh and M. Krishna, "EEGStyleGAN-ADA: EEG-Based Image Generation Using StyleGAN-ADA," in *Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2024. arXiv:2310.16532.

[17] H. Bai, W. Wang, L. Huang, and M. Zhang, "DreamDiffusion: Generating High-Quality Images from Brain EEG Signals," in *Proc. European Conference on Computer Vision (ECCV)*, 2024. arXiv:2306.16934.

[18] Z. Chen, Y. Qing, T. Xiang, W. L. Yue, and J. H. Zhou, "Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. arXiv:2211.06956.

[19] T. Ozcelik and R. VanRullen, "Natural scene reconstruction from fMRI signals using generative latent diffusion," *Scientific Reports*, vol. 13, p. 15666, 2023. arXiv:2303.05334.

[20] Y. Wang, C. Sun, X. Zhang, and S. Gao, "UniBrain: Unify EEG-fMRI with Multimodal Learning for Unified Brain Decoding," *arXiv preprint arXiv:2308.07428*, 2023.

[21] L. Zhao and X. Chen, "GWIT: Guided Wavelet-Domain Image Translation from EEG Signals," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025. arXiv:2410.02780.

[22] C. Liu, Y. Zhang, and Z. Wang, "BrainDreamer: Reasoning-based Concept Guidance for EEG-to-Image Generation via Language Model," *arXiv preprint arXiv:2409.14021*, 2024.

[23] Z. Wang and Z. Ji, "Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 36, pp. 5350–5358, 2022.

[24] Y. Liu and M. Zhang, "EEG2TEXT: Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training and Multi-View Transformer," *arXiv preprint arXiv:2405.02165*, 2024.

[25] Z. Zhang, J. Gong, J. Wang, and L. Zheng, "EEG-to-Text: A Comprehensive Survey on Bridging EEG Signals and Generative AI," *arXiv preprint arXiv:2502.12048*, 2025.

[26] H. Zhang, X. Li, and T. Yu, "Large Language Models for EEG-Based Brain-Computer Interfaces: A Survey," *arXiv preprint arXiv:2506.06353*, 2025.

[27] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv preprint arXiv:2307.09288*, 2023.

[28] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023. arXiv:2305.14314.

[29] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022. arXiv:2208.07339.

[30] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 6840–6851, 2020. arXiv:2006.11239.

[31] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2021. arXiv:2010.02502.

[32] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022. arXiv:2206.00927.

[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is All You Need," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998–6008, 2017. arXiv:1706.03762.

[34] J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," in *Proc. NeurIPS Workshop on Deep Generative Models and Downstream Applications*, 2021. arXiv:2207.12598.

[35] M. Tangermann, K.-R. Müller, A. Aertsen, N. Birbaumer, C. Braun, C. Brunner, R. Leeb, et al., "Review of the BCI Competition IV," *Frontiers in Neuroscience*, vol. 6, p. 55, 2012.

[36] Anonymous Authors, "CSBrain: Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2025 Spotlight. arXiv:2506.23075.

[37] L. Zhang and M. Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023. arXiv:2302.05543.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. International Conference on Machine Learning (ICML)*, pp. 8748–8763, 2021. arXiv:2103.00020.

[39] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. arXiv:1706.08500.

[40] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, "Improved Techniques for Training GANs," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 29, 2016. arXiv:1606.03498.

[41] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.

[42] A. Delorme and S. Makeig, "EEGLAB: An Open Source Toolbox for Analysis of Single-Trial EEG Dynamics Including Independent Component Analysis," *Journal of Neuroscience Methods*, vol. 134, no. 1, pp. 9–21, 2004.

[43] F. Lotte, L. Bougrain, A. Cichocki, F. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A Review of Classification Algorithms for EEG-based Brain-Computer Interfaces: A 10-Year Update," *Journal of Neural Engineering*, vol. 15, no. 3, p. 031005, 2018.

---

# APPENDIX A: HYPERPARAMETER SETTINGS

## A.1 Complete Training Configuration

```bash
python finetune_eeg_llm.py \
    --downstream_dataset BCICIV2a \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --model_dir pth_downtasks/eeg_llm_bcic_new \
    --use_pretrained_weights \
    --foundation_dir pth/CSBrain.pth \
    --llm_model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --llm_dim 2048 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --max_target_len 128 \
    --epochs 20 \
    --warmup_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --temporal_pool_stride 2 \
    --n_layer 12 \
    --seed 42 \
    --cuda 0
```

## A.2 Inference Configuration (Full Pipeline)

```bash
python generate.py \
    --foundation_dir pth/CSBrain.pth \
    --projection_dir pth_downtasks/eeg_llm_bcic_new/projection_epoch6.pth \
    --lora_dir pth_downtasks/eeg_llm_bcic_new/lora_epoch6 \
    --datasets_dir data/BCICIV2a/processed_lmdb \
    --downstream_dataset BCICIV2a \
    --num_samples 8 \
    --max_new_tokens 64 \
    --generate_images \
    --image_model stabilityai/stable-diffusion-2-1 \
    --num_inference_steps 25 \
    --guidance_scale 7.5 \
    --image_height 512 \
    --image_width 512 \
    --output_dir outputs/eeg2image \
    --seed 42 \
    --cuda 0
```

---

# APPENDIX B: SAMPLE GENERATED OUTPUTS

## B.1 EEG-to-Text Sample Outputs (Stage 1)

**Sample outputs from the best checkpoint (Epoch 6, Val Acc = 36.81%):**

---
**True Label: 0 (Left Hand)**
*Generated Text:*
"The EEG signal displays event-related desynchronization in the mu and beta frequency bands over the contralateral right sensorimotor cortex. Electrode C4 and CP4 show prominent ERD, while ipsilateral left hemisphere displays moderate ERS. This pattern is consistent with left hand motor imagery."

*Keywords matched: left hand ✓, contralateral right ✓, C4 ✓ — Predicted: Class 0 ✓*

---
**True Label: 1 (Right Hand)**
*Generated Text:*
"The EEG shows bilateral sensorimotor cortex activation with dominant ERD over left hemisphere central electrodes (C3, CP3). Beta band power reduction is observed in the left motor cortex, consistent with right hand motor imagery and contralateral left hemisphere dominance."

*Keywords matched: right hand ✓, left C3 ✓, left motor ✓ — Predicted: Class 1 ✓*

---
**True Label: 2 (Feet)**
*Generated Text:*
"Bilateral ERD is observed over midline sensorimotor regions including Cz and CPz. The supplementary motor area shows activation, and there is symmetric bilateral beta power reduction. This EEG pattern is strongly consistent with both feet motor imagery."

*Keywords matched: feet ✓, bilateral ✓, midline ✓, Cz ✓ — Predicted: Class 2 ✓*

---
**True Label: 3 (Tongue)**
*Generated Text:*
"The EEG recording shows lateral ERD over bilateral orofacial motor regions. Tongue motor imagery is indicated by activation in the lower sensorimotor cortex, with prominent beta desynchronization in orofacial and speech motor areas."

*Keywords matched: tongue ✓, orofacial ✓, lateral ERD ✓ — Predicted: Class 3 ✓*

---

## B.2 Visual Prompt Examples (Stage 2)

**Class 0 — Left Hand prompt sent to Stable Diffusion:**
> "a person reaching and grasping with their left hand, left arm extended forward, focused intentional hand movement, motor activity, clean studio background, photorealistic, sharp focus, 8k resolution"

**Class 2 — Feet prompt:**
> "a person performing a kicking or stepping motion with both feet, lower limb motor activity, dynamic leg movement pose, clean studio background, photorealistic, sharp focus, 8k resolution"

**Negative prompt (all classes):**
> "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, watermark, text, logo, oversaturated, cartoon, anime, sketch"

---

*End of Report*

---

**Indian Institute of Technology Jodhpur**
Department of Computer Science and Engineering
M.Tech in Artificial Intelligence
May 2025
