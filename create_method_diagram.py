"""
create_method_diagram.py — NeurIPS-style method figure for the EEG2Image report.

Architecture (verified against the codebase):

  Shared backbone:
    Multichannel EEG (32 ch, 4 s @ 1 kHz, resampled to 800 samples and reshaped
    to 32 x 4 x 200) -> CSBrain encoder (frozen, loaded from pretrained weights)
    -> EEGTokenReducer (region-mean pooling, 5 regions x 4 patches = 20 tokens)
    -> EEG tokens (20 x 200).

  Pipeline 2 (Language-mediated):
    EEG tokens -> EEGProjection MLP (200 -> 2048) -> 20 prefix tokens prepended
    to a text prompt -> TinyLlama-1.1B (4-bit + LoRA r=8) -> generated text
    -> Stable Diffusion 2.1 -> 512 x 512 image.

  Pipeline 3 (Direct CLIP-aligned):
    EEG tokens -> EEGCLIPMapper (input MLP -> 4 transformer encoder layers ->
    learnable 77-query cross-attention -> CLIP-dim projections + classifier
    head). Outputs (B, 77, 768), pooled (B, 768), class logits (B, 10).
    At inference: argmax(class logits) -> class-specific prompt -> SD 1.5 -> image.
    Training losses: cross-entropy on class logits, InfoNCE against CLIP image
    prototypes, cosine to per-sample CLIP image targets, MSE to CLIP text
    targets.

Output: outputs/method_diagram.png and method_diagram.pdf
"""

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import rcParams


# ─── Style ────────────────────────────────────────────────────────────────────

rcParams['font.family']    = 'DejaVu Sans'
rcParams['font.size']      = 10
rcParams['axes.linewidth'] = 0.8
rcParams['savefig.dpi']    = 300

# Muted, print-friendly palette
C_FROZEN   = '#D8E4F2'   # frozen modules — pale blue
C_FROZEN_E = '#5A7BA8'
C_TRAIN    = '#FFE4C2'   # trainable modules — warm orange
C_TRAIN_E  = '#D17A2A'
C_FIXED    = '#E8E8E8'   # fixed/external modules — grey
C_FIXED_E  = '#888888'
C_TEXT     = '#1A1A1A'
C_BG2      = '#F8F4ED'   # Pipeline 2 card background
C_BG3      = '#EDF4F1'   # Pipeline 3 card background
C_BG_S     = '#F1F3F8'   # Shared backbone background
C_LOSS     = '#B83A3A'
C_ARROW    = '#333333'


# ─── Geometry helpers ─────────────────────────────────────────────────────────

@dataclass
class Box:
    x: float; y: float; w: float; h: float
    @property
    def left(self):    return (self.x,            self.y + self.h / 2)
    @property
    def right(self):   return (self.x + self.w,   self.y + self.h / 2)
    @property
    def top(self):     return (self.x + self.w/2, self.y + self.h)
    @property
    def bottom(self):  return (self.x + self.w/2, self.y)
    @property
    def center(self):  return (self.x + self.w/2, self.y + self.h / 2)


def box(ax, x, y, w, h, text='', fill=C_FROZEN, edge=C_FROZEN_E, fc='black',
        fs=10, fw='normal', radius=0.012, lw=1.2, ls='-'):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls,
    )
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=fs, color=fc, fontweight=fw,
                linespacing=1.2)
    return Box(x, y, w, h)


def card(ax, x, y, w, h, title, fill=C_BG_S, edge='#BBBBBB'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.005,rounding_size=0.012",
                       linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=0)
    ax.add_patch(p)
    ax.text(x + 0.012, y + h - 0.020, title,
            ha='left', va='center', fontsize=11, fontweight='bold',
            color='#1F3357')
    return Box(x, y, w, h)


def arrow(ax, p1, p2, color=C_ARROW, lw=1.3, ls='-', mut=12, rad=0.0,
          label=None, lfs=8, lcol='#333', loff=(0, 0.014),
          lbg=True):
    a = FancyArrowPatch(p1, p2,
                        arrowstyle='-|>', color=color, lw=lw,
                        linestyle=ls, mutation_scale=mut,
                        connectionstyle=f'arc3,rad={rad}',
                        zorder=4)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + loff[0], (p1[1] + p2[1]) / 2 + loff[1]
        bbox = dict(facecolor='white', edgecolor='none',
                    pad=1.5, alpha=0.95) if lbg else None
        ax.text(mx, my, label, ha='center', va='center', fontsize=lfs,
                color=lcol, style='italic', bbox=bbox, zorder=5)


def waveforms(ax, x, y, w, h, n_lines=5, color='#3F4757'):
    rect = Rectangle((x, y), w, h, linewidth=0.8,
                     edgecolor='#888', facecolor='white')
    ax.add_patch(rect)
    rng = np.random.default_rng(7)
    xs = np.linspace(x + 0.005, x + w - 0.005, 80)
    for i in range(n_lines):
        baseline = y + h * (i + 1) / (n_lines + 1)
        amp = h * 0.07
        ys = baseline + amp * np.sin(28 * (xs - x) + i * 0.7) \
                      + amp * 0.6 * rng.standard_normal(len(xs))
        ax.plot(xs, ys, color=color, linewidth=0.7)
    return Box(x, y, w, h)


def encoder_stack(ax, x, y, w, h, n_layers=6, fill=C_FROZEN, edge=C_FROZEN_E):
    gap = 0.003
    sw = (w - gap * (n_layers - 1)) / n_layers
    for i in range(n_layers):
        sx = x + i * (sw + gap)
        rect = FancyBboxPatch(
            (sx, y), sw, h,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            linewidth=0.8, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(rect)
    return Box(x, y, w, h)


def img_from_file(ax, path, x, y, w, h, border='#888', label=None, lpos='below'):
    """Embed a real image inside the figure axes at given normalised coords."""
    img = mpimg.imread(path)
    ax.imshow(img, extent=(x, x + w, y, y + h),
              aspect='auto', zorder=2, interpolation='bilinear')
    rect = Rectangle((x, y), w, h, linewidth=0.8,
                     edgecolor=border, facecolor='none', zorder=3)
    ax.add_patch(rect)
    if label:
        if lpos == 'below':
            ax.text(x + w / 2, y - 0.012, label,
                    ha='center', va='top', fontsize=8.5, color='#333')
        else:
            ax.text(x + w / 2, y + h + 0.005, label,
                    ha='center', va='bottom', fontsize=8.5, color='#333')
    return Box(x, y, w, h)


def status_tag(ax, x, y, kind='frozen'):
    if kind == 'frozen':
        fill, edge, txt = '#E6F0FA', '#5A7BA8', 'frozen'
    else:
        fill, edge, txt = '#FFF1DC', '#D17A2A', 'trainable'
    box(ax, x, y, 0.052, 0.020, txt, fill=fill, edge=edge, fs=7.5,
        fw='bold', radius=0.008, lw=0.8)


def token_strip(ax, x, y, w, h, n=10, fill='#F0F0F4', edge='#888'):
    gap = 0.002
    tw = (w - gap * (n - 1)) / n
    for i in range(n):
        sx = x + i * (tw + gap)
        rect = FancyBboxPatch((sx, y), tw, h,
                              boxstyle="round,pad=0.0,rounding_size=0.003",
                              linewidth=0.7, edgecolor=edge, facecolor=fill)
        ax.add_patch(rect)
    return Box(x, y, w, h)


# ─── Build the figure ─────────────────────────────────────────────────────────

PROJ = os.path.dirname(os.path.abspath(__file__))
STIM_BIRD     = os.path.join(PROJ, 'data', 'VisualImagery', 'stimuli', 'Animal_bird.jpg')
STIM_DOG      = os.path.join(PROJ, 'data', 'VisualImagery', 'stimuli', 'Animal_dog.jpg')
GEN_BIRD_P2   = os.path.join(PROJ, 'outputs', 'vi_images',  'generated_correct_class1.png')
GEN_BIRD_P3   = os.path.join(PROJ, 'outputs', 'vi_eeg2img', 'generated_correct_class1.png')

fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Title
ax.text(0.5, 0.972,
        'Two-Pipeline Architecture for EEG-to-Image Visual Imagery Decoding',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color='#1F3357')
ax.text(0.5, 0.945,
        'Shared CSBrain backbone (frozen) with two branched heads: '
        'language-mediated and direct CLIP-aligned',
        ha='center', va='center', fontsize=9.5, style='italic', color='#444')


# ──────────────────────────────────────────────────────────────────────────────
# Card 1: Shared backbone (left column)
# ──────────────────────────────────────────────────────────────────────────────

S = card(ax, 0.013, 0.14, 0.245, 0.78,
         'Shared Encoder', fill=C_BG_S)

# 1) Stimulus shown to subject (real image from dataset)
stim_w, stim_h = 0.10, 0.07
stim_x = S.x + (S.w - stim_w) / 2
stim_y = S.y + S.h - 0.155
stim_box = img_from_file(
    ax, STIM_BIRD, stim_x, stim_y, stim_w, stim_h,
    border='#666',
    label='stimulus shown to subject  (class: bird)',
    lpos='below',
)

# 2) EEG waveforms recorded
wf_w, wf_h = 0.205, 0.085
wf_x = S.x + (S.w - wf_w) / 2
wf_y = stim_y - 0.135
wf_box = waveforms(ax, wf_x, wf_y, wf_w, wf_h, n_lines=6)
ax.text(wf_x + wf_w / 2, wf_y + wf_h + 0.010,
        'Multichannel EEG  (32 ch, 4 s @ 1 kHz)',
        ha='center', va='bottom', fontsize=9, color='#1F3357', fontweight='bold')
ax.text(wf_x + wf_w / 2, wf_y - 0.013,
        'preprocess + reshape  →  (32, 4, 200)',
        ha='center', va='top', fontsize=8.2, color='#444', style='italic')

# Stimulus -> EEG arrow
arrow(ax, stim_box.bottom, (wf_x + wf_w / 2, wf_y + wf_h),
      lw=1.0, mut=10, label='neural response', lfs=7.5)

# 3) CSBrain encoder
cs_w, cs_h = 0.205, 0.07
cs_x = S.x + (S.w - cs_w) / 2
cs_y = wf_y - 0.13
cs_box = encoder_stack(ax, cs_x, cs_y, cs_w, cs_h, n_layers=6)
ax.text(cs_x + cs_w / 2, cs_y + cs_h + 0.012,
        'CSBrain Encoder',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1F3357')
status_tag(ax, cs_x + cs_w - 0.06, cs_y + cs_h + 0.027, kind='frozen')

# Arrow EEG -> CSBrain
arrow(ax, (wf_x + wf_w / 2, wf_y), (cs_x + cs_w / 2, cs_y + cs_h))

ax.text(cs_x + cs_w / 2, cs_y - 0.014,
        '(32, 4, 200)  spatiotemporal features',
        ha='center', va='top', fontsize=8.2, color='#444', style='italic')

# 4) Token reducer
tr_w, tr_h = 0.175, 0.06
tr_x = S.x + (S.w - tr_w) / 2
tr_y = cs_y - 0.115
tr_box = box(ax, tr_x, tr_y, tr_w, tr_h,
             'EEG Token Reducer\n(region-mean pool, 5 regions)',
             fill=C_FROZEN, edge=C_FROZEN_E, fs=8.5)

arrow(ax, (cs_x + cs_w / 2, cs_y), (tr_x + tr_w / 2, tr_y + tr_h))

# 5) 20 EEG tokens
tk_w, tk_h = 0.18, 0.04
tk_x = S.x + (S.w - tk_w) / 2
tk_y = tr_y - 0.07
tk_box = token_strip(ax, tk_x, tk_y, tk_w, tk_h, n=10)
ax.text(tk_x + tk_w / 2, tk_y - 0.013,
        '20 EEG tokens   ∈ ℝ²⁰ˣ²⁰⁰',
        ha='center', va='top', fontsize=9, color='#1F3357', fontweight='bold')

arrow(ax, (tr_x + tr_w / 2, tr_y), (tk_x + tk_w / 2, tk_y + tk_h))


# Branching arrows out of shared backbone (start at right edge of token strip)
share_pt = (tk_x + tk_w + 0.005, tk_y + tk_h / 2)


# ──────────────────────────────────────────────────────────────────────────────
# Card 2: Pipeline 2 — EEG → Text → Image (top-right)
# ──────────────────────────────────────────────────────────────────────────────

P2 = card(ax, 0.290, 0.535, 0.696, 0.385,
          'Pipeline 2  —  Language-Mediated  (EEG → Text → Image)', fill=C_BG2)

# Layout: left to right within P2
# (A) EEG Projection MLP
proj_w, proj_h = 0.10, 0.085
proj_x = P2.x + 0.022
proj_y = P2.y + 0.13
proj_box = box(ax, proj_x, proj_y, proj_w, proj_h,
               'EEG\nProjection MLP\n200 → 2048',
               fill=C_TRAIN, edge=C_TRAIN_E, fs=8.5)
status_tag(ax, proj_x + (proj_w - 0.052) / 2, proj_y - 0.026, kind='trainable')

ax.text(proj_x + proj_w / 2, proj_y + proj_h + 0.012,
        'EEG prefix tokens   (B, 20, 2048)',
        ha='center', va='bottom', fontsize=8, color='#444', style='italic')

# (B) Prompt template box
prompt_w, prompt_h = 0.20, 0.085
prompt_x = proj_x + proj_w + 0.03
prompt_y = proj_y
prompt_box = box(ax, prompt_x, prompt_y, prompt_w, prompt_h,
                 '"Analyse this EEG and describe\n'
                 'which visual image the\nsubject is imagining."',
                 fill='white', edge='#AAAAAA', fs=8, radius=0.008)
ax.text(prompt_x + prompt_w / 2, prompt_y + prompt_h + 0.012,
        'system + user prompt   →   tokenised',
        ha='center', va='bottom', fontsize=8, color='#444', style='italic')

# (C) Concat box
concat_w, concat_h = 0.07, 0.06
concat_x = prompt_x + prompt_w + 0.025
concat_y = proj_y + (proj_h - concat_h) / 2
concat_box = box(ax, concat_x, concat_y, concat_w, concat_h,
                 'concat\n[EEG | prompt]', fill='#F0F0F0', edge='#888',
                 fs=8.2)

arrow(ax, proj_box.right,   concat_box.left, lw=1.0)
arrow(ax, prompt_box.right, concat_box.left, lw=1.0)

# (D) TinyLlama
tl_w, tl_h = 0.18, 0.085
tl_x = concat_x + concat_w + 0.03
tl_y = proj_y
tl_box = box(ax, tl_x, tl_y, tl_w, tl_h,
             'TinyLlama-1.1B\n4-bit (NF4) base + LoRA r = 8',
             fill=C_FROZEN, edge=C_FROZEN_E, fs=10, fw='bold')
status_tag(ax, tl_x + 0.02,             tl_y - 0.026, kind='frozen')
status_tag(ax, tl_x + tl_w - 0.072,     tl_y - 0.026, kind='trainable')

arrow(ax, concat_box.right, tl_box.left, lw=1.0)

# (E) Generated text bubble  (above TinyLlama)
gt_w, gt_h = 0.30, 0.045
gt_x = tl_x + (tl_w - gt_w) / 2
gt_y = tl_y + tl_h + 0.045
gt_box = box(ax, gt_x, gt_y, gt_w, gt_h,
             '"…a four-legged mammal — a dog."',
             fill='white', edge='#999', fs=9, fw='bold', radius=0.008)

arrow(ax, tl_box.top, gt_box.bottom,
      label='generated text  (greedy decode)', lfs=8)

# (F) Stable Diffusion 2.1
sd_w, sd_h = 0.205, 0.05
sd_x = tl_x + (tl_w - sd_w) / 2
sd_y = P2.y + 0.04
sd_box = box(ax, sd_x, sd_y, sd_w, sd_h,
             'Stable Diffusion 2.1   (text → image)',
             fill=C_FIXED, edge=C_FIXED_E, fs=10, fw='bold')

# Text routes down to SD: from bottom of generated-text box, down past TinyLlama,
# we draw a curved arrow on the right side
arrow(ax, gt_box.right, sd_box.right, rad=-0.55, lw=1.2,
      label='conditioning text', lfs=8, loff=(0.020, 0.0))

# (G) Output image (real)
out_w, out_h = 0.085, 0.105
out_x = sd_x + sd_w + 0.025
out_y = sd_y - 0.04
img_from_file(ax, GEN_BIRD_P2, out_x, out_y, out_w, out_h,
              border='#5C9A4A', label='generated image', lpos='above')
arrow(ax, sd_box.right, (out_x, out_y + out_h / 2), lw=1.2)


# ──────────────────────────────────────────────────────────────────────────────
# Card 3: Pipeline 3 — EEG → CLIP → Image (bottom-right)
# ──────────────────────────────────────────────────────────────────────────────

P3 = card(ax, 0.290, 0.075, 0.696, 0.435,
          'Pipeline 3  —  Direct CLIP-Aligned  (EEG → class → Image)', fill=C_BG3)

# (A) EEGCLIPMapper container with 4 internal sub-stages
mp_w, mp_h = 0.46, 0.13
mp_x = P3.x + 0.022
mp_y = P3.y + 0.21
mp_box = box(ax, mp_x, mp_y, mp_w, mp_h, '',
             fill=C_TRAIN, edge=C_TRAIN_E, lw=1.4)
ax.text(mp_x + 0.012, mp_y + mp_h - 0.018,
        'EEGCLIPMapper',
        ha='left', va='center', fontsize=10.5, fontweight='bold',
        color='#7A4515')
status_tag(ax, mp_x + mp_w - 0.064, mp_y + mp_h - 0.027, kind='trainable')

# Internal sub-stages
sub_count = 4
sub_gap = 0.005
sub_w = (mp_w - 0.024 - sub_gap * (sub_count - 1)) / sub_count
sub_h = 0.06
sub_y = mp_y + 0.025
labels = [
    'Input MLP\n200 → 512',
    '4× Transformer\nencoder layers',
    'Cross-attn\n77 learnable Q\n→ 20 KV',
    'Output proj.\n→ (B, 77, 768)\npool → (B, 768)',
]
sub_boxes = []
for i, lbl in enumerate(labels):
    sx = mp_x + 0.012 + i * (sub_w + sub_gap)
    sb = box(ax, sx, sub_y, sub_w, sub_h, lbl,
             fill='white', edge=C_TRAIN_E, fs=7.8, radius=0.006, lw=0.9)
    sub_boxes.append(sb)
    if i < sub_count - 1:
        arrow(ax, sb.right, (sx + sub_w + sub_gap, sub_y + sub_h / 2),
              lw=0.9, mut=8)

# Tensor outputs label (right of mapper)
out_label_x = mp_x + mp_w + 0.012
ax.text(out_label_x, mp_y + 0.10,
        '(B, 10) class logits',
        ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1F3357')
ax.text(out_label_x, mp_y + 0.07,
        '(B, 768) pooled embedding',
        ha='left', va='center', fontsize=8.5, color='#444')
ax.text(out_label_x, mp_y + 0.04,
        '(B, 77, 768) sequence',
        ha='left', va='center', fontsize=8.5, color='#444')

# (B) Predicted class -> SD prompt box
pc_w, pc_h = 0.34, 0.05
pc_x = mp_x + (mp_w - pc_w) / 2
pc_y = P3.y + 0.115
pc_box = box(ax, pc_x, pc_y, pc_w, pc_h,
             'predicted class  →  prompt\n("a colorful tropical bird, photorealistic, 8k …")',
             fill='white', edge='#999', fs=8.2, radius=0.008)

arrow(ax, mp_box.bottom, pc_box.top, label='argmax', lfs=8)

# (C) SD 1.5
sd3_w, sd3_h = 0.24, 0.05
sd3_x = mp_x + (mp_w - sd3_w) / 2
sd3_y = P3.y + 0.04
sd3_box = box(ax, sd3_x, sd3_y, sd3_w, sd3_h,
              'Stable Diffusion 1.5   (text → image)',
              fill=C_FIXED, edge=C_FIXED_E, fs=10, fw='bold')

arrow(ax, pc_box.bottom, sd3_box.top)

# (D) Output image (real)
out3_w, out3_h = 0.085, 0.115
out3_x = sd3_x + sd3_w + 0.025
out3_y = sd3_y - 0.025
img_from_file(ax, GEN_BIRD_P3, out3_x, out3_y, out3_w, out3_h,
              border='#5C9A4A', label='generated image', lpos='above')
arrow(ax, sd3_box.right, (out3_x, out3_y + out3_h / 2), lw=1.2)

# ── Training-time supervision panel (right side of Pipeline 3 card) ──────────
TX = P3.x + P3.w - 0.18
TY = P3.y + 0.045
TW = 0.158

ax.text(TX + TW / 2, P3.y + P3.h - 0.05,
        'Training-time supervision',
        ha='center', va='center', fontsize=10, fontweight='bold',
        color='#7A1818', style='italic')

# Stimulus image (real, dog as second example)
si_w, si_h = 0.06, 0.05
si_x = TX + (TW - si_w) / 2
si_y = P3.y + P3.h - 0.13
img_from_file(ax, STIM_DOG, si_x, si_y, si_w, si_h,
              border='#888', label='10 stimulus images', lpos='above')

# CLIP image encoder
cie_y = si_y - 0.08
cie_box = box(ax, TX, cie_y, TW, 0.05,
              'CLIP ViT-L/14 image encoder',
              fill=C_FIXED, edge=C_FIXED_E, fs=8.5)
arrow(ax, (si_x + si_w / 2, si_y), cie_box.top, lw=1.0)

# CLIP text encoder
cte_y = cie_y - 0.075
cte_box = box(ax, TX, cte_y, TW, 0.05,
              'CLIP text encoder\n(aMUSEd-aligned)',
              fill=C_FIXED, edge=C_FIXED_E, fs=8.5)

# Joint loss
jl_y = cte_y - 0.10
jl_box = box(ax, TX, jl_y, TW, 0.085,
             'Joint loss\nL = λ_cls·CE  +  λ_cont·InfoNCE\n+ λ_cos·cos  +  λ_mse·MSE',
             fill='#FBE9E9', edge=C_LOSS, fs=8)

arrow(ax, cie_box.bottom, jl_box.top, color=C_LOSS, ls='--', lw=1.0)
arrow(ax, cte_box.bottom, jl_box.top, color=C_LOSS, ls='--', lw=1.0, rad=0.15)

# Predictions feeding into loss (dashed red arrow from mapper outputs label area)
pred_pt = (out_label_x + 0.06, mp_y + 0.03)
ax.text(pred_pt[0] + 0.005, pred_pt[1] + 0.005, 'preds',
        ha='left', va='bottom', fontsize=7.5, color=C_LOSS, style='italic')
arrow(ax, pred_pt, (TX + TW / 2, jl_y + 0.085 / 2),
      color=C_LOSS, ls='--', lw=1.0, rad=-0.20, lbg=False)


# ──────────────────────────────────────────────────────────────────────────────
# Branching arrows: shared 20-token output → both pipelines
# ──────────────────────────────────────────────────────────────────────────────

# To Pipeline 2 (EEG Projection MLP top-left input)
arrow(ax, share_pt, proj_box.left, lw=1.5,
      label='shared EEG tokens  (20 × 200)', lfs=8.5,
      loff=(0.0, 0.012))

# To Pipeline 3 (sub_boxes[0] left)
arrow(ax, share_pt, sub_boxes[0].left, lw=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Legend (bottom)
# ──────────────────────────────────────────────────────────────────────────────

leg_y = 0.018
legend_items = [
    ('Frozen / pretrained',        C_FROZEN, C_FROZEN_E),
    ('Trainable',                  C_TRAIN,  C_TRAIN_E),
    ('Fixed external model',       C_FIXED,  C_FIXED_E),
]
lx = 0.10
for label, fill, edge in legend_items:
    p = FancyBboxPatch((lx, leg_y), 0.022, 0.022,
                       boxstyle="round,pad=0.0,rounding_size=0.005",
                       linewidth=1.0, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(lx + 0.028, leg_y + 0.011, label,
            ha='left', va='center', fontsize=9, color='#222')
    lx += 0.20

ax.plot([lx + 0.005, lx + 0.04], [leg_y + 0.011, leg_y + 0.011],
        color=C_LOSS, ls='--', lw=1.0)
ax.text(lx + 0.05, leg_y + 0.011, 'training-only loss path',
        ha='left', va='center', fontsize=9, color='#222')


# ─── Save ────────────────────────────────────────────────────────────────────

out_dir = os.path.join(PROJ, 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'method_diagram.png')
out_pdf = os.path.join(out_dir, 'method_diagram.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
plt.savefig(out_pdf,            bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
