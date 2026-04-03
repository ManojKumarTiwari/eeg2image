"""
eeg_clip_mapper.py — EEG token features -> CLIP-compatible conditioning.

Maps EEG token features (B, 20, 200) produced by CSBrain + EEGTokenReducer
into the CLIP text embedding space expected by aMUSEd-512:
  encoder_hidden_states : (B, 77, 768)  — per-token sequence (cross-attention)
  prompt_embeds         : (B, 768)      — pooled conditioning vector

Architecture:
  1. Input MLP projection: (B, 20, 200) -> (B, 20, 512)
  2. Transformer encoder (self-attention over 20 EEG tokens)
  3. Query expansion via cross-attention: 20 EEG tokens -> 77 output tokens
  4. Output projections to CLIP dim (768)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGCLIPMapper(nn.Module):
    """
    Maps EEG token features to CLIP text embedding space for aMUSEd-512 conditioning.

    Args:
        eeg_dim:              Input feature dim from CSBrain (default 200)
        n_eeg_tokens:         Number of EEG tokens from TokenReducer (default 20)
        clip_seq_len:         CLIP text sequence length — must match aMUSEd (77)
        clip_dim:             CLIP embedding dim — must match aMUSEd text encoder (768)
        mapper_dim:           Internal transformer hidden size
        n_transformer_layers: Depth of EEG self-attention transformer
        n_heads:              Attention heads (mapper_dim must be divisible by n_heads)
        dropout:              Dropout rate
    """

    def __init__(
        self,
        eeg_dim: int = 200,
        n_eeg_tokens: int = 20,
        clip_seq_len: int = 77,
        clip_dim: int = 768,
        mapper_dim: int = 512,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_classes: int = 10,
    ):
        super().__init__()

        self.clip_seq_len = clip_seq_len
        self.clip_dim = clip_dim

        # ── Stage 1: Input projection MLP ───────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(eeg_dim, mapper_dim),
            nn.LayerNorm(mapper_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mapper_dim, mapper_dim),
        )

        # ── Stage 2: Transformer encoder (self-attention over 20 EEG tokens) ─
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mapper_dim,
            nhead=n_heads,
            dim_feedforward=mapper_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # ── Stage 3: Learnable query expansion (20 tokens -> 77 tokens) ─────
        # Q-Former / Perceiver Resampler pattern: 77 learnable queries attend
        # to the 20 compressed EEG tokens to produce 77 output tokens.
        self.expand_queries = nn.Parameter(
            torch.randn(clip_seq_len, mapper_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=mapper_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(mapper_dim)

        # ── Stage 4: Output projections to CLIP space ────────────────────────
        self.seq_proj = nn.Sequential(
            nn.Linear(mapper_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.Tanh(),
        )

        # ── Auxiliary classification head (training only) ─────────────────
        # Direct cross-entropy supervision — provides the strongest
        # discriminative signal; discarded at inference time.
        self.classifier = nn.Linear(clip_dim, n_classes)

    def forward(self, eeg_tokens: torch.Tensor):
        """
        Args:
            eeg_tokens: (B, 20, 200) — output of EEGTokenReducer

        Returns:
            encoder_hidden_states: (B, 77, 768)
            prompt_embeds:         (B, 768)
        """
        B = eeg_tokens.shape[0]

        # Stage 1: project each EEG token
        x = self.input_proj(eeg_tokens)                              # (B, 20, 512)

        # Stage 2: model inter-token relationships
        x = self.transformer(x)                                      # (B, 20, 512)

        # Stage 3: expand 20 -> 77 via cross-attention
        queries = self.expand_queries.unsqueeze(0).expand(B, -1, -1) # (B, 77, 512)
        attended, _ = self.cross_attn(queries, x, x)                 # (B, 77, 512)
        attended = self.cross_attn_norm(attended + queries)           # residual

        # Stage 4: project to CLIP dim
        encoder_hidden_states = self.seq_proj(attended)              # (B, 77, 768)
        pooled = encoder_hidden_states.mean(dim=1)                   # (B, 768)
        prompt_embeds = self.pool_proj(pooled)                       # (B, 768)
        class_logits = self.classifier(pooled)                       # (B, n_classes)

        return encoder_hidden_states, prompt_embeds, class_logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CLIPImageTargetBuilder:
    """
    Precomputes CLIP image embeddings for all 10 VI stimulus images.

    Uses openai/clip-vit-large-patch14 (cached). Image embeddings are much
    more discriminative than text embeddings for the 10 classes
    (mean pairwise cosine sim ~0.54 vs ~0.90 for text), giving stronger
    gradient signal for contrastive training.

    Returns:
        targets_pooled: (n_classes, 768)  — projected image embeddings
    """

    # Class index -> stimulus filename (alphabetical sort of stimuli dir)
    VI_STIMULUS_FILES = [
        'Animal_dog.jpg',     # 0 dog
        'Animal_bird.jpg',    # 1 bird
        'Animal_fish.jpg',    # 2 fish
        'Figure_pentagram.jpg', # 3 pentagram
        'Figure_square.jpg',  # 4 square
        'Figure_circle.jpg',  # 5 circle
        'Object_scissor.jpg', # 6 scissor
        'Object_watch.jpg',   # 7 watch
        'Object_cup.jpg',     # 8 cup
        'Object_chair.jpg',   # 9 chair
    ]

    def __init__(self, stimuli_dir: str, clip_model_id: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda"):
        self.stimuli_dir = stimuli_dir
        self.clip_model_id = clip_model_id
        self.device = device

    @torch.no_grad()
    def build(self):
        """Encode all 10 stimulus images with CLIP ViT-L/14."""
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        from PIL import Image

        print(f"Loading CLIP vision model from {self.clip_model_id}...")
        model = CLIPVisionModelWithProjection.from_pretrained(
            self.clip_model_id, torch_dtype=torch.float32,
        ).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(self.clip_model_id)

        images = []
        for fname in self.VI_STIMULUS_FILES:
            path = os.path.join(self.stimuli_dir, fname)
            images.append(Image.open(path).convert('RGB'))

        inputs = processor(images=images, return_tensors='pt').to(self.device)
        outputs = model(**inputs)
        targets_pooled = outputs.image_embeds.cpu()    # (10, 768)

        del model
        torch.cuda.empty_cache()
        print(f"Built CLIP image targets: pooled={tuple(targets_pooled.shape)}")
        return targets_pooled


class CLIPTextTargetBuilder:
    """
    Precomputes frozen CLIP text conditioning targets for all VI classes.

    Uses the CLIP text encoder bundled with aMUSEd-512 (amused/amused-512)
    so the embedding space is exactly aligned with the image generation model.

    Returns:
        targets_hidden: (n_classes, 77, 768) — penultimate hidden states
        targets_pooled: (n_classes, 768)     — pooled text embeds
    """

    # Class-name prompts — simple, unambiguous descriptions
    VI_CLASS_PROMPTS = [
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a fish",
        "a photo of a pentagram star shape",
        "a photo of a square shape",
        "a photo of a circle shape",
        "a photo of scissors",
        "a photo of a wristwatch",
        "a photo of a cup",
        "a photo of a chair",
    ]

    def __init__(self, amused_model_id: str = "amused/amused-512", device: str = "cuda"):
        self.model_id = amused_model_id
        self.device = device

    @torch.no_grad()
    def build(self):
        """Load CLIP text encoder from aMUSEd and encode all 10 class prompts."""
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        print(f"Loading CLIP text encoder from {self.model_id}...")
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="text_encoder")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        ).to(self.device).eval()

        inputs = tokenizer(
            self.VI_CLASS_PROMPTS,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(self.device)

        outputs = text_encoder(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        targets_pooled = outputs.text_embeds.cpu()          # (10, 768)
        targets_hidden = outputs.hidden_states[-2].cpu()    # (10, 77, 768)

        # Free the text encoder — not needed after this
        del text_encoder
        torch.cuda.empty_cache()

        print(f"Built CLIP targets: hidden={tuple(targets_hidden.shape)}, pooled={tuple(targets_pooled.shape)}")
        return targets_hidden, targets_pooled

    @torch.no_grad()
    def build_null_conditioning(self):
        """Build empty-string conditioning for classifier-free guidance."""
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="text_encoder")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        ).to(self.device).eval()

        inputs = tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(self.device)

        outputs = text_encoder(**inputs, return_dict=True, output_hidden_states=True)
        null_pooled  = outputs.text_embeds.cpu()         # (1, 768)
        null_hidden  = outputs.hidden_states[-2].cpu()   # (1, 77, 768)

        del text_encoder
        torch.cuda.empty_cache()

        return null_hidden, null_pooled
