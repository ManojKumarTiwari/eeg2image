"""
image_generator.py — Text-to-Image generation using Stable Diffusion 2.1.

Converts EEG-decoded text descriptions into visual images using
stabilityai/stable-diffusion-2-1 (Apache 2.0 license, fully open source).

Pipeline:
    EEG signal
        → CSBrain encoder  (frozen)
        → EEGProjection    (trained)
        → TinyLlama LLM    (LoRA fine-tuned)
        → neuroscience text description
        → EEGImageGenerator (this module)
        → PNG image
"""

import os
import re
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# ─── Motor imagery visual prompts (BCIC-IV-2a, 4 classes) ────────────────────

_MI_VISUAL_PROMPTS = {
    0: (
        "a person reaching and grasping with their left hand, left arm extended forward, "
        "focused intentional hand movement, motor activity, clean studio background, "
        "photorealistic, sharp focus, 8k resolution"
    ),
    1: (
        "a person reaching and grasping with their right hand, right arm extended forward, "
        "focused intentional hand movement, motor activity, clean studio background, "
        "photorealistic, sharp focus, 8k resolution"
    ),
    2: (
        "a person performing a kicking or stepping motion with both feet, "
        "lower limb motor activity, dynamic leg movement pose, clean studio background, "
        "photorealistic, sharp focus, 8k resolution"
    ),
    3: (
        "a close-up of a person moving their tongue, orofacial motor activity, "
        "detailed facial muscles, medical illustration style, "
        "photorealistic, sharp focus, 8k resolution"
    ),
}

# ─── Emotion visual prompts (FACED, 9 classes) ───────────────────────────────

_EMOTION_VISUAL_PROMPTS = {
    0: (
        "a person laughing with amusement, joyful facial expression, bright eyes, "
        "natural smile, warm lighting, detailed portrait, photorealistic, sharp focus"
    ),
    1: (
        "a person with an inspired and thoughtful expression, sense of wonder, "
        "uplifted face, contemplative look, warm lighting, detailed portrait, photorealistic"
    ),
    2: (
        "a person expressing pure joy and elation, radiant smile, happy eyes, "
        "euphoric expression, golden lighting, detailed portrait, photorealistic"
    ),
    3: (
        "a person showing tenderness and gentle affection, soft caring eyes, "
        "warm gentle expression, soft lighting, detailed portrait, photorealistic"
    ),
    4: (
        "a person expressing anger, furrowed brows, tense jaw, frustrated expression, "
        "intense gaze, dramatic lighting, detailed portrait, photorealistic"
    ),
    5: (
        "a person showing disgust, wrinkling nose, grimacing expression, "
        "repelled look, cool lighting, detailed portrait, photorealistic"
    ),
    6: (
        "a person experiencing fear, wide alarmed eyes, startled expression, "
        "tense facial muscles, dramatic shadow lighting, detailed portrait, photorealistic"
    ),
    7: (
        "a person feeling deep sadness, downcast eyes, melancholic expression, "
        "subtle tears, soft diffused lighting, detailed portrait, photorealistic"
    ),
    8: (
        "a person with a calm neutral expression, relaxed face, no strong emotion, "
        "balanced soft lighting, detailed portrait, photorealistic"
    ),
}

_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs, "
    "watermark, text, logo, oversaturated, cartoon, anime, sketch"
)


class EEGImageGenerator:
    """
    Generates images from EEG-decoded text descriptions using Stable Diffusion 2.1.

    Model: stabilityai/stable-diffusion-2-1
    License: Apache 2.0 (open source, no commercial restrictions)
    Source: https://huggingface.co/stabilityai/stable-diffusion-2-1

    The generator converts neuroscience text descriptions produced by the
    EEG-LLM into structured visual prompts, then synthesises corresponding
    images via a DPMSolver-accelerated diffusion pipeline.

    Args:
        model_id:                 HuggingFace model ID (default: SD 2.1)
        device:                   Torch device string, e.g. "cuda" or "cpu"
        use_half_precision:       Use float16 for reduced VRAM (recommended)
        enable_attention_slicing: Trade a small speed cost for lower peak VRAM
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda",
        use_half_precision: bool = True,
        enable_attention_slicing: bool = True,
    ):
        dtype = torch.float16 if use_half_precision else torch.float32
        print(f"Loading image generation model: {model_id} ...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        # DPMSolverMultistepScheduler: ~25 steps gives quality comparable to
        # DDPM at 1000 steps, cutting inference time by ~40×
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(device)
        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        self.device = device
        print(f"Image generator ready on {device}.")

    # ─── Prompt construction ──────────────────────────────────────────────────

    @staticmethod
    def build_prompt(
        eeg_text: str,
        label: int = None,
        dataset: str = "BCICIV2A",
    ) -> str:
        """
        Build a structured visual prompt from an EEG-decoded text description.

        Prefers the class label (high information, short) over free-form text
        extraction when available. Falls back to keyword extraction otherwise.

        Args:
            eeg_text: Text generated by the EEG-LLM for one sample
            label:    Ground-truth or predicted class index (None = unknown)
            dataset:  'BCICIV2A' (motor imagery) or 'FACED' (emotion)

        Returns:
            A visual text prompt suitable for Stable Diffusion
        """
        if dataset.upper() == "BCICIV2A":
            visual_map = _MI_VISUAL_PROMPTS
        else:
            visual_map = _EMOTION_VISUAL_PROMPTS

        if label is not None and label in visual_map:
            return visual_map[label]

        # Fallback: extract domain terms from the generated description
        terms = _extract_key_terms(eeg_text)
        if terms:
            return (
                f"neural motor activity visualization, {', '.join(terms)}, "
                "medical illustration, photorealistic, high detail"
            )
        return (
            "brain neural activity visualization, motor cortex, "
            "medical illustration, photorealistic, high detail"
        )

    # ─── Core generation ─────────────────────────────────────────────────────

    def generate(
        self,
        prompts: list,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = 42,
    ) -> list:
        """
        Generate one image per prompt.

        Args:
            prompts:              List of text prompts (one per sample)
            num_inference_steps:  Denoising steps (25 is fast; 50 is higher quality)
            guidance_scale:       CFG scale — higher = closer to prompt (7.5 default)
            height, width:        Output resolution in pixels (multiple of 8)
            seed:                 RNG seed for reproducibility

        Returns:
            List of PIL.Image.Image objects, one per prompt
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompts,
            negative_prompt=[_NEGATIVE_PROMPT] * len(prompts),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        return result.images

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_images(
        self,
        images: list,
        output_dir: str,
        prefix: str = "eeg2image",
        true_labels: list = None,
        prompts: list = None,
    ) -> list:
        """
        Save generated images to *output_dir*.

        Filenames: ``{prefix}_{index:04d}_label{label}.png``

        Optionally writes a ``prompts.txt`` sidecar listing every prompt so
        the generation is reproducible.

        Returns:
            List of absolute paths to saved PNG files
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for i, img in enumerate(images):
            label_str = f"_label{true_labels[i]}" if true_labels is not None else ""
            path = os.path.join(output_dir, f"{prefix}_{i:04d}{label_str}.png")
            img.save(path)
            saved.append(path)
            print(f"  Saved: {path}")

        if prompts is not None:
            prompt_log = os.path.join(output_dir, "prompts.txt")
            with open(prompt_log, "w") as f:
                for i, p in enumerate(prompts):
                    label_str = f"[label={true_labels[i]}]" if true_labels is not None else ""
                    f.write(f"[{i:04d}] {label_str} {p}\n")
            print(f"  Prompt log: {prompt_log}")

        return saved


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_key_terms(text: str) -> list:
    """
    Extract visually meaningful terms from a neuroscience text description.
    Returns up to 5 unique terms in order of appearance.
    """
    patterns = [
        r"\b(left hand|right hand|both feet|tongue)\b",
        r"\b(motor imagery|motor cortex|sensorimotor|hand movement|foot movement)\b",
        r"\b(mu rhythm|beta band|alpha band|ERD|ERS|theta)\b",
        r"\b(frontal|central|parietal|occipital|temporal lobe)\b",
        r"\b(contralateral|ipsilateral|bilateral)\b",
    ]
    terms = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text.lower()))
    # Preserve order, deduplicate, cap at 5
    seen = set()
    result = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) == 5:
            break
    return result
