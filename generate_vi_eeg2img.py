"""
generate_vi_eeg2img.py — Direct EEG -> Image pipeline.

Stage 1: EEG -> CSBrain -> EEGCLIPMapper -> classifier -> predicted class label
Stage 2: predicted label -> SD 1.5 text prompt -> 512x512 image

Using SD 1.5 (runwayml/stable-diffusion-v1-5, ~4GB VRAM) for high-quality
photorealistic generation. EEG is used purely for class prediction; text prompts
ensure visually coherent output regardless of EEG conditioning noise.

Usage:
    python generate_vi_eeg2img.py --num_samples 20
    python generate_vi_eeg2img.py \\
        --mapper_path pth_downtasks/eeg_direct/mapper_epoch17.pth \\
        --num_samples 20 --output_dir outputs/vi_eeg2img
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from models.eeg_llm import (
    EEGTokenReducer, VI_BRAIN_REGIONS, VI_ELECTRODE_LABELS,
    VI_TOPOLOGY, _build_sorted_indices,
)
from models.eeg_clip_mapper import EEGCLIPMapper
from models.CSBrain import CSBrain
from datasets.visual_imagery_llm_dataset import VisualImageryLLMDataset
import lmdb


VI_CLASS_NAMES = [
    'dog', 'bird', 'fish',
    'pentagram', 'square', 'circle',
    'scissor', 'watch', 'cup', 'chair',
]
VI_STIMULUS_FILES = {
    0: 'Animal_dog.jpg',
    1: 'Animal_bird.jpg',
    2: 'Animal_fish.jpg',
    3: 'Figure_pentagram.jpg',
    4: 'Figure_square.jpg',
    5: 'Figure_circle.jpg',
    6: 'Object_scissor.jpg',
    7: 'Object_watch.jpg',
    8: 'Object_cup.jpg',
    9: 'Object_chair.jpg',
}

# High-quality SD 1.5 prompts — photorealistic, white background
VI_SD_PROMPTS = {
    0: ("a golden retriever dog, full body, studio white background, "
        "photorealistic, 8k, sharp focus, professional photography"),
    1: ("a colorful tropical bird perched on a branch, studio white background, "
        "photorealistic, 8k, sharp focus, professional photography"),
    2: ("a tropical fish swimming, studio white background, "
        "photorealistic, 8k, sharp focus, professional photography"),
    3: ("a pentagram star shape drawn with thick black lines, "
        "clean white background, vector art, sharp, high contrast"),
    4: ("a perfect square drawn with thick black lines, "
        "clean white background, vector art, sharp, high contrast"),
    5: ("a perfect circle drawn with thick black lines, "
        "clean white background, vector art, sharp, high contrast"),
    6: ("a pair of scissors, product photography, studio white background, "
        "photorealistic, 8k, sharp focus"),
    7: ("a wristwatch with metal band, product photography, studio white background, "
        "photorealistic, 8k, sharp focus"),
    8: ("a white ceramic coffee cup, product photography, studio white background, "
        "photorealistic, 8k, sharp focus"),
    9: ("a wooden dining chair, product photography, studio white background, "
        "photorealistic, 8k, sharp focus"),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, logo, oversaturated, cartoon, anime, sketch, noisy"
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_comparison_image(stimulus_path, gen_img, true_label, pred_label, idx, size=512):
    """Side-by-side: [original stimulus | EEG-predicted generated image]."""
    if stimulus_path and os.path.exists(stimulus_path):
        stimulus = Image.open(stimulus_path).convert('RGB').resize((size, size))
    else:
        stimulus = Image.new('RGB', (size, size), color=(200, 200, 200))

    gen = gen_img.resize((size, size))

    caption_h = 40
    canvas = Image.new('RGB', (size * 2, size + caption_h), color=(255, 255, 255))
    canvas.paste(stimulus, (0, 0))
    canvas.paste(gen, (size, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, size, 30], fill=(50, 50, 50))
    draw.text((10, 8), f"Original: {VI_CLASS_NAMES[true_label]}", fill='white')

    correct = pred_label == true_label
    header_color = (50, 150, 50) if correct else (180, 50, 50)
    draw.rectangle([size, 0, size * 2, 30], fill=header_color)
    pred_name = VI_CLASS_NAMES[pred_label] if pred_label >= 0 else 'unknown'
    draw.text((size + 10, 8), f"EEG decoded: {pred_name} {'OK' if correct else 'X'}", fill='white')
    draw.text((10, size + 8), f"[{idx}] EEG->class->SD1.5 generation", fill=(60, 60, 60))
    return canvas


def build_eeg_encoder(args, device):
    sorted_indices = _build_sorted_indices(
        VI_BRAIN_REGIONS, VI_ELECTRODE_LABELS, VI_TOPOLOGY
    )
    encoder = CSBrain(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30,
        n_layer=args.n_layer, nhead=8,
        brain_regions=VI_BRAIN_REGIONS,
        sorted_indices=sorted_indices,
    )
    if args.use_pretrained_weights:
        state_dict = torch.load(args.foundation_dir, map_location=device, weights_only=False)
        new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_sd = encoder.state_dict()
        matching = {k: v for k, v in new_sd.items()
                    if k in model_sd and v.size() == model_sd[k].size()}
        model_sd.update(matching)
        encoder.load_state_dict(model_sd)
        print(f"Loaded {len(matching)}/{len(model_sd)} pretrained weights into CSBrain")

    encoder.proj_out = nn.Identity()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir',    type=str, default='data/VisualImagery/processed_lmdb')
    parser.add_argument('--mapper_path',     type=str, default='pth_downtasks/eeg_direct/mapper_epoch17.pth')
    parser.add_argument('--foundation_dir',  type=str, default='pth/CSBrain.pth')
    parser.add_argument('--stimuli_dir',     type=str, default='data/VisualImagery/stimuli')
    parser.add_argument('--output_dir',      type=str, default='outputs/vi_eeg2img')
    parser.add_argument('--image_model',     type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--num_samples',     type=int, default=20)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--guidance_scale',  type=float, default=7.5)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--cuda',            type=int, default=0)
    parser.add_argument('--n_layer',         type=int, default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    parser.add_argument('--mapper_dim',      type=int, default=512)
    parser.add_argument('--n_transformer_layers', type=int, default=4)
    parser.add_argument('--n_heads',         type=int, default=8)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Stage 1: EEG -> predicted class label ────────────────────────────────
    print("=" * 60)
    print("Stage 1: EEG -> EEGCLIPMapper classifier -> predicted class")
    print("=" * 60)

    encoder = build_eeg_encoder(args, device)

    token_reducer_tmp = EEGCLIPMapper(  # load to get area_config
        eeg_dim=200, n_eeg_tokens=20, clip_seq_len=77, clip_dim=768,
        mapper_dim=args.mapper_dim, n_transformer_layers=args.n_transformer_layers,
        n_heads=args.n_heads, dropout=0.0,
    )

    # Rebuild token_reducer using encoder's area_config
    token_reducer = EEGTokenReducer(
        area_config=encoder.area_config, temporal_pool_stride=1,
    ).to(device)

    mapper = EEGCLIPMapper(
        eeg_dim=200, n_eeg_tokens=20, clip_seq_len=77, clip_dim=768,
        mapper_dim=args.mapper_dim, n_transformer_layers=args.n_transformer_layers,
        n_heads=args.n_heads, dropout=0.0,
    ).to(device).eval()

    ckpt = torch.load(args.mapper_path, map_location=device, weights_only=False)
    mapper.load_state_dict(ckpt['mapper'])
    token_reducer.load_state_dict(ckpt['token_reducer'])
    print(f"Loaded mapper (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f})")

    # Load dataset
    shared_db = lmdb.open(
        args.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False
    )
    test_set = VisualImageryLLMDataset(args.datasets_dir, mode='test', db=shared_db)

    from torch.utils.data import DataLoader

    def collate(batch):
        eeg = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {'eeg_data': eeg, 'label_ids': labels}

    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             collate_fn=collate, shuffle=False)

    print(f"\nRunning EEG -> classifier for {args.num_samples} samples...")
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="EEG->class"):
            if len(results) >= args.num_samples:
                break
            eeg_data  = batch['eeg_data'].to(device)
            label_ids = batch['label_ids']

            with torch.amp.autocast('cuda', enabled=False):
                features = encoder(eeg_data[:, :32, :, :].float())
            eeg_tokens = token_reducer(features)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, _, class_logits = mapper(eeg_tokens.half())

            preds = class_logits.float().argmax(dim=-1).cpu()

            for j in range(eeg_data.shape[0]):
                if len(results) >= args.num_samples:
                    break
                results.append({
                    'true_label': int(label_ids[j]),
                    'pred_label': int(preds[j]),
                })
            torch.cuda.empty_cache()

    correct = sum(1 for r in results if r['pred_label'] == r['true_label'])
    print(f"EEG classification accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    # Free EEG models before loading SD
    del encoder, token_reducer, mapper
    torch.cuda.empty_cache()

    # ── Stage 2: predicted label -> SD 1.5 -> high-quality image ─────────────
    print("\n" + "=" * 60)
    print(f"Stage 2: Generating images with SD 1.5 (predicted class prompts)")
    print("=" * 60)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.image_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = []

    print(f"\nGenerating {len(results)} images...")
    for r in tqdm(results, desc="Generating"):
        pred_label = r['pred_label']
        prompt = VI_SD_PROMPTS.get(pred_label, f"a {VI_CLASS_NAMES[pred_label]}, photorealistic")

        img = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=512, width=512,
            generator=generator,
        ).images[0]
        images.append(img)
        torch.cuda.empty_cache()

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving images...")
    comparison_imgs = []
    for i, (r, gen_img) in enumerate(zip(results, images)):
        true_label = r['true_label']
        pred_label = r['pred_label']

        gen_path = os.path.join(args.output_dir,
                                f'generated_{i:04d}_true{true_label}_pred{pred_label}.png')
        gen_img.save(gen_path)

        stimulus_path = os.path.join(
            args.stimuli_dir, VI_STIMULUS_FILES.get(true_label, '')
        )
        comp = make_comparison_image(stimulus_path, gen_img, true_label, pred_label, i)
        comp_path = os.path.join(args.output_dir,
                                 f'comparison_{i:04d}_true{true_label}_pred{pred_label}.png')
        comp.save(comp_path)
        comparison_imgs.append(comp)

        status = 'CORRECT' if pred_label == true_label else 'WRONG'
        print(f"  [{i:03d}] true={VI_CLASS_NAMES[true_label]:10s} "
              f"pred={VI_CLASS_NAMES[pred_label]:10s} {status}")

    # Summary grid
    cols = 4
    rows = (len(comparison_imgs) + cols - 1) // cols
    thumb_w, thumb_h = 512, 276
    grid = Image.new('RGB', (thumb_w * cols, thumb_h * rows), color=(240, 240, 240))
    for i, img in enumerate(comparison_imgs):
        thumb = img.resize((thumb_w, thumb_h))
        row, col = divmod(i, cols)
        grid.paste(thumb, (col * thumb_w, row * thumb_h))

    grid_path = os.path.join(args.output_dir, 'summary_grid.png')
    grid.save(grid_path)

    print(f"\nDone! {len(images)} images saved to: {args.output_dir}")
    print(f"Summary grid: {grid_path}")
    print(f"EEG accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()
