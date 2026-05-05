"""
generate_summary_grids.py — Generate clean summary grids for Pipeline 2 and Pipeline 3.

For each of the 10 classes, shows the target stimulus image alongside a
generated image produced by the correct class prompt. Reuses existing generated
images (matched by predicted class). If a class is missing, generates it with
the diffusion model; if that's not possible (low RAM), uses a placeholder.

Usage:
    python generate_summary_grids.py
    python generate_summary_grids.py --pipeline 2
    python generate_summary_grids.py --pipeline 3
    python generate_summary_grids.py --pipeline both
"""

import argparse
import gc
import glob
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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

# Pipeline 2 prompts (aMUSEd-512) — from generate_vi.py
AMUSED_PROMPTS = {
    0: "a golden retriever dog, white background, photorealistic",
    1: "a colorful bird perched on a branch, white background, photorealistic",
    2: "a tropical fish swimming, white background, photorealistic",
    3: "a pentagram star shape, black lines on white background",
    4: "a square shape, black lines on white background",
    5: "a circle shape, black lines on white background",
    6: "a pair of scissors, product photo, white background",
    7: "a wristwatch, product photo, white background",
    8: "a ceramic coffee cup, product photo, white background",
    9: "a wooden chair, product photo, white background",
}

# Pipeline 3 prompts (SD 1.5) — from generate_vi_eeg2img.py
SD_PROMPTS = {
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

SD_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, logo, oversaturated, cartoon, anime, sketch, noisy"
)


def try_load_font(size=16):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def find_existing_images(output_dir):
    """Find existing generated images, indexed by predicted class."""
    images_by_pred = {}
    pattern = os.path.join(output_dir, 'generated_*_pred*.png')
    for fpath in glob.glob(pattern):
        m = re.search(r'pred(\d+)', os.path.basename(fpath))
        if m:
            pred_class = int(m.group(1))
            if pred_class not in images_by_pred:
                images_by_pred[pred_class] = fpath
    # Also check for generated_correct_classN.png files
    for class_idx in range(10):
        if class_idx not in images_by_pred:
            p = os.path.join(output_dir, f'generated_correct_class{class_idx}.png')
            if os.path.exists(p):
                images_by_pred[class_idx] = p
    return images_by_pred


def make_placeholder(class_idx, size=512):
    """Create a placeholder image for a missing class."""
    img = Image.new('RGB', (size, size), color=(230, 230, 240))
    draw = ImageDraw.Draw(img)
    font = try_load_font(20)
    font_small = try_load_font(14)
    name = VI_CLASS_NAMES[class_idx]
    draw.text((size // 2 - 60, size // 2 - 30), f"[{name}]", fill=(120, 120, 120), font=font)
    draw.text((size // 2 - 80, size // 2 + 10), "Not yet generated", fill=(160, 160, 160), font=font_small)
    draw.text((size // 2 - 80, size // 2 + 35), "(free RAM to generate)", fill=(180, 180, 180), font=font_small)
    return img


def create_summary_grid(stimulus_dir, generated_images, pipeline_name, output_path, img_size=256):
    """
    Create a 10-row x 2-column grid: [Target Stimulus | Generated Image].
    """
    font_title = try_load_font(22)
    font_label = try_load_font(15)
    font_header = try_load_font(17)

    col_w = img_size
    row_h = img_size
    header_h = 40
    label_h = 28
    title_h = 45
    padding = 8

    grid_w = padding + col_w + padding + col_w + padding
    grid_h = title_h + header_h + 10 * (row_h + label_h) + padding

    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((padding + 10, 10), pipeline_name, fill=(30, 30, 30), font=font_title)

    # Column headers
    y_header = title_h
    x_stim = padding
    x_gen = padding + col_w + padding

    draw.rectangle([x_stim, y_header, x_stim + col_w, y_header + header_h], fill=(60, 60, 60))
    draw.text((x_stim + col_w // 2 - 55, y_header + 10), "Target Stimulus", fill='white', font=font_header)

    draw.rectangle([x_gen, y_header, x_gen + col_w, y_header + header_h], fill=(40, 100, 60))
    draw.text((x_gen + col_w // 2 - 55, y_header + 10), "Generated Image", fill='white', font=font_header)

    y = title_h + header_h

    for class_idx in range(10):
        # Stimulus
        stim_file = VI_STIMULUS_FILES[class_idx]
        stim_path = os.path.join(stimulus_dir, stim_file)
        if os.path.exists(stim_path):
            stim_img = Image.open(stim_path).convert('RGB').resize((col_w, row_h))
        else:
            stim_img = Image.new('RGB', (col_w, row_h), color=(200, 200, 200))

        # Generated
        gen_img = generated_images[class_idx].resize((col_w, row_h))

        grid.paste(stim_img, (x_stim, y))
        grid.paste(gen_img, (x_gen, y))

        # Label
        label_y = y + row_h + 3
        draw.text((x_stim + 5, label_y), f"{class_idx}: {VI_CLASS_NAMES[class_idx]}",
                   fill=(60, 60, 60), font=font_label)

        y += row_h + label_h

    grid.save(output_path, quality=95)
    print(f"Saved summary grid: {output_path}")
    return grid


def load_images_for_pipeline(output_dir):
    """Load existing generated images, use placeholders for missing."""
    existing = find_existing_images(output_dir)
    images = {}

    for class_idx in range(10):
        if class_idx in existing:
            images[class_idx] = Image.open(existing[class_idx]).convert('RGB')
            print(f"  class {class_idx} ({VI_CLASS_NAMES[class_idx]:10s}) -> {os.path.basename(existing[class_idx])}")
        else:
            images[class_idx] = make_placeholder(class_idx)
            print(f"  class {class_idx} ({VI_CLASS_NAMES[class_idx]:10s}) -> [placeholder - no image available]")

    return images


def try_generate_missing(output_dir, pipeline, missing_classes, device, seed):
    """Try to generate missing classes. Returns True if successful."""
    gc.collect()
    torch.cuda.empty_cache()

    try:
        if pipeline == '2':
            from diffusers import AmusedPipeline
            pipe = AmusedPipeline.from_pretrained(
                "amused/amused-512", variant="fp16",
                torch_dtype=torch.float16, low_cpu_mem_usage=True,
            )
            pipe.enable_model_cpu_offload()
            for class_idx in tqdm(missing_classes, desc="Generating (aMUSEd)"):
                gen = torch.Generator(device=device).manual_seed(seed + class_idx)
                img = pipe(AMUSED_PROMPTS[class_idx], num_inference_steps=12,
                           guidance_scale=10.0, generator=gen).images[0]
                img.save(os.path.join(output_dir, f'generated_correct_class{class_idx}.png'))
                torch.cuda.empty_cache()
        else:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
                safety_checker=None, low_cpu_mem_usage=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            for class_idx in tqdm(missing_classes, desc="Generating (SD 1.5)"):
                gen = torch.Generator(device=device).manual_seed(seed + class_idx)
                img = pipe(SD_PROMPTS[class_idx], negative_prompt=SD_NEGATIVE_PROMPT,
                           num_inference_steps=25, guidance_scale=7.5,
                           height=512, width=512, generator=gen).images[0]
                img.save(os.path.join(output_dir, f'generated_correct_class{class_idx}.png'))
                torch.cuda.empty_cache()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except (MemoryError, OSError) as e:
        print(f"  Could not generate missing images (low memory): {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default='both', choices=['2', '3', 'both'])
    parser.add_argument('--stimuli_dir', type=str, default='data/VisualImagery/stimuli')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--skip_generate', action='store_true',
                        help='Skip generating missing images, use placeholders only')
    args = parser.parse_args()

    device = f'cuda:{args.cuda}'

    if args.pipeline in ('2', 'both'):
        print("\n" + "=" * 60)
        print("Pipeline 2: EEG -> Text -> aMUSEd-512")
        print("=" * 60)
        p2_dir = os.path.join(args.output_dir, 'vi_images')
        os.makedirs(p2_dir, exist_ok=True)

        existing = find_existing_images(p2_dir)
        missing = [i for i in range(10) if i not in existing]

        if missing and not args.skip_generate:
            print(f"  Attempting to generate missing classes: {missing}")
            success = try_generate_missing(p2_dir, '2', missing, device, args.seed)
            if success:
                print("  Successfully generated all missing images!")

        p2_images = load_images_for_pipeline(p2_dir)
        p2_out = os.path.join(p2_dir, 'summary_grid_correct.png')
        create_summary_grid(
            args.stimuli_dir, p2_images,
            "Pipeline 2: EEG -> Text -> aMUSEd",
            p2_out, img_size=args.img_size,
        )
        del p2_images
        gc.collect()

    if args.pipeline in ('3', 'both'):
        print("\n" + "=" * 60)
        print("Pipeline 3: EEG -> CLIP -> SD 1.5")
        print("=" * 60)
        p3_dir = os.path.join(args.output_dir, 'vi_eeg2img')
        os.makedirs(p3_dir, exist_ok=True)

        existing = find_existing_images(p3_dir)
        missing = [i for i in range(10) if i not in existing]

        if missing and not args.skip_generate:
            print(f"  Attempting to generate missing classes: {missing}")
            success = try_generate_missing(p3_dir, '3', missing, device, args.seed)
            if success:
                print("  Successfully generated all missing images!")

        p3_images = load_images_for_pipeline(p3_dir)
        p3_out = os.path.join(p3_dir, 'summary_grid_correct.png')
        create_summary_grid(
            args.stimuli_dir, p3_images,
            "Pipeline 3: EEG -> CLIP -> SD 1.5",
            p3_out, img_size=args.img_size,
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
