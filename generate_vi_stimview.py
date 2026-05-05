"""
generate_vi_stimview.py — Pipeline 2 (EEG->Text->Image) for stimulus-view experiment.

Same as generate_vi.py but defaults to the stimview LMDB and model checkpoints
trained on the 4s image-shown window (rather than the 4s imagery window).

Usage:
    python generate_vi_stimview.py --num_samples 20
    python generate_vi_stimview.py --num_samples 20 --output_dir outputs/vi_images_stimview
"""

import argparse
import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from peft import PeftModel

from diffusers import AmusedPipeline
from models.eeg_llm import EEGLanguageModel
from datasets.visual_imagery_llm_dataset import (
    VisualImageryLLMDataset, VisualImageryLLMCollator, VI_KEYWORDS
)
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

VI_PROMPTS = {
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_label(text, keywords):
    text = text.lower()
    best_label, best_count = -1, 0
    for label_id, kws in keywords.items():
        count = sum(1 for kw in kws if kw in text)
        if count > best_count:
            best_count = count
            best_label = label_id
    return best_label


def make_comparison_image(
    stimulus_path, generated_img, true_label, pred_label, pred_text, idx, size=512
):
    if stimulus_path and os.path.exists(stimulus_path):
        stimulus = Image.open(stimulus_path).convert('RGB').resize((size, size))
    else:
        stimulus = Image.new('RGB', (size, size), color=(200, 200, 200))

    gen = generated_img.resize((size, size))

    caption_h = 60
    canvas = Image.new('RGB', (size * 2, size + caption_h), color=(255, 255, 255))
    canvas.paste(stimulus, (0, 0))
    canvas.paste(gen, (size, 0))

    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, size, 30], fill=(50, 50, 50))
    draw.text((10, 8), f"Original: {VI_CLASS_NAMES[true_label]}", fill='white')

    pred_name = VI_CLASS_NAMES[pred_label] if pred_label >= 0 else 'unknown'
    correct = pred_label == true_label
    header_color = (50, 150, 50) if correct else (180, 50, 50)
    draw.rectangle([size, 0, size * 2, 30], fill=header_color)
    draw.text((size + 10, 8), f"Predicted: {pred_name} {'OK' if correct else 'X'}", fill='white')

    caption_y = size + 5
    short_text = (pred_text[:90] + '...') if len(pred_text) > 90 else pred_text
    draw.text((10, caption_y), f"[{idx}] {short_text}", fill=(30, 30, 30))

    return canvas


def main():
    parser = argparse.ArgumentParser()
    # ── Stimview-specific defaults ────────────────────────────────────────────
    parser.add_argument('--datasets_dir',    type=str,
                        default='data/VisualImagery/processed_lmdb_stimview')
    parser.add_argument('--projection_path', type=str,
                        default='pth_downtasks/eeg_llm_vi_stimview/projection_epoch5.pth')
    parser.add_argument('--lora_dir',        type=str,
                        default='pth_downtasks/eeg_llm_vi_stimview/lora_epoch5')
    parser.add_argument('--output_dir',      type=str,
                        default='outputs/vi_images_stimview')
    # ── Shared args (same as generate_vi.py) ─────────────────────────────────
    parser.add_argument('--foundation_dir',  type=str, default='pth/CSBrain.pth')
    parser.add_argument('--stimuli_dir',     type=str, default='data/VisualImagery/stimuli')
    parser.add_argument('--image_model',     type=str, default='amused/amused-512')
    parser.add_argument('--num_samples',     type=int, default=20)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--max_new_tokens',  type=int, default=64)
    parser.add_argument('--num_inference_steps', type=int, default=12)
    parser.add_argument('--guidance_scale',  type=float, default=10.0)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--cuda',            type=int, default=0)
    parser.add_argument('--n_layer',         type=int, default=12)
    parser.add_argument('--llm_model_name',  type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--llm_dim',         type=int, default=2048)
    parser.add_argument('--lora_rank',       type=int, default=8)
    parser.add_argument('--lora_alpha',      type=int, default=16)
    parser.add_argument('--dropout',         type=float, default=0.1)
    parser.add_argument('--temporal_pool_stride', type=int, default=1)
    args = parser.parse_args()

    args.downstream_dataset   = 'VI'
    args.use_pretrained_weights = True

    setup_seed(args.seed)
    torch.cuda.set_device(args.cuda)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Stage 1: Load EEG-LLM ────────────────────────────────────────────────
    print("=" * 60)
    print("Stage 1: Loading EEG -> Text model  [stimview experiment]")
    print("=" * 60)
    model = EEGLanguageModel(args)

    state = torch.load(args.projection_path, map_location=f'cuda:{args.cuda}', weights_only=False)
    model.eeg_projection.load_state_dict(state['projection'])
    model.token_reducer.load_state_dict(state['token_reducer'])
    print(f"Loaded projection (epoch {state['epoch']}, val_acc {state['val_acc']:.4f})")

    model.llm = PeftModel.from_pretrained(model.llm, args.lora_dir)
    model.eval()

    shared_db = lmdb.open(args.datasets_dir, readonly=True, lock=False,
                          readahead=True, meminit=False)
    test_set  = VisualImageryLLMDataset(args.datasets_dir, mode='test', db=shared_db)
    collator  = VisualImageryLLMCollator(model.tokenizer, max_target_len=128, mode='eval')
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             collate_fn=collator, shuffle=False)

    print(f"\nGenerating text for {args.num_samples} test samples...")
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="EEG->Text"):
            if len(results) >= args.num_samples:
                break
            texts = model.generate(
                eeg_data=batch['eeg_data'].cuda(),
                prompt_ids=batch['prompt_ids'].cuda(),
                prompt_mask=batch['prompt_mask'].cuda(),
                max_new_tokens=args.max_new_tokens,
            )
            for text, true_label in zip(texts, batch['label_ids'].numpy()):
                if len(results) >= args.num_samples:
                    break
                pred_label = extract_label(text, VI_KEYWORDS)
                results.append({
                    'true_label': int(true_label),
                    'pred_label': pred_label,
                    'text': text.strip(),
                })
            torch.cuda.empty_cache()

    correct = sum(1 for r in results if r['pred_label'] == r['true_label'])
    print(f"Text decoding accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    del model
    torch.cuda.empty_cache()

    # ── Stage 2: Text -> Image (aMUSEd-512) ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 2: Loading {args.image_model} for image generation")
    print("=" * 60)

    device = f'cuda:{args.cuda}'
    pipe = AmusedPipeline.from_pretrained(
        args.image_model,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    prompts = [
        VI_PROMPTS.get(r['pred_label'],
                       VI_CLASS_NAMES[r['pred_label']] if r['pred_label'] >= 0 else "abstract shape")
        for r in results
    ]

    print(f"\nGenerating {len(prompts)} images with aMUSEd-512...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = []
    for prompt in tqdm(prompts, desc="Generating"):
        img = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        images.append(img)
        torch.cuda.empty_cache()

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
        comp = make_comparison_image(stimulus_path, gen_img, true_label, pred_label, r['text'], i)
        comp_path = os.path.join(args.output_dir,
                                 f'comparison_{i:04d}_true{true_label}_pred{pred_label}.png')
        comp.save(comp_path)
        comparison_imgs.append(comp)

        status = 'CORRECT' if pred_label == true_label else 'WRONG'
        print(f"  [{i:03d}] true={VI_CLASS_NAMES[true_label]:10s} "
              f"pred={VI_CLASS_NAMES[pred_label] if pred_label>=0 else 'unknown':10s} {status}")

    print("\nCreating summary grid...")
    cols = 4
    rows = (len(comparison_imgs) + cols - 1) // cols
    thumb_w, thumb_h = 512, 256
    grid = Image.new('RGB', (thumb_w * cols, thumb_h * rows), color=(240, 240, 240))
    for i, img in enumerate(comparison_imgs):
        thumb = img.resize((thumb_w, thumb_h))
        row, col = divmod(i, cols)
        grid.paste(thumb, (col * thumb_w, row * thumb_h))

    grid_path = os.path.join(args.output_dir, 'summary_grid.png')
    grid.save(grid_path)

    print(f"\nDone! {len(images)} images saved to: {args.output_dir}")
    print(f"Summary grid: {grid_path}")
    print(f"Final accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()
