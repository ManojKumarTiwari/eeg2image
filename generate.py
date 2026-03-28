"""
generate.py — Full EEG → Text → Image inference pipeline.

Stage 1 (EEG → Text):
    Loads a trained EEGLanguageModel (CSBrain encoder + EEGProjection + LoRA-fine-tuned
    TinyLlama) and generates neuroscience text descriptions from raw EEG signals.

Stage 2 (Text → Image, optional):
    Passes the generated text through EEGImageGenerator (Stable Diffusion 2.1,
    Apache 2.0) to synthesise a corresponding visual image.

Usage — text only:
    python generate.py \\
        --foundation_dir pth/CSBrain.pth \\
        --projection_dir pth_downtasks/eeg_llm_bcic/best_projection.pth \\
        --lora_dir pth_downtasks/eeg_llm_bcic/best_lora \\
        --datasets_dir data/BCICIV2a/processed_lmdb \\
        --downstream_dataset BCICIV2a \\
        --num_samples 5

Usage — full EEG → Text → Image pipeline:
    python generate.py \\
        --foundation_dir pth/CSBrain.pth \\
        --projection_dir pth_downtasks/eeg_llm_bcic/best_projection.pth \\
        --lora_dir pth_downtasks/eeg_llm_bcic/best_lora \\
        --datasets_dir data/BCICIV2a/processed_lmdb \\
        --downstream_dataset BCICIV2a \\
        --num_samples 5 \\
        --generate_images \\
        --output_dir outputs/eeg2image
"""

import argparse
import random
import numpy as np
import torch

from datasets.bciciv2a_llm_dataset import LoadDataset as BCICDataset, BCIC_LABEL_MAP
from models.eeg_llm import EEGLanguageModel


BCIC_LABEL_TEXTS = {0: "left hand", 1: "right hand", 2: "feet", 3: "tongue"}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(params) -> EEGLanguageModel:
    """Build EEGLanguageModel and load saved projection + LoRA weights."""
    from peft import PeftModel

    model = EEGLanguageModel(params)
    device = next(model.parameters()).device

    # Load projection + token reducer weights
    state = torch.load(params.projection_dir, map_location=device)
    if 'projection' in state:
        model.eeg_projection.load_state_dict(state['projection'])
        if 'token_reducer' in state:
            model.token_reducer.load_state_dict(state['token_reducer'])
    else:
        model.eeg_projection.load_state_dict(state)
    print(f"Loaded projection from {params.projection_dir}")

    # Load LoRA adapter
    model.llm = PeftModel.from_pretrained(model.llm, params.lora_dir)
    print(f"Loaded LoRA adapter from {params.lora_dir}")

    model.eval()
    return model


def run_eeg_to_text(model, test_loader, params, device):
    """
    Stage 1 — Generate text descriptions from EEG samples.

    Returns:
        results: list of dicts with keys 'true_label', 'generated_text'
    """
    results = []
    count = 0
    for batch in test_loader:
        if count >= params.num_samples:
            break

        generated_texts = model.generate(
            eeg_data=batch['eeg_data'].to(device),
            prompt_ids=batch['prompt_ids'].to(device),
            prompt_mask=batch['prompt_mask'].to(device),
            max_new_tokens=params.max_new_tokens,
        )

        for text, true_label in zip(generated_texts, batch['label_ids'].tolist()):
            if count >= params.num_samples:
                break
            results.append({'true_label': true_label, 'generated_text': text})
            count += 1

    return results


def run_text_to_image(results, params):
    """
    Stage 2 — Generate images from EEG-decoded text descriptions.

    Uses stabilityai/stable-diffusion-2-1 (Apache 2.0 license).

    Returns:
        saved_paths: list of paths to saved PNG files
    """
    from models.image_generator import EEGImageGenerator

    generator = EEGImageGenerator(
        model_id=params.image_model,
        device=f"cuda:{params.cuda}",
        use_half_precision=True,
        enable_attention_slicing=True,
    )

    prompts = []
    true_labels = []
    for r in results:
        prompt = EEGImageGenerator.build_prompt(
            eeg_text=r['generated_text'],
            label=r['true_label'],
            dataset=params.downstream_dataset,
        )
        prompts.append(prompt)
        true_labels.append(r['true_label'])

    print(f"\nGenerating {len(prompts)} image(s) with {params.image_model} ...")
    images = generator.generate(
        prompts=prompts,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        height=params.image_height,
        width=params.image_width,
        seed=params.seed,
    )

    saved = generator.save_images(
        images=images,
        output_dir=params.output_dir,
        prefix="eeg2image",
        true_labels=true_labels,
        prompts=prompts,
    )
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="EEG → Text → Image inference pipeline"
    )

    # ── Model paths ──────────────────────────────────────────────────────────
    parser.add_argument('--foundation_dir', type=str, default='pth/CSBrain.pth')
    parser.add_argument('--projection_dir', type=str,
                        default='pth_downtasks/eeg_llm_bcic/best_projection.pth')
    parser.add_argument('--lora_dir', type=str,
                        default='pth_downtasks/eeg_llm_bcic/best_lora')

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument('--downstream_dataset', type=str, default='BCICIV2a')
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, default=4)

    # ── LLM settings ─────────────────────────────────────────────────────────
    parser.add_argument('--llm_model_name', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--llm_dim', type=int, default=2048)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--max_target_len', type=int, default=128)
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Max tokens to generate per EEG sample')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    parser.add_argument('--temporal_pool_stride', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='pth_downtasks/eeg_llm_bcic')

    # ── Image generation (Stage 2) ───────────────────────────────────────────
    parser.add_argument('--generate_images', action='store_true',
                        help='Run Stage 2: generate images from the decoded text')
    parser.add_argument('--image_model', type=str,
                        default='stabilityai/stable-diffusion-2-1',
                        help='HuggingFace diffusion model ID (Apache 2.0 license)')
    parser.add_argument('--output_dir', type=str, default='outputs/eeg2image',
                        help='Directory to save generated images')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                        help='Diffusion denoising steps (25 fast, 50 high-quality)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)

    # ── General ──────────────────────────────────────────────────────────────
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of test samples to process')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)

    params = parser.parse_args()
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = torch.device(f'cuda:{params.cuda}')

    # ── Stage 1: EEG → Text ──────────────────────────────────────────────────
    print("=" * 60)
    print("Stage 1: EEG → Text")
    print("=" * 60)

    print("Loading EEG-LLM model...")
    model = load_model(params)

    print("Loading test data...")
    load_dataset = BCICDataset(params, model.tokenizer)
    data_loaders = load_dataset.get_data_loader()
    test_loader = data_loaders['test']

    print(f"\nGenerating text for {params.num_samples} test samples:\n")
    results = run_eeg_to_text(model, test_loader, params, device)

    for i, r in enumerate(results):
        label = r['true_label']
        print(f"Sample {i + 1}:")
        print(f"  True class : {label} — {BCIC_LABEL_TEXTS.get(label, str(label))}")
        print(f"  Generated  : {r['generated_text']}")
        print()

    # ── Stage 2: Text → Image ────────────────────────────────────────────────
    if params.generate_images:
        print("=" * 60)
        print("Stage 2: Text → Image  (Stable Diffusion 2.1, Apache 2.0)")
        print("=" * 60)

        # Free LLM memory before loading diffusion model
        del model
        torch.cuda.empty_cache()

        saved_paths = run_text_to_image(results, params)

        print(f"\nGenerated {len(saved_paths)} image(s) → {params.output_dir}")
        for i, (r, path) in enumerate(zip(results, saved_paths)):
            label = r['true_label']
            print(
                f"  [{i + 1}] label={label} ({BCIC_LABEL_TEXTS.get(label, '')}) → {path}"
            )
    else:
        print("\nTip: Add --generate_images to also synthesise images from the text.")


if __name__ == '__main__':
    main()
