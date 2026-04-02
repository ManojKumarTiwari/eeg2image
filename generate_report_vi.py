"""
Generate a CSV report of predictions on the Visual Imagery test set.

Columns: index, original_label, predicted_label, original_text, predicted_text

Usage:
    python generate_report_vi.py
    python generate_report_vi.py --projection_path pth_downtasks/eeg_llm_vi/projection_epoch5.pth
                                  --lora_dir pth_downtasks/eeg_llm_vi/lora_epoch5
                                  --output outputs/vi_test_report.csv
"""

import argparse
import csv
import os
import torch
from tqdm import tqdm
from peft import PeftModel

from models.eeg_llm import EEGLanguageModel
from datasets.visual_imagery_llm_dataset import (
    VisualImageryLLMDataset, VisualImageryLLMCollator,
    VI_LABEL_MAP, VI_KEYWORDS
)
import lmdb


VI_CLASS_NAMES = [
    'dog', 'bird', 'fish',
    'pentagram', 'square', 'circle',
    'scissor', 'watch', 'cup', 'chair',
]


def extract_label(text, keywords):
    text = text.lower()
    best_label, best_count = -1, 0
    for label_id, kws in keywords.items():
        count = sum(1 for kw in kws if kw in text)
        if count > best_count:
            best_count = count
            best_label = label_id
    return best_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir',    type=str, default='data/VisualImagery/processed_lmdb')
    parser.add_argument('--projection_path', type=str, default='pth_downtasks/eeg_llm_vi/projection_epoch5.pth')
    parser.add_argument('--lora_dir',        type=str, default='pth_downtasks/eeg_llm_vi/lora_epoch5')
    parser.add_argument('--foundation_dir',  type=str, default='pth/CSBrain.pth')
    parser.add_argument('--output',          type=str, default='outputs/vi_test_report.csv')
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--max_new_tokens',  type=int, default=64)
    parser.add_argument('--cuda',            type=int, default=0)
    parser.add_argument('--n_layer',         type=int, default=12)
    parser.add_argument('--llm_model_name',  type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--llm_dim',         type=int, default=2048)
    parser.add_argument('--lora_rank',       type=int, default=8)
    parser.add_argument('--lora_alpha',      type=int, default=16)
    parser.add_argument('--dropout',         type=float, default=0.1)
    parser.add_argument('--temporal_pool_stride', type=int, default=1)
    args = parser.parse_args()

    args.downstream_dataset = 'VI'
    args.use_pretrained_weights = True
    args.lora_rank = 8
    args.lora_alpha = 16

    torch.cuda.set_device(args.cuda)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── Build model ──────────────────────────────────────────────────────────
    print("Loading model...")
    model = EEGLanguageModel(args)

    # Load projection + token reducer
    state = torch.load(args.projection_path, map_location=f'cuda:{args.cuda}', weights_only=False)
    model.eeg_projection.load_state_dict(state['projection'])
    model.token_reducer.load_state_dict(state['token_reducer'])
    print(f"Loaded projection from {args.projection_path} (epoch {state['epoch']}, val_acc {state['val_acc']:.4f})")

    # Load LoRA adapter
    model.llm = PeftModel.from_pretrained(model.llm, args.lora_dir)
    print(f"Loaded LoRA from {args.lora_dir}")

    model.eval()

    # ── Load test dataset ────────────────────────────────────────────────────
    print("Loading test dataset...")
    shared_db = lmdb.open(args.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False)
    test_set = VisualImageryLLMDataset(args.datasets_dir, mode='test', db=shared_db)
    collator = VisualImageryLLMCollator(model.tokenizer, max_target_len=128, mode='eval')
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             collate_fn=collator, shuffle=False)
    print(f"Test samples: {len(test_set)}")

    # ── Run inference ────────────────────────────────────────────────────────
    rows = []
    correct = 0
    idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            generated_texts = model.generate(
                eeg_data=batch['eeg_data'].cuda(),
                prompt_ids=batch['prompt_ids'].cuda(),
                prompt_mask=batch['prompt_mask'].cuda(),
                max_new_tokens=args.max_new_tokens,
            )
            label_ids = batch['label_ids'].numpy()

            for pred_text, true_label in zip(generated_texts, label_ids):
                true_label = int(true_label)
                pred_label = extract_label(pred_text, VI_KEYWORDS)

                original_text = VI_LABEL_MAP[true_label][0]
                original_class = VI_CLASS_NAMES[true_label]
                predicted_class = VI_CLASS_NAMES[pred_label] if pred_label >= 0 else 'unknown'

                if pred_label == true_label:
                    correct += 1

                rows.append({
                    'index':           idx,
                    'original_label':  true_label,
                    'predicted_label': pred_label,
                    'original_class':  original_class,
                    'predicted_class': predicted_class,
                    'original_text':   original_text,
                    'predicted_text':  pred_text.strip(),
                    'correct':         pred_label == true_label,
                })
                idx += 1

            torch.cuda.empty_cache()

    acc = correct / len(rows) if rows else 0
    print(f"\nTest Accuracy: {acc:.4f} ({correct}/{len(rows)})")

    # ── Write CSV ────────────────────────────────────────────────────────────
    fieldnames = ['index', 'original_label', 'predicted_label',
                  'original_class', 'predicted_class',
                  'original_text', 'predicted_text', 'correct']

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report saved to: {args.output}")

    # ── Per-class accuracy ───────────────────────────────────────────────────
    print("\nPer-class accuracy:")
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for row in rows:
        class_total[row['original_class']] += 1
        if row['correct']:
            class_correct[row['original_class']] += 1
    for cls in VI_CLASS_NAMES:
        total = class_total[cls]
        corr  = class_correct[cls]
        print(f"  {cls:12s}: {corr:3d}/{total:3d}  ({corr/total*100:.1f}%)")


if __name__ == '__main__':
    main()
