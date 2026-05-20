"""
baseline_linear_probe.py — Baseline 1: Linear Probe on frozen CSBrain features.

Pipeline:
  EEG (B,32,4,200)
    -> CSBrain (frozen)       -> (B,32,4,200)
    -> EEGTokenReducer        -> (B,20,200)
    -> mean pool              -> (B,200)
    -> Linear(200, 10)        -> class logits

This is the simplest possible classifier on top of the frozen foundation
encoder. If our EEGCLIPMapper (35.8%) substantially exceeds this, it validates
that the learned mapping to CLIP space adds discriminative value.

Usage:
    python baseline_linear_probe.py \\
        --datasets_dir data/VisualImagery/processed_lmdb \\
        --foundation_dir pth/CSBrain.pth \\
        --model_dir pth_downtasks/baseline_linear_probe \\
        --epochs 30 --seed 42
"""

import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from timeit import default_timer as timer
from tqdm import tqdm

from models.eeg_llm import (
    EEGTokenReducer, VI_BRAIN_REGIONS, VI_ELECTRODE_LABELS,
    VI_TOPOLOGY, _build_sorted_indices,
)
from models.CSBrain import CSBrain


VI_CLASS_NAMES = [
    'dog', 'bird', 'fish',
    'pentagram', 'square', 'circle',
    'scissor', 'watch', 'cup', 'chair',
]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ─── Dataset ─────────────────────────────────────────────────────────────────

def get_data_loaders(args):
    import lmdb
    from torch.utils.data import DataLoader
    from datasets.visual_imagery_llm_dataset import VisualImageryLLMDataset

    shared_db = lmdb.open(
        args.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False
    )

    def collate(batch):
        eeg = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {'eeg_data': eeg, 'label_ids': labels}

    loaders = {}
    for mode in ('train', 'val', 'test'):
        ds = VisualImageryLLMDataset(args.datasets_dir, mode=mode, db=shared_db)
        loaders[mode] = DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=(mode == 'train'),
            collate_fn=collate,
            num_workers=0, pin_memory=True,
        )
        print(f"  {mode}: {len(ds)} samples, {len(loaders[mode])} batches")

    return loaders


# ─── Model ────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear layer on top of mean-pooled frozen CSBrain features."""

    def __init__(self, eeg_dim: int = 200, n_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(eeg_dim, n_classes)

    def forward(self, eeg_tokens: torch.Tensor):
        """
        Args:
            eeg_tokens: (B, 20, 200) — output of EEGTokenReducer
        Returns:
            class_logits: (B, 10)
        """
        pooled = eeg_tokens.mean(dim=1)       # (B, 200)
        return self.classifier(pooled)         # (B, 10)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ─── Encoder building ─────────────────────────────────────────────────────────

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
    encoder = encoder.to(device).eval()

    token_reducer = EEGTokenReducer(
        area_config=encoder.area_config, temporal_pool_stride=1,
    ).to(device)

    return encoder, token_reducer


# ─── Training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_eeg(encoder, token_reducer, eeg_data):
    encoder.eval()
    with torch.amp.autocast('cuda', enabled=False):
        features = encoder(eeg_data[:, :32, :, :].float())
    return token_reducer(features)   # (B, 20, 200)


def train_epoch(model, loader, optimizer, scheduler, encoder, token_reducer, device, epoch):
    model.train()
    losses, correct, total = [], 0, 0
    start = timer()

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}", mininterval=10):
        eeg_data  = batch['eeg_data'].to(device)
        label_ids = batch['label_ids'].to(device)

        eeg_tokens = encode_eeg(encoder, token_reducer, eeg_data)

        logits = model(eeg_tokens)
        loss = F.cross_entropy(logits, label_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        correct += (logits.argmax(dim=-1) == label_ids).sum().item()
        total   += len(label_ids)

    elapsed = (timer() - start) / 60
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}  train_acc={correct/total:.4f}  "
          f"lr={lr:.2e}  t={elapsed:.1f}min")


@torch.no_grad()
def evaluate(model, loader, encoder, token_reducer, device, split='val'):
    model.eval()
    correct, total = 0, 0
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for batch in tqdm(loader, desc=f"Evaluating ({split})", mininterval=10):
        eeg_data  = batch['eeg_data'].to(device)
        label_ids = batch['label_ids'].numpy()

        eeg_tokens = encode_eeg(encoder, token_reducer, eeg_data)
        logits = model(eeg_tokens)
        preds = logits.float().argmax(dim=-1).cpu().numpy()

        correct += (preds == label_ids).sum()
        total   += len(label_ids)
        for p, t in zip(preds, label_ids):
            class_total[t] += 1
            if p == t:
                class_correct[t] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, class_correct, class_total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--cuda',             type=int,   default=0)
    parser.add_argument('--epochs',           type=int,   default=30)
    parser.add_argument('--batch_size',       type=int,   default=32)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--weight_decay',     type=float, default=0.01)
    parser.add_argument('--datasets_dir',     type=str,   default='data/VisualImagery/processed_lmdb')
    parser.add_argument('--foundation_dir',   type=str,   default='pth/CSBrain.pth')
    parser.add_argument('--model_dir',        type=str,   default='pth_downtasks/baseline_linear_probe')
    parser.add_argument('--output_dir',       type=str,   default='outputs/baselines')
    parser.add_argument('--n_layer',          type=int,   default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nBuilding EEG encoder (frozen)...")
    encoder, token_reducer = build_eeg_encoder(args, device)

    model = LinearProbe(eeg_dim=200, n_classes=10).to(device)
    print(f"LinearProbe parameters: {model.num_parameters:,}")

    print("\nLoading datasets...")
    loaders = get_data_loaders(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loaders['train']) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining LinearProbe for {args.epochs} epochs...")
    print("=" * 60)
    for epoch in range(args.epochs):
        train_epoch(model, loaders['train'], optimizer, scheduler, encoder, token_reducer, device, epoch)
        val_acc, _, _ = evaluate(model, loaders['val'], encoder, token_reducer, device, split='val')
        print(f"  Val acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            path = os.path.join(args.model_dir, f'linear_probe_best.pth')
            torch.save({'model': best_state, 'val_acc': best_val_acc, 'epoch': epoch+1}, path)
            print(f"  New best val_acc={best_val_acc:.4f} -> saved to {path}")

    # Final test evaluation with best checkpoint
    print("\n" + "=" * 60)
    print("Final Test Evaluation (best checkpoint)")
    print("=" * 60)
    if best_state:
        model.load_state_dict(best_state)

    test_acc, class_correct, class_total = evaluate(
        model, loaders['test'], encoder, token_reducer, device, split='test'
    )

    print(f"\nTest accuracy: {test_acc:.4f} ({int(test_acc * sum(class_total.values()))}/{sum(class_total.values())})")
    print(f"Chance level:  0.1000 (10 classes)")
    print("\nPer-class accuracy:")
    per_class = []
    for i, name in enumerate(VI_CLASS_NAMES):
        tot  = class_total[i]
        corr = class_correct[i]
        pct  = corr / tot * 100 if tot > 0 else 0.0
        print(f"  {name:12s}: {corr:3d}/{tot:3d}  ({pct:.1f}%)")
        per_class.append({'class': name, 'correct': corr, 'total': tot, 'accuracy': pct / 100})

    # Save per-class CSV
    per_class_path = os.path.join(args.output_dir, 'linear_probe_per_class.csv')
    with open(per_class_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'correct', 'total', 'accuracy'])
        writer.writeheader()
        writer.writerows(per_class)
    print(f"\nPer-class results -> {per_class_path}")

    # Append to summary CSV
    summary_path = os.path.join(args.output_dir, 'summary.csv')
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'test_acc', 'val_acc', 'params', 'seed', 'notes'])
        if not summary_exists:
            writer.writeheader()
        writer.writerow({
            'method': 'Linear Probe (CSBrain)',
            'test_acc': f'{test_acc:.4f}',
            'val_acc': f'{best_val_acc:.4f}',
            'params': model.num_parameters,
            'seed': args.seed,
            'notes': 'Frozen CSBrain -> mean pool -> Linear(200,10)',
        })
    print(f"Summary appended -> {summary_path}")


if __name__ == '__main__':
    main()
