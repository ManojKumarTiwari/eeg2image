"""
baseline_no_qformer.py — Baseline 2: MLP mapper without Q-Former / transformer.

Pipeline:
  EEG (B,32,4,200)
    -> CSBrain (frozen)               -> (B,32,4,200)
    -> EEGTokenReducer                -> (B,20,200)
    -> MLPMapper (token-wise MLP)     -> (B,20,768)
    -> mean pool                      -> (B,768)
    -> pool projection + classifier   -> class logits

This ablates the two key architectural contributions of EEGCLIPMapper:
  1. Transformer self-attention over EEG tokens
  2. Q-Former cross-attention (learnable queries 20->77)

Same training objectives as Pipeline 3 (cls + cont + cos) but with a simple
MLP instead of the transformer+cross-attention stack. This isolates whether
performance gains come from the architecture or merely from CLIP alignment.

Usage:
    python baseline_no_qformer.py \\
        --datasets_dir data/VisualImagery/processed_lmdb \\
        --stimuli_dir data/VisualImagery/stimuli \\
        --foundation_dir pth/CSBrain.pth \\
        --model_dir pth_downtasks/baseline_no_qformer \\
        --epochs 20 --seed 42
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
from models.eeg_clip_mapper import CLIPImageTargetBuilder
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

class MLPMapper(nn.Module):
    """
    Direct MLP mapping: EEG tokens -> CLIP embedding space.

    Ablates the transformer self-attention and Q-Former cross-attention from
    EEGCLIPMapper. Uses a token-wise MLP followed by mean pooling.

    Architecture:
        (B, 20, 200) -> token MLP -> (B, 20, 512) -> (B, 20, 768)
        -> mean pool -> (B, 768) -> pool proj -> (B, 768) -> classifier -> (B, 10)

    Trainable params: ~1.5M (vs 4.8M for EEGCLIPMapper)
    """

    def __init__(
        self,
        eeg_dim: int = 200,
        hidden_dim: int = 512,
        clip_dim: int = 768,
        n_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token-wise MLP (same input projection as EEGCLIPMapper but no transformer after)
        self.token_proj = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )

        # Pool projection (mirrors EEGCLIPMapper.pool_proj)
        self.pool_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.Tanh(),
        )

        # Classification head (same as EEGCLIPMapper.classifier)
        self.classifier = nn.Linear(clip_dim, n_classes)

    def forward(self, eeg_tokens: torch.Tensor):
        """
        Args:
            eeg_tokens: (B, 20, 200) from EEGTokenReducer
        Returns:
            prompt_embeds:  (B, 768)  — CLIP-aligned pooled vector
            class_logits:   (B, 10)
        """
        x = self.token_proj(eeg_tokens)         # (B, 20, 768)
        pooled = x.mean(dim=1)                  # (B, 768)
        prompt_embeds = self.pool_proj(pooled)  # (B, 768)
        class_logits = self.classifier(pooled)  # (B, 10)
        return prompt_embeds, class_logits

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


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(pred_pooled, class_logits, img_pooled_targets, img_class_pooled,
                 label_ids, temperature, lambdas):
    """
    Same losses as EEGCLIPMapper but without L_mse (no sequence output).
    """
    L_cls = F.cross_entropy(class_logits, label_ids)

    pred_norm  = F.normalize(pred_pooled, dim=-1)
    class_norm = F.normalize(img_class_pooled, dim=-1)
    logits_cont = pred_norm @ class_norm.T / temperature
    L_cont = F.cross_entropy(logits_cont, label_ids)

    img_norm = F.normalize(img_pooled_targets, dim=-1)
    L_cos = (1.0 - (pred_norm * img_norm).sum(dim=-1)).mean()

    loss = (lambdas['cls'] * L_cls
          + lambdas['cont'] * L_cont
          + lambdas['cos'] * L_cos)

    return loss, L_cls.item(), L_cont.item(), L_cos.item()


# ─── Training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_eeg(encoder, token_reducer, eeg_data):
    encoder.eval()
    with torch.amp.autocast('cuda', enabled=False):
        features = encoder(eeg_data[:, :32, :, :].float())
    return token_reducer(features)   # (B, 20, 200)


def train_epoch(model, loader, optimizer, scheduler, encoder, token_reducer,
                img_pooled, device, args, epoch):
    model.train()
    all_loss, all_cls, all_cont, all_cos = [], [], [], []
    start = timer()

    lambdas = {
        'cls':  args.lambda_cls,
        'cont': args.lambda_contrastive,
        'cos':  args.lambda_cos,
    }

    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", mininterval=10)):
        eeg_data  = batch['eeg_data'].to(device)
        label_ids = batch['label_ids'].to(device)

        eeg_tokens = encode_eeg(encoder, token_reducer, eeg_data)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred_pooled, class_logits = model(eeg_tokens.float())
            img_tgts = img_pooled[label_ids].float()

            loss, lcls, lcont, lcos = compute_loss(
                pred_pooled.float(), class_logits.float(),
                img_tgts, img_pooled.float(),
                label_ids, args.temperature, lambdas,
            )
            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        all_loss.append(loss.item() * args.gradient_accumulation_steps)
        all_cls.append(lcls); all_cont.append(lcont); all_cos.append(lcos)

    elapsed = (timer() - start) / 60
    lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch+1}: loss={np.mean(all_loss):.4f} "
        f"(cls={np.mean(all_cls):.4f} cont={np.mean(all_cont):.4f} "
        f"cos={np.mean(all_cos):.4f}) "
        f"lr={lr:.2e} t={elapsed:.1f}min"
    )


@torch.no_grad()
def evaluate(model, loader, encoder, token_reducer, img_pooled, device, split='val'):
    model.eval()
    correct_cls, correct_nn, total = 0, 0, 0
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    img_norm = F.normalize(img_pooled.float(), dim=-1)

    for batch in tqdm(loader, desc=f"Evaluating ({split})", mininterval=10):
        eeg_data  = batch['eeg_data'].to(device)
        label_ids = batch['label_ids'].numpy()

        eeg_tokens = encode_eeg(encoder, token_reducer, eeg_data)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred_pooled, class_logits = model(eeg_tokens.float())

        preds_cls = class_logits.float().argmax(dim=-1).cpu().numpy()
        pred_norm = F.normalize(pred_pooled.float(), dim=-1)
        preds_nn  = (pred_norm @ img_norm.T).argmax(dim=-1).cpu().numpy()

        correct_cls += (preds_cls == label_ids).sum()
        correct_nn  += (preds_nn  == label_ids).sum()
        total += len(label_ids)

        for p, t in zip(preds_cls, label_ids):
            class_total[t] += 1
            if p == t:
                class_correct[t] += 1
        torch.cuda.empty_cache()

    acc_cls = correct_cls / total if total > 0 else 0.0
    acc_nn  = correct_nn  / total if total > 0 else 0.0
    return acc_cls, acc_nn, class_correct, class_total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--cuda',             type=int,   default=0)
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--temperature',      type=float, default=0.5)
    parser.add_argument('--lambda_cls',       type=float, default=5.0)
    parser.add_argument('--lambda_contrastive', type=float, default=2.0)
    parser.add_argument('--lambda_cos',       type=float, default=1.0)
    parser.add_argument('--datasets_dir',     type=str,   default='data/VisualImagery/processed_lmdb')
    parser.add_argument('--stimuli_dir',      type=str,   default='data/VisualImagery/stimuli')
    parser.add_argument('--foundation_dir',   type=str,   default='pth/CSBrain.pth')
    parser.add_argument('--clip_model_id',    type=str,   default='openai/clip-vit-large-patch14')
    parser.add_argument('--model_dir',        type=str,   default='pth_downtasks/baseline_no_qformer')
    parser.add_argument('--output_dir',       type=str,   default='outputs/baselines')
    parser.add_argument('--n_layer',          type=int,   default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    # MLPMapper hyperparameters
    parser.add_argument('--hidden_dim',       type=int,   default=512)
    parser.add_argument('--clip_dim',         type=int,   default=768)
    parser.add_argument('--dropout',          type=float, default=0.1)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Precompute CLIP image targets ─────────────────────────────────────
    print("\nBuilding CLIP image targets (stimulus images)...")
    img_builder = CLIPImageTargetBuilder(
        stimuli_dir=args.stimuli_dir,
        clip_model_id=args.clip_model_id,
        device=str(device),
    )
    img_targets_pooled = img_builder.build().to(device)   # (10, 768)

    # ── 2. Build frozen EEG encoder ──────────────────────────────────────────
    print("\nBuilding EEG encoder (frozen)...")
    encoder, token_reducer = build_eeg_encoder(args, device)

    # ── 3. Build MLP mapper (no Q-Former) ────────────────────────────────────
    model = MLPMapper(
        eeg_dim=200,
        hidden_dim=args.hidden_dim,
        clip_dim=args.clip_dim,
        n_classes=10,
        dropout=args.dropout,
    ).to(device)
    print(f"MLPMapper parameters: {model.num_parameters:,}")

    # ── 4. Load data ─────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    loaders = get_data_loaders(args)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loaders['train']) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining MLPMapper (no Q-Former) for {args.epochs} epochs...")
    print("=" * 60)
    for epoch in range(args.epochs):
        train_epoch(model, loaders['train'], optimizer, scheduler, encoder, token_reducer,
                    img_targets_pooled, device, args, epoch)
        val_acc_cls, val_acc_nn, _, _ = evaluate(
            model, loaders['val'], encoder, token_reducer, img_targets_pooled, device, 'val'
        )
        print(f"  Val cls_acc={val_acc_cls:.4f}  nn_acc={val_acc_nn:.4f}")

        if val_acc_cls > best_val_acc:
            best_val_acc = val_acc_cls
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            path = os.path.join(args.model_dir, 'mlp_mapper_best.pth')
            torch.save({'model': best_state, 'val_acc': best_val_acc, 'epoch': epoch+1}, path)
            print(f"  New best val_acc={best_val_acc:.4f} -> saved to {path}")

    # ── 6. Final test evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final Test Evaluation (best checkpoint)")
    print("=" * 60)
    if best_state:
        model.load_state_dict(best_state)

    test_acc_cls, test_acc_nn, class_correct, class_total = evaluate(
        model, loaders['test'], encoder, token_reducer, img_targets_pooled, device, 'test'
    )

    total_samples = sum(class_total.values())
    print(f"\nTest cls_acc: {test_acc_cls:.4f} ({int(test_acc_cls * total_samples)}/{total_samples})")
    print(f"Test nn_acc:  {test_acc_nn:.4f}")
    print(f"Chance level: 0.1000 (10 classes)")
    print("\nPer-class accuracy (classifier):")
    per_class = []
    for i, name in enumerate(VI_CLASS_NAMES):
        tot  = class_total[i]
        corr = class_correct[i]
        pct  = corr / tot * 100 if tot > 0 else 0.0
        print(f"  {name:12s}: {corr:3d}/{tot:3d}  ({pct:.1f}%)")
        per_class.append({'class': name, 'correct': corr, 'total': tot, 'accuracy': pct / 100})

    # Save per-class CSV
    per_class_path = os.path.join(args.output_dir, 'no_qformer_per_class.csv')
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
            'method': 'No Q-Former (MLP only)',
            'test_acc': f'{test_acc_cls:.4f}',
            'val_acc': f'{best_val_acc:.4f}',
            'params': model.num_parameters,
            'seed': args.seed,
            'notes': 'Frozen CSBrain -> token MLP -> mean pool -> Linear head; no transformer/cross-attn',
        })
    print(f"Summary appended -> {summary_path}")


if __name__ == '__main__':
    main()
