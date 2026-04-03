"""
finetune_eeg_to_image.py — Train EEGCLIPMapper to bridge EEG -> CLIP image space.

Pipeline:
  EEG (B,32,4,200)
    -> CSBrain (frozen)          -> (B,32,4,200)
    -> EEGTokenReducer           -> (B,20,200)
    -> EEGCLIPMapper (trainable) -> encoder_hidden_states (B,77,768)
                                    prompt_embeds         (B,768)
                                    class_logits          (B,10)

Training objectives (combined):
  L_cls   : cross-entropy on class_logits (strongest discriminative signal)
  L_cont  : InfoNCE against CLIP image prototypes (pooled, image targets)
  L_cos   : cosine alignment to CLIP image embeddings (pooled)
  L_mse   : MSE on sequence to CLIP text embeddings (keeps aMUSEd-compatible format)

Key insight: CLIP image embeddings of the 10 stimulus images have mean pairwise
cosine sim ~0.54 (vs ~0.90 for text embeddings) — far more discriminative as targets.

Usage:
    python finetune_eeg_to_image.py \\
        --datasets_dir data/VisualImagery/processed_lmdb \\
        --stimuli_dir data/VisualImagery/stimuli \\
        --epochs 20 --warmup_epochs 5 \\
        --model_dir pth_downtasks/eeg_direct
"""

import argparse
import copy
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
from models.eeg_clip_mapper import EEGCLIPMapper, CLIPTextTargetBuilder, CLIPImageTargetBuilder
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
        # VisualImageryLLMDataset returns (ndarray, int) tuples
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


# ─── Model building ───────────────────────────────────────────────────────────

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

    if os.path.exists(args.projection_path):
        ckpt = torch.load(args.projection_path, map_location=device, weights_only=False)
        if 'token_reducer' in ckpt:
            token_reducer.load_state_dict(ckpt['token_reducer'])
            print(f"Warm-started token_reducer from {args.projection_path}")

    return encoder, token_reducer


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(pred_hidden, pred_pooled, class_logits,
                 text_hidden, img_pooled_targets,
                 img_class_pooled, label_ids, temperature, lambdas, phase):
    """
    Args:
        pred_hidden:        (B, 77, 768)  mapper sequence output
        pred_pooled:        (B, 768)      mapper pooled output
        class_logits:       (B, 10)       auxiliary classification logits
        text_hidden:        (B, 77, 768)  CLIP text targets for sequence MSE
        img_pooled_targets: (B, 768)      CLIP image targets per sample (for cosine)
        img_class_pooled:   (10, 768)     CLIP image class prototypes (for InfoNCE)
        label_ids:          (B,)          ground-truth class indices
        lambdas:            dict with keys cls, cont, cos, mse
    """
    # Classification loss (cross-entropy) — strongest discriminative signal
    L_cls = F.cross_entropy(class_logits, label_ids)

    # InfoNCE contrastive against CLIP image class prototypes
    pred_norm  = F.normalize(pred_pooled, dim=-1)
    class_norm = F.normalize(img_class_pooled, dim=-1)
    logits_cont = pred_norm @ class_norm.T / temperature   # (B, 10)
    L_cont = F.cross_entropy(logits_cont, label_ids)

    # Cosine alignment to CLIP image target
    img_norm = F.normalize(img_pooled_targets, dim=-1)
    L_cos = (1.0 - (pred_norm * img_norm).sum(dim=-1)).mean()

    # MSE on sequence to CLIP text targets (keeps aMUSEd conditioning format)
    L_mse = F.mse_loss(pred_hidden, text_hidden) if phase != 'warmup' else torch.tensor(0.0, device=pred_hidden.device)

    loss = (lambdas['cls']  * L_cls
          + lambdas['cont'] * L_cont
          + lambdas['cos']  * L_cos
          + lambdas['mse']  * L_mse)

    return loss, L_cls.item(), L_cont.item(), L_cos.item(), L_mse.item()


# ─── Trainer ─────────────────────────────────────────────────────────────────

class EEGToImageTrainer:
    def __init__(self, args, data_loaders, encoder, token_reducer, mapper,
                 text_targets_hidden, img_targets_pooled):
        self.args = args
        self.loaders = data_loaders
        self.encoder = encoder
        self.token_reducer = token_reducer
        self.mapper = mapper
        self.device = next(mapper.parameters()).device

        # CLIP text targets for sequence MSE (10, 77, 768)
        self.text_hidden = text_targets_hidden.to(self.device)
        # CLIP image targets for pooled contrastive/cosine (10, 768)
        self.img_pooled = img_targets_pooled.to(self.device)

        self.grad_accum = args.gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_val_acc = 0.0
        self.best_mapper_state = None

    def train(self):
        print("=" * 60)
        print(f"Phase 1: Warmup ({self.args.warmup_epochs} epochs) — cls + cont + cos")
        print("=" * 60)
        opt1, sch1 = self._make_optimizer(lr=self.args.lr * 5)
        for epoch in range(self.args.warmup_epochs):
            self._train_epoch(epoch, opt1, sch1, phase='warmup')
            self._validate(epoch)

        joint_epochs = self.args.epochs - self.args.warmup_epochs
        print("=" * 60)
        print(f"Phase 2: Full training ({joint_epochs} epochs) — cls + cont + cos + mse")
        print("=" * 60)
        opt2, sch2 = self._make_optimizer(lr=self.args.lr)
        for epoch in range(self.args.warmup_epochs, self.args.epochs):
            self._train_epoch(epoch, opt2, sch2, phase='joint')
            self._validate(epoch)

        self._test()

    def _make_optimizer(self, lr):
        opt = torch.optim.AdamW(self.mapper.parameters(), lr=lr,
                                weight_decay=self.args.weight_decay)
        steps = len(self.loaders['train']) * self.args.epochs
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
        return opt, sch

    def _encode_eeg(self, eeg_data):
        with torch.no_grad():
            self.encoder.eval()
            with torch.amp.autocast('cuda', enabled=False):
                features = self.encoder(eeg_data[:, :32, :, :].float())
        return self.token_reducer(features)   # (B, 20, 200)

    def _train_epoch(self, epoch, optimizer, scheduler, phase):
        self.mapper.train()
        all_loss, all_cls, all_cont, all_cos, all_mse = [], [], [], [], []
        start = timer()
        optimizer.zero_grad()

        lambdas = {
            'cls':  self.args.lambda_cls,
            'cont': self.args.lambda_contrastive,
            'cos':  self.args.lambda_cos,
            'mse':  self.args.lambda_mse if phase != 'warmup' else 0.0,
        }

        dtype = next(self.mapper.parameters()).dtype

        for step, batch in enumerate(tqdm(
            self.loaders['train'], desc=f"Epoch {epoch+1}", mininterval=10
        )):
            eeg_data  = batch['eeg_data'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            eeg_tokens = self._encode_eeg(eeg_data)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_hidden, pred_pooled, class_logits = self.mapper(eeg_tokens.to(dtype))

                text_hidden = self.text_hidden[label_ids].float()       # (B, 77, 768)
                img_pooled  = self.img_pooled[label_ids].float()        # (B, 768)

                total, lcls, lcont, lcos, lmse = compute_loss(
                    pred_hidden.float(), pred_pooled.float(), class_logits.float(),
                    text_hidden, img_pooled,
                    self.img_pooled.float(), label_ids,
                    self.args.temperature, lambdas, phase,
                )
                loss = total / self.grad_accum

            self.scaler.scale(loss).backward()
            all_loss.append(loss.item() * self.grad_accum)
            all_cls.append(lcls); all_cont.append(lcont)
            all_cos.append(lcos); all_mse.append(lmse)

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            scheduler.step()

        elapsed = (timer() - start) / 60
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1} [{phase}]: loss={np.mean(all_loss):.4f} "
            f"(cls={np.mean(all_cls):.4f} cont={np.mean(all_cont):.4f} "
            f"cos={np.mean(all_cos):.4f} mse={np.mean(all_mse):.4f}) "
            f"lr={lr:.2e} t={elapsed:.1f}min"
        )

    @torch.no_grad()
    def _validate(self, epoch):
        self.mapper.eval()
        # Use classifier logits directly (most reliable during training)
        correct_cls, correct_nn, total = 0, 0, 0
        img_norm = F.normalize(self.img_pooled.float(), dim=-1)
        dtype = next(self.mapper.parameters()).dtype

        for batch in tqdm(self.loaders['val'], desc="Validating", mininterval=10):
            eeg_data  = batch['eeg_data'].to(self.device)
            label_ids = batch['label_ids'].numpy()
            eeg_tokens = self._encode_eeg(eeg_data)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, pred_pooled, class_logits = self.mapper(eeg_tokens.to(dtype))

            # Classifier prediction
            preds_cls = class_logits.float().argmax(dim=-1).cpu().numpy()
            correct_cls += (preds_cls == label_ids).sum()

            # Nearest-neighbor prediction in CLIP image space
            pred_norm = F.normalize(pred_pooled.float(), dim=-1)
            sims = pred_norm @ img_norm.T
            preds_nn = sims.argmax(dim=-1).cpu().numpy()
            correct_nn += (preds_nn == label_ids).sum()

            total += len(label_ids)
            torch.cuda.empty_cache()

        acc_cls = correct_cls / total if total > 0 else 0
        acc_nn  = correct_nn  / total if total > 0 else 0
        print(f"Epoch {epoch+1} Val: cls_acc={acc_cls:.4f} ({correct_cls}/{total})  "
              f"nn_acc={acc_nn:.4f} ({correct_nn}/{total})")

        # Save on classifier accuracy (more reliable)
        if acc_cls > self.best_val_acc:
            self.best_val_acc = acc_cls
            print(f"New best: {acc_cls:.4f} — saving...")
            self._save(epoch + 1)

    @torch.no_grad()
    def _test(self):
        print("=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)

        if self.best_mapper_state:
            self.mapper.load_state_dict(self.best_mapper_state)

        self.mapper.eval()
        img_norm = F.normalize(self.img_pooled.float(), dim=-1)
        dtype = next(self.mapper.parameters()).dtype

        correct_cls, correct_nn, total = 0, 0, 0
        class_correct = defaultdict(int)
        class_total   = defaultdict(int)

        for batch in tqdm(self.loaders['test'], desc="Testing", mininterval=10):
            eeg_data  = batch['eeg_data'].to(self.device)
            label_ids = batch['label_ids'].numpy()
            eeg_tokens = self._encode_eeg(eeg_data)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, pred_pooled, class_logits = self.mapper(eeg_tokens.to(dtype))

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

        print(f"Test cls_acc:  {correct_cls/total:.4f} ({correct_cls}/{total})")
        print(f"Test nn_acc:   {correct_nn/total:.4f}  ({correct_nn}/{total})")
        print("\nPer-class accuracy (classifier):")
        for i, name in enumerate(VI_CLASS_NAMES):
            tot  = class_total[i]
            corr = class_correct[i]
            pct  = corr / tot * 100 if tot > 0 else 0
            print(f"  {name:12s}: {corr:3d}/{tot:3d}  ({pct:.1f}%)")

    def _save(self, epoch):
        os.makedirs(self.args.model_dir, exist_ok=True)
        path = os.path.join(self.args.model_dir, f"mapper_epoch{epoch}.pth")
        torch.save({
            'mapper': self.mapper.state_dict(),
            'token_reducer': self.token_reducer.state_dict(),
            'epoch': epoch,
            'val_acc': self.best_val_acc,
        }, path)
        self.best_mapper_state = copy.deepcopy(self.mapper.state_dict())
        print(f"Saved -> {path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--cuda',             type=int,   default=0)
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--warmup_epochs',    type=int,   default=5)
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--temperature',      type=float, default=0.5)
    parser.add_argument('--lambda_cls',       type=float, default=5.0)
    parser.add_argument('--lambda_contrastive', type=float, default=2.0)
    parser.add_argument('--lambda_cos',       type=float, default=1.0)
    parser.add_argument('--lambda_mse',       type=float, default=0.1)
    # Paths
    parser.add_argument('--datasets_dir',     type=str, default='data/VisualImagery/processed_lmdb')
    parser.add_argument('--stimuli_dir',      type=str, default='data/VisualImagery/stimuli')
    parser.add_argument('--foundation_dir',   type=str, default='pth/CSBrain.pth')
    parser.add_argument('--projection_path',  type=str, default='pth_downtasks/eeg_llm_vi/projection_epoch5.pth')
    parser.add_argument('--model_dir',        type=str, default='pth_downtasks/eeg_direct')
    parser.add_argument('--amused_model_id',  type=str, default='amused/amused-512')
    parser.add_argument('--clip_model_id',    type=str, default='openai/clip-vit-large-patch14')
    # CSBrain
    parser.add_argument('--n_layer',          type=int, default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    # Mapper
    parser.add_argument('--mapper_dim',       type=int, default=512)
    parser.add_argument('--n_transformer_layers', type=int, default=4)
    parser.add_argument('--n_heads',          type=int, default=8)
    parser.add_argument('--dropout',          type=float, default=0.1)

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)

    # ── 1. Precompute CLIP targets ────────────────────────────────────────────
    print("\nBuilding CLIP image targets (stimulus images)...")
    img_builder = CLIPImageTargetBuilder(
        stimuli_dir=args.stimuli_dir,
        clip_model_id=args.clip_model_id,
        device=str(device),
    )
    img_targets_pooled = img_builder.build()   # (10, 768)

    print("\nBuilding CLIP text targets (for sequence MSE)...")
    txt_builder = CLIPTextTargetBuilder(
        amused_model_id=args.amused_model_id, device=str(device)
    )
    text_targets_hidden, _ = txt_builder.build()  # (10, 77, 768)

    # ── 2. Build EEG encoder ─────────────────────────────────────────────────
    print("\nBuilding EEG encoder...")
    encoder, token_reducer = build_eeg_encoder(args, device)

    # ── 3. Build mapper ───────────────────────────────────────────────────────
    mapper = EEGCLIPMapper(
        eeg_dim=200, n_eeg_tokens=20, clip_seq_len=77, clip_dim=768,
        mapper_dim=args.mapper_dim, n_transformer_layers=args.n_transformer_layers,
        n_heads=args.n_heads, dropout=args.dropout, n_classes=10,
    ).to(device)
    print(f"EEGCLIPMapper parameters: {mapper.num_parameters:,}")

    # ── 4. Load data ──────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    loaders = get_data_loaders(args)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    trainer = EEGToImageTrainer(
        args, loaders, encoder, token_reducer, mapper,
        text_targets_hidden, img_targets_pooled,
    )
    trainer.train()
    print("\nDone!")


if __name__ == '__main__':
    main()
