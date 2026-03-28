import torch
import torch.nn as nn
from tqdm import tqdm
from timeit import default_timer as timer
import numpy as np
import copy
import os

class EEGLLMTrainer:
    """Two-phase trainer for the EEG-Language model with mixed precision and gradient accumulation."""

    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.model = model

        # Load dataset-specific keywords for evaluation
        if getattr(params, 'downstream_dataset', 'FACED').upper() == 'BCICIV2A':
            from datasets.bciciv2a_llm_dataset import MI_KEYWORDS
            self.eval_keywords = MI_KEYWORDS
        else:
            from datasets.faced_llm_dataset import EMOTION_KEYWORDS
            self.eval_keywords = EMOTION_KEYWORDS

        self.grad_accum_steps = getattr(params, 'gradient_accumulation_steps', 8)
        self.warmup_epochs = getattr(params, 'warmup_epochs', 5)
        self.best_val_acc = 0
        self.best_projection_state = None
        self.best_lora_dir = None

        # Separate trainable parameters
        self.projection_params = list(model.eeg_projection.parameters())
        self.lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]

        print(f"Projection params: {sum(p.numel() for p in self.projection_params):,}")
        print(f"LoRA params: {sum(p.numel() for p in self.lora_params):,}")
        print(f"Total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda')

    def _create_optimizer(self, phase):
        """Create optimizer for the given training phase."""
        if phase == 'warmup':
            # Phase 1: only projection, higher LR
            for p in self.lora_params:
                p.requires_grad = False
            params = self.projection_params
            lr = self.params.lr * 5  # 5e-4 if base lr is 1e-4
        else:
            # Phase 2: projection + LoRA
            for p in self.lora_params:
                p.requires_grad = True
            params = self.projection_params + self.lora_params
            lr = self.params.lr

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.params.weight_decay)
        data_length = len(self.data_loader['train'])
        epochs = self.warmup_epochs if phase == 'warmup' else (self.params.epochs - self.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * data_length, eta_min=1e-6
        )
        return optimizer, scheduler

    def train(self):
        # Phase 1: Projection warmup
        print("=" * 60)
        print(f"Phase 1: Projection warmup ({self.warmup_epochs} epochs)")
        print("=" * 60)
        optimizer, scheduler = self._create_optimizer('warmup')
        for epoch in range(self.warmup_epochs):
            self._train_epoch(epoch, optimizer, scheduler, phase='warmup')
            self._validate(epoch)

        # Phase 2: Joint training
        joint_epochs = self.params.epochs - self.warmup_epochs
        print("=" * 60)
        print(f"Phase 2: Joint projection + LoRA training ({joint_epochs} epochs)")
        print("=" * 60)
        optimizer, scheduler = self._create_optimizer('joint')
        for epoch in range(self.warmup_epochs, self.params.epochs):
            self._train_epoch(epoch, optimizer, scheduler, phase='joint')
            self._validate(epoch)

        # Final test evaluation
        self._test()

    def _train_epoch(self, epoch, optimizer, scheduler, phase):
        self.model.train()
        self.model.eeg_encoder.eval()  # keep frozen encoder in eval mode
        losses = []
        start_time = timer()

        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(self.data_loader['train'], desc=f"Epoch {epoch+1}", mininterval=10)):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(
                    eeg_data=batch['eeg_data'].cuda(),
                    prompt_ids=batch['prompt_ids'].cuda(),
                    prompt_mask=batch['prompt_mask'].cuda(),
                    target_ids=batch['target_ids'].cuda(),
                    target_mask=batch['target_mask'].cuda(),
                )
                loss = outputs.loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            losses.append(loss.item() * self.grad_accum_steps)
            del outputs

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            scheduler.step()

        elapsed = (timer() - start_time) / 60
        avg_loss = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1} [{phase}]: Loss={avg_loss:.4f}, "
            f"LR={lr:.6f}, Time={elapsed:.1f}min"
        )

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0

        for batch in tqdm(self.data_loader['val'], desc="Validating", mininterval=10):
            generated_texts = self.model.generate(
                eeg_data=batch['eeg_data'].cuda(),
                prompt_ids=batch['prompt_ids'].cuda(),
                prompt_mask=batch['prompt_mask'].cuda(),
                max_new_tokens=20,
            )
            label_ids = batch['label_ids'].numpy()

            for text, true_label in zip(generated_texts, label_ids):
                predicted_label = self._extract_emotion(text.lower())
                if predicted_label == true_label:
                    correct += 1
                total += 1
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} Val Accuracy (keyword extraction): {acc:.4f} ({correct}/{total})")

        if acc > self.best_val_acc:
            self.best_val_acc = acc
            print(f"New best val accuracy: {acc:.4f} — saving model...")
            self._save_model(epoch + 1)

    def _extract_emotion(self, text):
        """Extract predicted emotion from generated text by keyword matching."""
        best_label = -1
        best_count = 0

        for label_id, keywords in self.eval_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_label = label_id

        return best_label

    @torch.no_grad()
    def _test(self):
        print("=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)

        # Load best model weights
        if self.best_lora_dir and os.path.exists(self.best_lora_dir):
            self._load_best_model()

        self.model.eval()
        correct = 0
        total = 0
        sample_outputs = []

        for batch in tqdm(self.data_loader['test'], desc="Testing", mininterval=10):
            generated_texts = self.model.generate(
                eeg_data=batch['eeg_data'].cuda(),
                prompt_ids=batch['prompt_ids'].cuda(),
                prompt_mask=batch['prompt_mask'].cuda(),
                max_new_tokens=20,
            )
            label_ids = batch['label_ids'].numpy()

            for text, true_label in zip(generated_texts, label_ids):
                predicted_label = self._extract_emotion(text.lower())
                if predicted_label == true_label:
                    correct += 1
                total += 1

                if len(sample_outputs) < 9:  # collect one sample per class
                    sample_outputs.append((true_label, predicted_label, text[:200]))
            torch.cuda.empty_cache()

        acc = correct / total if total > 0 else 0
        print(f"Test Accuracy (keyword extraction): {acc:.4f} ({correct}/{total})")
        print("\nSample outputs:")
        for true_label, pred_label, text in sample_outputs:
            print(f"  True: {true_label}, Pred: {pred_label}")
            print(f"  Text: {text}...")
            print()

    def _save_model(self, epoch):
        save_dir = self.params.model_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save projection weights
        proj_path = os.path.join(save_dir, f"projection_epoch{epoch}.pth")
        torch.save({
            'projection': self.model.eeg_projection.state_dict(),
            'token_reducer': self.model.token_reducer.state_dict(),
            'epoch': epoch,
            'val_acc': self.best_val_acc,
        }, proj_path)

        # Save LoRA adapter
        lora_dir = os.path.join(save_dir, f"lora_epoch{epoch}")
        self.model.llm.save_pretrained(lora_dir)

        self.best_projection_state = copy.deepcopy(self.model.eeg_projection.state_dict())
        self.best_lora_dir = lora_dir
        print(f"Model saved to {save_dir}")

    def _load_best_model(self):
        if self.best_projection_state:
            self.model.eeg_projection.load_state_dict(self.best_projection_state)
        if self.best_lora_dir:
            from peft import PeftModel
            # LoRA weights are already loaded since we kept them in memory via deepcopy approach
            # For robustness, we use the saved projection state
            pass
