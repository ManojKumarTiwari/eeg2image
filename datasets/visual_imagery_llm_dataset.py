"""
EEG Visual Imagery dataset for EEG-to-text generation.

10 classes across 3 categories:
  Animals (0-2): dog, bird, fish
  Figures (3-5): pentagram, square, circle
  Objects (6-9): scissor, watch, cup, chair

Sample shape: (32, 4, 200) — 32 channels, 4 patches, 200 samples/patch
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle
import random


# ─── Label text descriptions ─────────────────────────────────────────────────

VI_LABEL_MAP = {
    # Animals
    0: [
        "The subject is imagining a dog.",
        "The image being imagined is a dog.",
        "A dog is being mentally visualized.",
    ],
    1: [
        "The subject is imagining a bird.",
        "The image being imagined is a bird.",
        "A bird is being mentally visualized.",
    ],
    2: [
        "The subject is imagining a fish.",
        "The image being imagined is a fish.",
        "A fish is being mentally visualized.",
    ],
    # Figures
    3: [
        "The subject is imagining a pentagram.",
        "The image being imagined is a pentagram.",
        "A pentagram is being mentally visualized.",
    ],
    4: [
        "The subject is imagining a square.",
        "The image being imagined is a square.",
        "A square is being mentally visualized.",
    ],
    5: [
        "The subject is imagining a circle.",
        "The image being imagined is a circle.",
        "A circle is being mentally visualized.",
    ],
    # Objects
    6: [
        "The subject is imagining scissors.",
        "The image being imagined is scissors.",
        "Scissors are being mentally visualized.",
    ],
    7: [
        "The subject is imagining a watch.",
        "The image being imagined is a watch.",
        "A watch is being mentally visualized.",
    ],
    8: [
        "The subject is imagining a cup.",
        "The image being imagined is a cup.",
        "A cup is being mentally visualized.",
    ],
    9: [
        "The subject is imagining a chair.",
        "The image being imagined is a chair.",
        "A chair is being mentally visualized.",
    ],
}

# Keywords for evaluation by keyword matching
VI_KEYWORDS = {
    0: ["dog"],
    1: ["bird"],
    2: ["fish"],
    3: ["pentagram"],
    4: ["square"],
    5: ["circle"],
    6: ["scissor", "scissors"],
    7: ["watch"],
    8: ["cup"],
    9: ["chair"],
}

SYSTEM_PROMPT = (
    "You are a brain-computer interface system that decodes visual imagery from EEG signals. "
    "Identify what image the subject is imagining."
)
USER_PROMPT = (
    "Based on this EEG recording, what image is the subject imagining?"
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VisualImageryLLMDataset(Dataset):
    def __init__(self, data_dir, mode='train', db=None):
        super().__init__()
        self._owns_db = db is None
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False) if db is None else db
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.mode = mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data  = pair['sample']   # (32, 4, 200)
        label = int(pair['label'])
        return data, label


# ─── Collator ────────────────────────────────────────────────────────────────

class VisualImageryLLMCollator:
    """Builds tokenized prompt + target for the EEG-LLM model."""

    def __init__(self, tokenizer, max_target_len=128, mode='train'):
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.mode = mode

        self.prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n[EEG_TOKENS]\n{USER_PROMPT}</s>\n"
            f"<|assistant|>\n"
        )

        prompt_encoded = self.tokenizer(
            self.prompt_text, return_tensors="pt",
            add_special_tokens=False, padding=False
        )
        self.prompt_ids  = prompt_encoded['input_ids'].squeeze(0)
        self.prompt_mask = prompt_encoded['attention_mask'].squeeze(0)

    def __call__(self, batch):
        eeg_data   = np.array([x[0] for x in batch])
        labels     = [x[1] for x in batch]
        batch_size = len(batch)

        target_texts = []
        for label_id in labels:
            paraphrases = VI_LABEL_MAP[label_id]
            text = random.choice(paraphrases) if self.mode == 'train' else paraphrases[0]
            target_texts.append(text + "</s>")

        target_encoded = self.tokenizer(
            target_texts, return_tensors="pt",
            add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_target_len,
        )

        prompt_ids  = self.prompt_ids.unsqueeze(0).expand(batch_size, -1)
        prompt_mask = self.prompt_mask.unsqueeze(0).expand(batch_size, -1)

        return {
            'eeg_data':    to_tensor(eeg_data),
            'prompt_ids':  prompt_ids,
            'prompt_mask': prompt_mask,
            'target_ids':  target_encoded['input_ids'],
            'target_mask': target_encoded['attention_mask'],
            'label_ids':   torch.tensor(labels, dtype=torch.long),
        }


# ─── Loader ──────────────────────────────────────────────────────────────────

class LoadDataset:
    def __init__(self, params, tokenizer):
        self.params         = params
        self.datasets_dir   = params.datasets_dir
        self.tokenizer      = tokenizer
        self.max_target_len = getattr(params, 'max_target_len', 128)

    def get_data_loader(self):
        shared_db = lmdb.open(self.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False)
        train_set = VisualImageryLLMDataset(self.datasets_dir, mode='train', db=shared_db)
        val_set   = VisualImageryLLMDataset(self.datasets_dir, mode='val',   db=shared_db)
        test_set  = VisualImageryLLMDataset(self.datasets_dir, mode='test',  db=shared_db)
        print(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        train_collator = VisualImageryLLMCollator(self.tokenizer, self.max_target_len, mode='train')
        eval_collator  = VisualImageryLLMCollator(self.tokenizer, self.max_target_len, mode='eval')

        return {
            'train': DataLoader(train_set, batch_size=self.params.batch_size,
                                collate_fn=train_collator, shuffle=True),
            'val':   DataLoader(val_set,   batch_size=self.params.batch_size,
                                collate_fn=eval_collator,  shuffle=False),
            'test':  DataLoader(test_set,  batch_size=self.params.batch_size,
                                collate_fn=eval_collator,  shuffle=False),
        }
