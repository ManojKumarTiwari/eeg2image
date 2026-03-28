"""
BCIC-IV-2a dataset for EEG-to-text generation.

Returns batches in the same format as faced_llm_dataset.py so the same
trainer and model work for both datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle
import random


# ─── Label text descriptions ─────────────────────────────────────────────────

BCIC_LABEL_MAP = {
    0: [
        "The EEG recording shows motor imagery patterns consistent with left hand movement. Contralateral right hemisphere activation is observed, particularly in central and sensorimotor regions with characteristic mu and beta rhythm desynchronization.",
        "This EEG signal reveals neural signatures of left hand motor imagery. Event-related desynchronization in the right central regions (C4) indicates the subject is imagining grasping or moving their left hand.",
        "Analysis of the EEG indicates left hand motor imagery. The brainwave patterns show right-lateralized sensorimotor activity with suppressed beta oscillations, consistent with imagined left hand motor execution.",
    ],
    1: [
        "The EEG recording shows motor imagery patterns consistent with right hand movement. Contralateral left hemisphere activation is observed in central and sensorimotor regions with mu and beta rhythm desynchronization.",
        "This EEG signal reveals neural signatures of right hand motor imagery. Event-related desynchronization in the left central regions (C3) indicates the subject is imagining grasping or moving their right hand.",
        "Analysis of the EEG indicates right hand motor imagery. The brainwave patterns show left-lateralized sensorimotor activity with suppressed beta oscillations, consistent with imagined right hand motor execution.",
    ],
    2: [
        "The EEG recording shows motor imagery patterns consistent with feet movement. Bilateral central midline activation is observed with vertex (CZ) and central parietal desynchronization characteristic of lower limb motor imagery.",
        "This EEG signal reveals neural signatures of feet motor imagery. Midline sensorimotor activity with strong CZ and CPZ involvement indicates the subject is imagining movement of both feet or lower limbs.",
        "Analysis of the EEG indicates feet motor imagery. The brainwave patterns show bilateral symmetrical sensorimotor desynchronization at central midline electrodes, consistent with imagined bipedal motor activity.",
    ],
    3: [
        "The EEG recording shows motor imagery patterns consistent with tongue movement. Activation in bilateral central and frontocentral regions is observed, with characteristic orofacial motor cortex engagement.",
        "This EEG signal reveals neural signatures of tongue motor imagery. Frontocentral and bilateral central activation indicates the subject is imagining articulation or tongue movement involving orofacial motor areas.",
        "Analysis of the EEG indicates tongue motor imagery. The brainwave patterns show central and frontocentral desynchronization consistent with imagined oral motor execution and tongue articulation.",
    ],
}

# Keywords for evaluation by keyword matching
MI_KEYWORDS = {
    0: ["left hand", "left motor", "left hemisphere", "right hemisphere activation", "left hand movement", "left"],
    1: ["right hand", "right motor", "right hemisphere", "left hemisphere activation", "right hand movement", "right"],
    2: ["feet", "foot", "lower limb", "bilateral", "midline", "bipedal"],
    3: ["tongue", "oral", "articulation", "orofacial", "mouth"],
}

SYSTEM_PROMPT = (
    "You are an expert EEG analyst specializing in motor imagery decoding from brain signals. "
    "Analyze the provided EEG recording and describe the motor imagery task being performed."
)
USER_PROMPT = (
    "Analyze this EEG recording and describe which motor imagery task the subject is performing, "
    "including the observed neural patterns and activated brain regions."
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BCICIV2aLLMDataset(Dataset):
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
        data = pair['sample']      # (22, 4, 200), already /100 during preprocessing
        label = int(pair['label'])
        return data, label


# ─── Collator ────────────────────────────────────────────────────────────────

class BCICIV2aLLMCollator:
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
        self.prompt_ids = prompt_encoded['input_ids'].squeeze(0)
        self.prompt_mask = prompt_encoded['attention_mask'].squeeze(0)

    def __call__(self, batch):
        eeg_data = np.array([x[0] for x in batch])
        labels = [x[1] for x in batch]
        batch_size = len(batch)

        target_texts = []
        for label_id in labels:
            paraphrases = BCIC_LABEL_MAP[label_id]
            text = random.choice(paraphrases) if self.mode == 'train' else paraphrases[0]
            target_texts.append(text + "</s>")

        target_encoded = self.tokenizer(
            target_texts, return_tensors="pt",
            add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_target_len,
        )

        prompt_ids = self.prompt_ids.unsqueeze(0).expand(batch_size, -1)
        prompt_mask = self.prompt_mask.unsqueeze(0).expand(batch_size, -1)

        return {
            'eeg_data': to_tensor(eeg_data),
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'target_ids': target_encoded['input_ids'],
            'target_mask': target_encoded['attention_mask'],
            'label_ids': torch.tensor(labels, dtype=torch.long),
        }


# ─── Loader ──────────────────────────────────────────────────────────────────

class LoadDataset:
    def __init__(self, params, tokenizer):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.tokenizer = tokenizer
        self.max_target_len = getattr(params, 'max_target_len', 128)

    def get_data_loader(self):
        shared_db = lmdb.open(self.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False)
        train_set = BCICIV2aLLMDataset(self.datasets_dir, mode='train', db=shared_db)
        val_set   = BCICIV2aLLMDataset(self.datasets_dir, mode='val',   db=shared_db)
        test_set  = BCICIV2aLLMDataset(self.datasets_dir, mode='test',  db=shared_db)
        print(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        train_collator = BCICIV2aLLMCollator(self.tokenizer, self.max_target_len, mode='train')
        eval_collator  = BCICIV2aLLMCollator(self.tokenizer, self.max_target_len, mode='eval')

        return {
            'train': DataLoader(train_set, batch_size=self.params.batch_size,
                                collate_fn=train_collator, shuffle=True),
            'val':   DataLoader(val_set,   batch_size=self.params.batch_size,
                                collate_fn=eval_collator,  shuffle=False),
            'test':  DataLoader(test_set,  batch_size=self.params.batch_size,
                                collate_fn=eval_collator,  shuffle=False),
        }
