import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle
import random


# FACED 9 emotion classes with multiple paraphrases for training augmentation
FACED_LABEL_MAP = {
    0: [
        "The EEG recording shows patterns consistent with amusement. The subject appears to be experiencing a positive, humorous emotional state with increased frontal alpha asymmetry suggesting approach motivation.",
        "This EEG signal reveals neural signatures of amusement. Frontal lobe activity patterns indicate the subject is in a lighthearted, entertained state with positive valence.",
        "Analysis of the EEG indicates an amused emotional state. The brainwave patterns show characteristic frontal activation associated with humor processing and positive affect.",
    ],
    1: [
        "The EEG recording indicates an inspired emotional state. The neural patterns suggest heightened engagement and positive arousal, with increased beta activity in frontal regions.",
        "This EEG signal shows patterns characteristic of inspiration. Enhanced frontal and central activation suggests the subject is experiencing a state of elevated motivation and creative engagement.",
        "Analysis of the EEG reveals neural signatures of inspiration. The brainwave activity demonstrates increased coherence and frontal beta power consistent with a deeply engaged, positively aroused state.",
    ],
    2: [
        "The EEG recording reveals patterns associated with joy. Strong left frontal activation and coherent alpha rhythms indicate a state of happiness and positive valence.",
        "This EEG signal demonstrates neural patterns typical of joy. The subject shows pronounced left-hemispheric frontal activity characteristic of a happy, high-valence emotional state.",
        "Analysis of the EEG indicates a joyful emotional state. Increased left frontal alpha asymmetry and enhanced gamma coherence are consistent with a state of happiness and elation.",
    ],
    3: [
        "The EEG recording shows neural signatures of tenderness. The patterns suggest a calm, warm emotional state with moderate arousal and positive valence.",
        "This EEG signal reveals patterns consistent with tenderness. The brainwave activity indicates a gentle, caring emotional state with low arousal and positive affect.",
        "Analysis of the EEG demonstrates tenderness. Balanced frontal activity with enhanced theta rhythms suggests a warm, compassionate emotional state with soft positive valence.",
    ],
    4: [
        "The EEG recording shows patterns characteristic of anger. Increased right frontal beta activity and disrupted alpha rhythms indicate a negative, high-arousal emotional state.",
        "This EEG signal reveals neural signatures of anger. Right-lateralized frontal activation and elevated beta power are consistent with an agitated, negatively valenced state.",
        "Analysis of the EEG indicates anger. The brainwave patterns show heightened right frontal asymmetry and increased high-frequency activity characteristic of hostile, high-arousal negative emotion.",
    ],
    5: [
        "The EEG recording indicates a state of disgust. The neural patterns show increased theta activity in frontal regions and right hemispheric dominance associated with withdrawal motivation.",
        "This EEG signal demonstrates patterns typical of disgust. Frontal theta enhancement and right-lateralized activity suggest the subject is experiencing aversion and negative affect.",
        "Analysis of the EEG reveals disgust. Increased frontal midline theta and right hemispheric activation indicate a withdrawal-oriented state with strong negative valence.",
    ],
    6: [
        "The EEG recording reveals patterns consistent with fear. Heightened beta and gamma activity across frontal and temporal regions indicate high arousal with negative valence.",
        "This EEG signal shows neural signatures of fear. Widespread high-frequency activation in frontal and temporal areas suggests a state of threat detection and anxious arousal.",
        "Analysis of the EEG indicates fear. The brainwave patterns demonstrate enhanced beta-gamma power and disrupted alpha rhythms consistent with a highly aroused, negatively valenced defensive state.",
    ],
    7: [
        "The EEG recording shows neural signatures of sadness. Increased right frontal alpha power and reduced left frontal activity suggest a low-arousal negative emotional state.",
        "This EEG signal reveals patterns associated with sadness. Right-lateralized frontal alpha asymmetry indicates the subject is experiencing a withdrawn, melancholic emotional state.",
        "Analysis of the EEG demonstrates sadness. The brainwave activity shows characteristic right frontal dominance and reduced overall activation consistent with a low-arousal state of sorrow.",
    ],
    8: [
        "The EEG recording displays patterns associated with a neutral emotional state. Balanced bilateral alpha activity and moderate arousal levels suggest the absence of strong emotional engagement.",
        "This EEG signal shows a neutral baseline pattern. Symmetric alpha distribution and stable arousal indicators suggest the subject is in a calm, emotionally non-activated state.",
        "Analysis of the EEG indicates a neutral emotional state. The brainwave patterns demonstrate balanced hemispheric activity and typical resting-state alpha rhythms without significant emotional modulation.",
    ],
}

# Emotion keywords for evaluation (matching generated text to ground truth)
EMOTION_KEYWORDS = {
    0: ["amusement", "amused", "humor", "entertained", "lighthearted"],
    1: ["inspiration", "inspired", "motivation", "creative", "engaged"],
    2: ["joy", "joyful", "happiness", "happy", "elation"],
    3: ["tenderness", "tender", "warm", "caring", "compassionate", "gentle"],
    4: ["anger", "angry", "agitated", "hostile"],
    5: ["disgust", "disgusted", "aversion"],
    6: ["fear", "fearful", "threat", "anxious", "defensive"],
    7: ["sadness", "sad", "melancholic", "sorrow", "withdrawn"],
    8: ["neutral", "baseline", "calm", "non-activated"],
}

SYSTEM_PROMPT = "You are an expert EEG analyst specializing in emotion recognition from brain signals. Analyze the provided EEG recording and describe the emotional state of the subject."
USER_PROMPT = "Analyze this EEG recording and describe the subject's emotional state in detail, including the neural patterns observed."


class FACEDLLMDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.mode = mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = int(pair['label'])
        return data / 100, label


class FACEDLLMCollator:
    """Collate function that builds tokenized prompts and targets for the EEG-LLM model."""

    def __init__(self, tokenizer, max_target_len=128, mode='train'):
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.mode = mode

        # Build the prompt text (same for all samples)
        # TinyLlama chat template (Zephyr format)
        self.prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n[EEG_TOKENS]\n{USER_PROMPT}</s>\n"
            f"<|assistant|>\n"
        )

        # Pre-tokenize the prompt (it's the same for every sample)
        prompt_encoded = self.tokenizer(
            self.prompt_text, return_tensors="pt",
            add_special_tokens=False, padding=False
        )
        self.prompt_ids = prompt_encoded['input_ids'].squeeze(0)       # (prompt_len,)
        self.prompt_mask = prompt_encoded['attention_mask'].squeeze(0)  # (prompt_len,)

    def __call__(self, batch):
        eeg_data = np.array([x[0] for x in batch])
        labels = [x[1] for x in batch]
        batch_size = len(batch)

        # Build target texts
        target_texts = []
        for label_id in labels:
            paraphrases = FACED_LABEL_MAP[label_id]
            if self.mode == 'train':
                text = random.choice(paraphrases)
            else:
                text = paraphrases[0]  # deterministic for val/test
            target_texts.append(text + "</s>")

        # Tokenize targets with padding
        target_encoded = self.tokenizer(
            target_texts, return_tensors="pt",
            add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_target_len,
        )
        target_ids = target_encoded['input_ids']      # (batch, target_len)
        target_mask = target_encoded['attention_mask']  # (batch, target_len)

        # Expand prompt to batch
        prompt_ids = self.prompt_ids.unsqueeze(0).expand(batch_size, -1)
        prompt_mask = self.prompt_mask.unsqueeze(0).expand(batch_size, -1)

        return {
            'eeg_data': to_tensor(eeg_data),
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'label_ids': torch.tensor(labels, dtype=torch.long),
        }


class LoadDataset:
    def __init__(self, params, tokenizer):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.tokenizer = tokenizer
        self.max_target_len = getattr(params, 'max_target_len', 128)

    def get_data_loader(self):
        train_set = FACEDLLMDataset(self.datasets_dir, mode='train')
        val_set = FACEDLLMDataset(self.datasets_dir, mode='val')
        test_set = FACEDLLMDataset(self.datasets_dir, mode='test')
        print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        train_collator = FACEDLLMCollator(self.tokenizer, self.max_target_len, mode='train')
        eval_collator = FACEDLLMCollator(self.tokenizer, self.max_target_len, mode='eval')

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_collator,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=eval_collator,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=eval_collator,
                shuffle=False,
            ),
        }
        return data_loader
