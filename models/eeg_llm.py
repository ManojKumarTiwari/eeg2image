import torch
import torch.nn as nn
from .CSBrain import CSBrain


# ─── BCIC-IV-2a brain region config (22 channels, 3 regions) ──────────────────

BCIC_BRAIN_REGIONS = [
    0,                                # Fz        → Frontal
    4, 4, 4, 4, 4,                    # FC3-FC4   → Central
    4, 4, 4, 4, 4, 4, 4,             # C5-C6     → Central
    4, 4, 4, 4, 4,                    # CP3-CP4   → Central
    1, 1, 1, 1,                       # P1,PZ,P2,POZ → Parietal
]
BCIC_ELECTRODE_LABELS = [
    "Fz",
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
]
BCIC_TOPOLOGY = {
    0: ["Fz"],
    4: ["FC3", "FC1", "FCZ", "FC2", "FC4",
        "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
        "CP3", "CP1", "CPZ", "CP2", "CP4"],
    1: ["P1", "PZ", "P2", "POZ"],
}

# ─── FACED brain region config (30 channels, 5 regions) ───────────────────────

FACED_BRAIN_REGIONS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 4, 4, 2, 2, 4, 4, 4, 4,
    1, 1, 1, 1, 1,
    3, 3, 3, 3, 3,
]
FACED_ELECTRODE_LABELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8",
    "PO3", "PO4", "Oz", "O1", "O2",
]
FACED_TOPOLOGY = {
    0: ["Fp1", "FC5", "FC1", "F7", "F3", "Fz", "F4", "F8", "FC2", "FC6", "Fp2"],
    4: ["CP5", "CP1", "C3", "Cz", "C4", "CP2", "CP6"],
    1: ["P7", "P3", "Pz", "P4", "P8"],
    2: ["T7", "T8"],
    3: ["PO3", "O1", "Oz", "O2", "PO4"],
}


def _build_sorted_indices(brain_regions, electrode_labels, topology):
    region_groups = {}
    for i, region in enumerate(brain_regions):
        region_groups.setdefault(region, []).append((i, electrode_labels[i]))
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        sorted_electrodes = sorted(
            region_groups[region],
            key=lambda x: topology[region].index(x[1])
        )
        sorted_indices.extend([e[0] for e in sorted_electrodes])
    return sorted_indices


class EEGTokenReducer(nn.Module):
    """Reduces EEG tokens by pooling channels within brain regions and optionally pooling temporal patches."""

    def __init__(self, area_config, temporal_pool_stride=1):
        super().__init__()
        self.area_config = area_config
        self.temporal_pool_stride = temporal_pool_stride

    def forward(self, x):
        # x: (batch, n_channels, n_patches, d_model) e.g. (batch, 30, 30, 200)
        batch, n_ch, n_patches, d_model = x.shape
        region_tokens = []

        for region_name in sorted(self.area_config.keys()):
            s = self.area_config[region_name]['slice']
            region_x = x[:, s, :, :]           # (batch, region_ch, n_patches, d_model)
            region_avg = region_x.mean(dim=1)   # (batch, n_patches, d_model)
            region_tokens.append(region_avg)

        n_regions = len(region_tokens)
        # Stack into (batch, n_regions, n_patches, d_model) then reshape
        tokens = torch.stack(region_tokens, dim=1)  # (batch, 5, 30, 200)

        if self.temporal_pool_stride > 1:
            n_out = n_patches // self.temporal_pool_stride
            tokens = tokens[:, :, :n_out * self.temporal_pool_stride, :]
            tokens = tokens.view(batch, n_regions, n_out, self.temporal_pool_stride, d_model)
            tokens = tokens.mean(dim=3)  # (batch, 5, 15, 200)

        tokens = tokens.view(batch, -1, d_model)  # (batch, 75, 200)
        return tokens


class EEGProjection(nn.Module):
    """Projects EEG tokens from CSBrain dim to LLM hidden dim via 2-layer MLP."""

    def __init__(self, eeg_dim=200, llm_dim=2048, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(eeg_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(x)  # (batch, n_tokens, eeg_dim) -> (batch, n_tokens, llm_dim)


class EEGLanguageModel(nn.Module):
    """Multimodal EEG-to-Language model: CSBrain encoder + projection + LLaMA decoder with LoRA."""

    def __init__(self, param):
        super().__init__()

        # Select brain region config based on dataset
        dataset = getattr(param, 'downstream_dataset', 'FACED').upper()
        if dataset == 'BCICIV2A':
            brain_regions    = BCIC_BRAIN_REGIONS
            electrode_labels = BCIC_ELECTRODE_LABELS
            topology         = BCIC_TOPOLOGY
            self._n_channels = 22
            temporal_pool_stride = 1   # BCIC only has 4 patches, no pooling needed
        else:  # FACED
            brain_regions    = FACED_BRAIN_REGIONS
            electrode_labels = FACED_ELECTRODE_LABELS
            topology         = FACED_TOPOLOGY
            self._n_channels = 30
            temporal_pool_stride = getattr(param, 'temporal_pool_stride', 2)

        sorted_indices = _build_sorted_indices(brain_regions, electrode_labels, topology)

        # ---- 1. CSBrain Encoder (frozen) ----
        self.eeg_encoder = CSBrain(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=param.n_layer, nhead=8,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices,
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model_state_dict = self.eeg_encoder.state_dict()
            matching_dict = {
                k: v for k, v in new_state_dict.items()
                if k in model_state_dict and v.size() == model_state_dict[k].size()
            }
            model_state_dict.update(matching_dict)
            self.eeg_encoder.load_state_dict(model_state_dict)
            print(f"Loaded {len(matching_dict)}/{len(model_state_dict)} pretrained weights into CSBrain encoder")

        self.eeg_encoder.proj_out = nn.Identity()

        # Freeze encoder
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False

        # ---- 2. Token Reducer ----
        self.token_reducer = EEGTokenReducer(
            area_config=self.eeg_encoder.area_config,
            temporal_pool_stride=temporal_pool_stride,
        )

        # ---- 3. Projection MLP ----
        llm_dim = getattr(param, 'llm_dim', 2048)
        self.eeg_projection = EEGProjection(
            eeg_dim=200, llm_dim=llm_dim, dropout=param.dropout
        )

        # ---- 4. LLM Decoder (4-bit quantized + LoRA) ----
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        llm_model_name = getattr(param, 'llm_model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.llm = prepare_model_for_kbit_training(self.llm)

        lora_rank = getattr(param, 'lora_rank', 8)
        lora_alpha = getattr(param, 'lora_alpha', 16)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move EEG components to CUDA (LLM uses device_map="auto" separately)
        device = torch.device(f'cuda:{param.cuda}')
        self.eeg_encoder.to(device)
        self.token_reducer.to(device)
        self.eeg_projection.to(device)

    def get_text_embeddings(self, token_ids):
        """Get embeddings from the LLM's embedding layer."""
        return self.llm.get_input_embeddings()(token_ids)

    def forward(self, eeg_data, prompt_ids, prompt_mask, target_ids, target_mask):
        """
        Args:
            eeg_data:    (batch, channels, patches, patch_size) raw EEG
            prompt_ids:  (batch, prompt_len) tokenized instruction text
            prompt_mask: (batch, prompt_len) attention mask for prompt
            target_ids:  (batch, target_len) tokenized target text
            target_mask: (batch, target_len) attention mask for target
        Returns:
            CausalLMOutput with .loss and .logits
        """
        batch_size = eeg_data.shape[0]

        # 1. EEG encoding (no grad — encoder is frozen, must run in float32)
        with torch.no_grad():
            self.eeg_encoder.eval()
            with torch.amp.autocast('cuda', enabled=False):
                eeg_features = self.eeg_encoder(eeg_data[:, :self._n_channels, :, :].float())

        # 2. Token reduction → (batch, 75, 200)
        eeg_tokens = self.token_reducer(eeg_features)

        # 3. Project to LLM space → (batch, 75, llm_dim)
        eeg_embeds = self.eeg_projection(eeg_tokens.to(self.eeg_projection.proj[0].weight.dtype))

        # 4. Get text embeddings
        prompt_embeds = self.get_text_embeddings(prompt_ids)   # (batch, prompt_len, llm_dim)
        target_embeds = self.get_text_embeddings(target_ids)   # (batch, target_len, llm_dim)

        # 5. Concatenate: [prompt | eeg | target]
        inputs_embeds = torch.cat([prompt_embeds, eeg_embeds, target_embeds], dim=1)

        # 6. Build attention mask
        eeg_attn_mask = torch.ones(
            batch_size, eeg_tokens.shape[1],
            device=eeg_data.device, dtype=prompt_mask.dtype
        )
        attention_mask = torch.cat([prompt_mask, eeg_attn_mask, target_mask], dim=1)

        # 7. Build labels: -100 for prompt + eeg positions, actual ids for target
        ignore_len = prompt_ids.shape[1] + eeg_tokens.shape[1]
        ignore_labels = torch.full(
            (batch_size, ignore_len), -100,
            device=eeg_data.device, dtype=target_ids.dtype
        )
        labels = torch.cat([ignore_labels, target_ids], dim=1)
        # Also mask padding tokens in target
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        # 8. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(self, eeg_data, prompt_ids, prompt_mask, max_new_tokens=128):
        """Generate text from EEG data at inference time."""
        self.eeg_encoder.eval()
        with torch.amp.autocast('cuda', enabled=False):
            eeg_features = self.eeg_encoder(eeg_data[:, :self._n_channels, :, :].float())
        eeg_tokens = self.token_reducer(eeg_features)
        eeg_embeds = self.eeg_projection(eeg_tokens.to(self.eeg_projection.proj[0].weight.dtype))

        prompt_embeds = self.get_text_embeddings(prompt_ids)

        inputs_embeds = torch.cat([prompt_embeds, eeg_embeds], dim=1)
        eeg_attn_mask = torch.ones(
            eeg_data.shape[0], eeg_tokens.shape[1],
            device=eeg_data.device, dtype=prompt_mask.dtype
        )
        attention_mask = torch.cat([prompt_mask, eeg_attn_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
