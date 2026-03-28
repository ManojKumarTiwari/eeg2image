import argparse
import random
import numpy as np
import torch

from finetune_eeg_llm_trainer import EEGLLMTrainer
from models.eeg_llm import EEGLanguageModel


def main():
    parser = argparse.ArgumentParser(description='EEG-LLM Fine-tuning: CSBrain Encoder + LLaMA Decoder')

    # General training args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Dataset args
    parser.add_argument('--downstream_dataset', type=str, default='FACED')
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, default=9)
    parser.add_argument('--model_dir', type=str, default='pth_downtasks/eeg_llm_faced')

    # CSBrain encoder args
    parser.add_argument('--model', type=str, default='CSBrain')
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true')
    parser.add_argument('--foundation_dir', type=str, default='pth/CSBrain.pth')

    # LLM args
    parser.add_argument('--llm_model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--llm_dim', type=int, default=2048)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--max_target_len', type=int, default=128)

    # Training strategy args
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--temporal_pool_stride', type=int, default=2)

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)

    print(f"Building EEG-LLM model with {params.llm_model_name}...")
    model = EEGLanguageModel(params)

    print(f"Loading {params.downstream_dataset} dataset with LLM text labels...")
    if params.downstream_dataset.upper() == 'BCICIV2A':
        from datasets import bciciv2a_llm_dataset
        load_dataset = bciciv2a_llm_dataset.LoadDataset(params, model.tokenizer)
    else:
        from datasets import faced_llm_dataset
        load_dataset = faced_llm_dataset.LoadDataset(params, model.tokenizer)
    data_loader = load_dataset.get_data_loader()

    print("Starting EEG-LLM training...")
    trainer = EEGLLMTrainer(params, data_loader, model)
    trainer.train()

    print("Done!")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
