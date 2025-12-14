# train_all_models.py
import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm
from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader

from gpt_model import GPT, GPTConfig


# =============================================================
# 1. HuggingFace dataset config
# =============================================================
HF_DATASET_NAME = "AdrianRasoOnHF/wikidump-en-2025-07"
HF_DATASET_CONFIG = None
HF_TEXT_COLUMN = "text"
HF_SPLIT = "train"


# =============================================================
# Load HF dataset
# =============================================================
def load_hf_dataset():
    print(f"Loading dataset: {HF_DATASET_NAME}")
    ds = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        split=HF_SPLIT,
    )
    if HF_TEXT_COLUMN not in ds.column_names:
        raise ValueError(
            f"Column `{HF_TEXT_COLUMN}` missing. Columns: {ds.column_names}"
        )
    return ds


# =============================================================
# Train SentencePiece tokenizer
# =============================================================
def train_sentencepiece_tokenizer_hf(
    ds, model_prefix="wiki_bpe", vocab_size=16000, sample_size=5_000_000
):
    """
    Sample dataset text and train a SentencePiece BPE tokenizer.
    """
    tmp_sample_path = "spm_sample.txt"
    collected = 0

    print("Sampling text for tokenizer training…")
    with open(tmp_sample_path, "w") as f:
        for row in ds.shuffle(seed=42):
            txt = row[HF_TEXT_COLUMN].replace("\n", " ")
            f.write(txt + "\n")
            collected += len(txt)
            if collected >= sample_size:
                break

    print(f"Sampler collected {collected} characters.")

    spm.SentencePieceTrainer.Train(
        f"--input={tmp_sample_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage=1.0 "
        f"--model_type=bpe "
        f"--max_sentence_length=2048"
    )

    print("Tokenizer trained successfully.")
    return f"{model_prefix}.model"


# =============================================================
# Tokenize dataset
# =============================================================
def tokenize_hf_dataset(ds, spm_model_path):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)

    all_tokens = []
    print("Tokenizing HF dataset…")

    for row in ds:
        text = row[HF_TEXT_COLUMN]
        ids = sp.encode(text, out_type=int)
        all_tokens.extend(ids)

    all_tokens = np.array(all_tokens, dtype=np.int32)
    print(f"Total tokens: {len(all_tokens):,}")
    return all_tokens


# =============================================================
# Torch Dataset for tokens
# =============================================================
class HFTokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx: idx+self.seq_len]
        y = self.tokens[idx+1: idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# =============================================================
# Model Configs
# =============================================================
def get_model_configs(vocab_size, n_ctx):
    SMALL = GPTConfig(
        vocab_size=vocab_size, n_ctx=n_ctx,
        n_layer=4, d_model=64, n_head=4, d_ff=256
    )
    MEDIUM = GPTConfig(
        vocab_size=vocab_size, n_ctx=n_ctx,
        n_layer=10, d_model=128, n_head=8, d_ff=512
    )
    LARGE = GPTConfig(
        vocab_size=vocab_size, n_ctx=n_ctx,
        n_layer=12, d_model=192, n_head=8, d_ff=768
    )
    return [("1M", SMALL), ("5M", MEDIUM), ("10M", LARGE)]


# =============================================================
# Training Loop
# =============================================================
def train_model(model, train_dl, n_epochs, lr, device, save_path):
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    step = 0
    model.train()

    for epoch in range(n_epochs):
        step_in_epoch = 0
        for xb, yb in train_dl:

            MAX_STEPS_PER_EPOCH = 200_000 
            if step_in_epoch >= MAX_STEPS_PER_EPOCH:
                print(f"Reached {MAX_STEPS_PER_EPOCH} steps → ending epoch early.")
                break
            
            xb, yb = xb.to(device), yb.to(device)

            _, loss = model(xb, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if step % 200 == 0:
                bits = loss.item() / math.log(2)
                print(f"epoch {epoch} step {step} loss={loss:.4f} ({bits:.3f} bits)")
            step += 1
            step_in_epoch += 1

        ckpt = os.path.join(save_path, f"model_epoch{epoch}.pt")
        torch.save({"model": model.state_dict()}, ckpt)

    return model


# =============================================================
# Evaluate loss at arbitrary truncated context length
# =============================================================
@torch.no_grad()
def compute_model_loss_at_context(model, tokens, context_len, device, max_eval_tokens=200_000):
    """
    Compute model conditional entropy H(X_t | X_{t-context_len}^{t-1})
    by manually truncating the context at inference time.
    """
    model.eval()

    L = len(tokens)
    end = min(L - context_len - 1, max_eval_tokens)

    total_loss = 0.0
    count = 0

    for i in range(0, end, context_len):
        x = tokens[i:i+context_len]
        y = tokens[i+1:i+context_len+1]

        xb = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        yb = torch.tensor(y, dtype=torch.long, device=device).unsqueeze(0)

        _, loss = model(xb, yb)
        total_loss += loss.item()
        count += 1

    avg_nats = total_loss / count
    return avg_nats / math.log(2)  # bits/token


# =============================================================
# Fit power-law exponent α from loss ∝ n^(α−1)
# =============================================================
def fit_model_exponent(context_values, loss_values):
    xs, ys = [], []
    for n, Hn in zip(context_values, loss_values):
        xs.append(math.log(n))
        ys.append(math.log(Hn))

    alpha_slope, _ = np.polyfit(xs, ys, 1)
    beta = alpha_slope + 1
    return float(alpha_slope), float(beta)


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lengths", type=str, default="128,512,1024")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--build_tokenizer", action="store_true")
    parser.add_argument("--build_tokens", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # Load dataset
    # --------------------
    ds = load_hf_dataset()

    # --------------------
    # Train tokenizer
    # --------------------
    if args.build_tokenizer:
        spm_model_path = train_sentencepiece_tokenizer_hf(ds)
    else:
        spm_model_path = "wiki_bpe.model"

    # --------------------
    # Tokenize
    # --------------------
    if args.build_tokens:
        tokens = tokenize_hf_dataset(ds, spm_model_path)
        np.save("hf_tokens.npy", tokens)
    else:
        tokens = np.load("hf_tokens.npy")

    print(f"Loaded {len(tokens):,} tokens.")

    # Create train/eval eplit (2% eval)
    split = int(len(tokens) * 0.98)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]

    print(f"Train tokens: {len(train_tokens):,} | Eval tokens: {len(eval_tokens):,}")

    MAX_TOKENS = 1_000_000_000
    if len(train_tokens) > MAX_TOKENS:
        print(f"Truncating train_tokens from {len(train_tokens):,} to  {MAX_TOKENS:,}")
        train_tokens = train_tokens[:MAX_TOKENS]

    # Setup vocab
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    vocab_size = sp.get_piece_size()

    seq_list = [int(x) for x in args.seq_lengths.split(",")]

    results = {}

    # =============================================================
    # Train models
    # =============================================================
    diagonal_models = [
        ("1M", 128),
        ("5M", 512),
        ("10M", 1024),
    ]

    EPOCHS_PER_MODEL = {
        "1M": 1,
        "5M": 1,
        "10M": 1,
    }

    for size_name, n_ctx in diagonal_models:

        print(f"\n=== Training {size_name} at context {n_ctx} ===")

        dataset = HFTokenDataset(train_tokens, n_ctx)
        train_dl = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

        # pick correct config
        model_cfg = None
        for candidate_name, cfg in get_model_configs(vocab_size, n_ctx):
            if candidate_name == size_name:
                model_cfg = cfg
                break

        model_id = f"{size_name}_{n_ctx}"
        save_dir = f"checkpoints/{model_id}"
        os.makedirs(save_dir, exist_ok=True)

        model = GPT(model_cfg)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model_epochs = EPOCHS_PER_MODEL[size_name]
        print(f"Using {model_epochs} epochs")

        model = train_model(
            model=model,
            train_dl=train_dl,
            n_epochs=model_epochs,
            lr=args.lr,
            device=device,
            save_path=save_dir,
        )

        # evaluate
        context_probe = [1,2,4,8,16,32,64,128,256,512,1024]

        losses = {}
        valid_contexts = []

        print(f"Probing context scaling for {model_id}")

        for n_probe in context_probe:
            if n_probe > n_ctx:
                continue
            H_n = compute_model_loss_at_context(
                model, eval_tokens, context_len=n_probe, device=device
            )
            losses[n_probe] = H_n
            valid_contexts.append(n_probe)

        α, β = fit_model_exponent(
            valid_contexts,
            [losses[n] for n in valid_contexts]
        )

        results[model_id] = {
            "loss_vs_context": losses,
            "alpha_exponent": float(α),
            "beta_exponent": float(β),
        }

    # =============================================================
    # Save JSON
    # =============================================================
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved → model_results.json")


if __name__ == "__main__":
    main()
