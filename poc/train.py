#!/usr/bin/env python3
"""
train.py — Main entry point for training CustomLLM from scratch.

Quick start:
    python train.py --data data/my_format.txt --spec_name MY_FORMAT

Full options:
    python train.py --help
"""

import argparse
import os
import sys
import torch

from config   import ModelConfig
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer
from dataset  import build_loaders, load_text_files
from model    import CustomLLM
from trainer  import Trainer


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CustomLLM (GPT-style transformer) from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data",      required=True, nargs="+",
                   help="One or more training text files")
    p.add_argument("--spec_name", default="custom",
                   help="Name for this specification / format (used as output subdir)")
    p.add_argument("--out_dir",   default="checkpoints",
                   help="Root directory for saving checkpoints")

    # Tokenizer
    p.add_argument("--tokenizer",  choices=["char", "bpe"], default="char",
                   help="Tokenizer type  (char = character-level, bpe = Byte-Pair Encoding)")
    p.add_argument("--bpe_vocab",  type=int, default=1000,
                   help="BPE vocabulary size (only used when --tokenizer bpe)")

    # Architecture
    p.add_argument("--context_len", type=int,   default=256,
                   help="Maximum sequence length (tokens) the model can attend to")
    p.add_argument("--n_embd",      type=int,   default=384,  help="Embedding dimension")
    p.add_argument("--n_head",      type=int,   default=6,    help="Number of attention heads")
    p.add_argument("--n_layer",     type=int,   default=6,    help="Number of transformer blocks")
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--no_bias",     action="store_true",
                   help="Disable bias in Linear layers (GPT-2 style, slightly faster)")

    # Training
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_iters",   type=int,   default=5000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--min_lr",      type=float, default=3e-5)
    p.add_argument("--warmup",      type=int,   default=200,  help="Warmup iterations")
    p.add_argument("--weight_decay",type=float, default=0.1)
    p.add_argument("--grad_clip",   type=float, default=1.0,  help="0 = disabled")
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--eval_every",  type=int,   default=200)
    p.add_argument("--eval_iters",  type=int,   default=50)

    # Misc
    p.add_argument("--device",  default="auto",
                   help="cuda / mps / cpu / auto")
    p.add_argument("--resume",  default=None,
                   help="Resume training from a checkpoint (.pt file)")
    p.add_argument("--seed",    type=int, default=42)

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    out_dir = os.path.join(args.out_dir, args.spec_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else
                  "cpu")
    else:
        device = args.device

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'═'*62}")
    print(f"  CustomLLM  —  Spec: {args.spec_name}")
    print(f"{'═'*62}")
    print(f"  Device    : {device}")
    print(f"  Tokenizer : {args.tokenizer}")
    print(f"  Out dir   : {out_dir}")

    # ── Load text ─────────────────────────────────────────────────────────────
    text = load_text_files(args.data)
    print(f"  Corpus    : {len(text):,} characters")

    # ── Build / load tokenizer ────────────────────────────────────────────────
    if args.resume:
        # Try to load the tokenizer saved alongside the checkpoint
        tok_path = os.path.join(out_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            print(f"  Loading existing tokenizer: {tok_path}")
            tokenizer = load_tokenizer(tok_path)
        else:
            raise FileNotFoundError(
                f"Resuming requires a tokenizer at {tok_path}"
            )
    else:
        if args.tokenizer == "char":
            tokenizer = CharTokenizer().train(text)
        else:
            tokenizer = BPETokenizer(vocab_size=args.bpe_vocab).train(text)

        tok_path = os.path.join(out_dir, "tokenizer.json")
        tokenizer.save(tok_path)
        print(f"  Tokenizer saved → {tok_path}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = ModelConfig(
        vocab_size      = tokenizer.vocab_size,
        context_length  = args.context_len,
        n_embd          = args.n_embd,
        n_head          = args.n_head,
        n_layer         = args.n_layer,
        dropout         = args.dropout,
        bias            = not args.no_bias,
        batch_size      = args.batch_size,
        max_iters       = args.max_iters,
        eval_interval   = args.eval_every,
        eval_iters      = args.eval_iters,
        learning_rate   = args.lr,
        min_lr          = args.min_lr,
        warmup_iters    = args.warmup,
        lr_decay_iters  = args.max_iters,
        weight_decay    = args.weight_decay,
        grad_clip       = args.grad_clip,
        train_split     = args.train_split,
        out_dir         = out_dir,
        spec_name       = args.spec_name,
    )
    config.save(os.path.join(out_dir, "config.json"))
    print(config)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(text, tokenizer, config)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model = CustomLLM.from_checkpoint(args.resume, device=device)
    else:
        model = CustomLLM(config).to(device)

    print(f"\n{model}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer   = Trainer(model, config, train_loader, val_loader, device)
    best_loss = trainer.train()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  Artifacts saved:")
    for fname in ["best_model.pt", "final_model.pt", "tokenizer.json",
                  "config.json", "training_log.csv"]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    {fname:<22}  {size/1024:.0f} KB")

    print(f"\n  Generate text with:")
    print(f"    python generate.py \\")
    print(f"        --checkpoint {out_dir}/best_model.pt \\")
    print(f"        --tokenizer  {out_dir}/tokenizer.json \\")
    print(f"        --prompt 'your seed text here'")
    print()


if __name__ == "__main__":
    main()
