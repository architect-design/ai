"""
finetune.py — Fine-tune a trained CustomLLM on new domain / spec data.

Fine-tuning adapts a base checkpoint to a new format or sub-domain without
training from scratch.  Two modes are supported:

  1. Full fine-tune  — all parameters updated (default)
  2. Frozen-base     — only the final LM head is updated (--freeze_base)
                       fastest; useful when the new dataset is very small

Usage (CLI):
    # Full fine-tune on new spec data
    python finetune.py \
        --checkpoint  checkpoints/BASE_SPEC/best_model.pt \
        --tokenizer   checkpoints/BASE_SPEC/tokenizer.json \
        --data        data/new_format.txt \
        --spec_name   NEW_SPEC \
        --max_iters   2000 \
        --lr          5e-5

    # Head-only fine-tune (tiny dataset)
    python finetune.py \
        --checkpoint  checkpoints/BASE_SPEC/best_model.pt \
        --tokenizer   checkpoints/BASE_SPEC/tokenizer.json \
        --data        data/new_format.txt \
        --spec_name   NEW_SPEC_SMALL \
        --freeze_base
"""

import argparse
import os
import time
import math
import torch

from config   import ModelConfig
from model    import CustomLLM
from tokenizer import load_tokenizer, CharTokenizer, BPETokenizer
from dataset  import build_loaders, load_text_files
from trainer  import Trainer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_base(model: CustomLLM):
    """
    Freeze everything except the LM head linear layer.
    This lets you adapt vocabulary prediction to a new domain
    without changing the attention / FFN weights.
    """
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = count_trainable(model)
    print(f"  [freeze_base] Frozen {n_frozen:,} params  "
          f"Trainable {n_trainable:,} params (lm_head only)")


def freeze_embeddings(model: CustomLLM):
    """Freeze token + positional embeddings (optional, separate from freeze_base)."""
    for name, param in model.named_parameters():
        if "wte" in name or "wpe" in name:
            param.requires_grad = False
    print(f"  [freeze_embeddings] Embeddings frozen  "
          f"Trainable {count_trainable(model):,} params")


class FineTuneTrainer(Trainer):
    """
    Thin subclass of Trainer that uses a lower default LR,
    shorter warmup, and logs fine-tuning metadata into the checkpoint.
    """

    def __init__(self, model, config, train_loader, val_loader, device,
                 base_checkpoint: str, spec_name: str):
        super().__init__(model, config, train_loader, val_loader, device)
        self.base_checkpoint = base_checkpoint
        self.spec_name       = spec_name

    def _save_best(self, val_loss: float):
        best_path = os.path.join(self.config.out_dir, "best_model.pt")
        self.model.save_checkpoint(
            best_path,
            extra={
                "iter":            self.iter_num,
                "val_loss":        val_loss,
                "finetune_base":   self.base_checkpoint,
                "finetune_spec":   self.spec_name,
            },
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a CustomLLM checkpoint on new spec data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to base .pt checkpoint")
    p.add_argument("--tokenizer",  required=True,
                   help="Path to tokenizer.json from the base checkpoint")
    p.add_argument("--data",       required=True, nargs="+",
                   help="New training text files")
    p.add_argument("--spec_name",  required=True,
                   help="Name for the fine-tuned model")

    # Fine-tune mode
    p.add_argument("--freeze_base", action="store_true",
                   help="Freeze all weights except the LM head (head-only FT)")
    p.add_argument("--freeze_emb",  action="store_true",
                   help="Also freeze token & position embeddings")

    # Training
    p.add_argument("--max_iters",    type=int,   default=2000)
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="Learning rate (lower than pretraining, e.g. 1e-5 – 1e-4)")
    p.add_argument("--min_lr",       type=float, default=5e-6)
    p.add_argument("--warmup",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--train_split",  type=float, default=0.9)
    p.add_argument("--eval_every",   type=int,   default=100)

    # Output
    p.add_argument("--out_dir",  default="checkpoints")
    p.add_argument("--device",   default="auto")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = os.path.join(args.out_dir, args.spec_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    torch.manual_seed(args.seed)

    print(f"\n{'═'*60}")
    print(f"  CustomLLM Fine-Tuning  |  spec='{args.spec_name}'")
    print(f"{'═'*60}")
    print(f"  Base checkpoint : {args.checkpoint}")
    print(f"  Device          : {device}")
    print(f"  Out dir         : {out_dir}")
    mode = "head-only" if args.freeze_base else "full"
    print(f"  Mode            : {mode}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    # Fine-tuning reuses the original tokenizer to keep the same vocabulary.
    tokenizer = load_tokenizer(args.tokenizer)
    import shutil
    shutil.copy(args.tokenizer, os.path.join(out_dir, "tokenizer.json"))
    print(f"  Tokenizer       : {tokenizer.__class__.__name__}  "
          f"vocab={tokenizer.vocab_size}")

    # ── Load base model ───────────────────────────────────────────────────────
    model = CustomLLM.from_checkpoint(args.checkpoint, device=device)

    # ── Freeze layers (optional) ──────────────────────────────────────────────
    if args.freeze_base:
        freeze_base(model)
    elif args.freeze_emb:
        freeze_embeddings(model)

    print(f"  Trainable params: {count_trainable(model):,}  "
          f"/ {model.num_params():,} total")

    # ── Build config for fine-tuning ──────────────────────────────────────────
    cfg = model.config
    ft_config = ModelConfig(
        vocab_size      = cfg.vocab_size,
        context_length  = cfg.context_length,
        n_embd          = cfg.n_embd,
        n_head          = cfg.n_head,
        n_layer         = cfg.n_layer,
        dropout         = cfg.dropout,
        bias            = cfg.bias,
        batch_size      = args.batch_size,
        max_iters       = args.max_iters,
        eval_interval   = args.eval_every,
        eval_iters      = 30,
        learning_rate   = args.lr,
        min_lr          = args.min_lr,
        warmup_iters    = args.warmup,
        lr_decay_iters  = args.max_iters,
        weight_decay    = 0.01,              # lower than pretraining
        grad_clip       = args.grad_clip,
        train_split     = args.train_split,
        out_dir         = out_dir,
        spec_name       = args.spec_name,
    )
    ft_config.save(os.path.join(out_dir, "config.json"))
    print(ft_config)

    # ── Load new domain data ──────────────────────────────────────────────────
    text = load_text_files(args.data)
    print(f"\n  New corpus: {len(text):,} characters")

    train_loader, val_loader = build_loaders(text, tokenizer, ft_config)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    trainer = FineTuneTrainer(
        model, ft_config, train_loader, val_loader, device,
        base_checkpoint = args.checkpoint,
        spec_name       = args.spec_name,
    )
    best_loss = trainer.train()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  Artifacts:")
    for fname in ["best_model.pt", "final_model.pt",
                  "tokenizer.json", "config.json", "training_log.csv"]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f"    {fname:<22}  {os.path.getsize(path)/1024:.0f} KB")

    print(f"\n  Generate with fine-tuned model:")
    print(f"    python generate.py \\")
    print(f"        --checkpoint {out_dir}/best_model.pt \\")
    print(f"        --tokenizer  {out_dir}/tokenizer.json \\")
    print(f"        --prompt 'seed' --temperature 0.7")
    print()


if __name__ == "__main__":
    main()
