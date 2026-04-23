"""
trainer.py — Training loop for CustomLLM

Features
────────
  • Cosine learning-rate schedule with linear warmup
  • Gradient norm clipping
  • Periodic validation-loss estimation
  • Auto-saves best checkpoint and final checkpoint
  • CSV training log (loss / lr / time per eval)
  • Graceful keyboard-interrupt: saves checkpoint before exiting
"""

import math
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        self.model        = model
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.iter_num      = 0
        self.best_val_loss = float("inf")

        # ── Optimizer (AdamW with selective weight decay) ─────────────────────
        # Only 2-D+ tensors (weight matrices) get weight decay.
        # Biases, LayerNorm, and embeddings do not.
        decay_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and p.dim() < 2]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr    = config.learning_rate,
            betas = (config.beta1, config.beta2),
            fused = (device == "cuda"),   # fused kernel when on GPU
        )

        # ── Logging ───────────────────────────────────────────────────────────
        os.makedirs(config.out_dir, exist_ok=True)
        log_path = os.path.join(config.out_dir, "training_log.csv")
        self._log_f = open(log_path, "w")
        self._log_f.write("iter,train_loss,val_loss,lr,elapsed_s\n")
        self._log_f.flush()

    # ── LR schedule: linear warmup → cosine decay ─────────────────────────────

    def _get_lr(self) -> float:
        cfg = self.config
        it  = self.iter_num

        if it < cfg.warmup_iters:                          # linear warmup
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it >= cfg.lr_decay_iters:                       # floor
            return cfg.min_lr

        # cosine annealing between warmup and decay_iters
        progress = (it - cfg.warmup_iters) / max(
            cfg.lr_decay_iters - cfg.warmup_iters, 1
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # ── Loss estimation (runs on both train & val splits) ─────────────────────

    @torch.no_grad()
    def _estimate_loss(self) -> Dict[str, float]:
        self.model.eval()
        result = {}
        for name, loader in [("train", self.train_loader),
                              ("val",   self.val_loader)]:
            total, count = 0.0, 0
            for i, (x, y) in enumerate(loader):
                if i >= self.config.eval_iters:
                    break
                x, y      = x.to(self.device), y.to(self.device)
                _, loss   = self.model(x, targets=y)
                total    += loss.item()
                count    += 1
            result[name] = total / max(count, 1)
        self.model.train()
        return result

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> float:
        cfg   = self.config
        model = self.model
        model.train()

        sep  = "─" * 62
        print(f"\n{sep}")
        print(f"  CustomLLM Training  |  device={self.device}")
        print(f"  Parameters : {model.num_params():,}")
        print(f"  Iterations : {cfg.max_iters:,}")
        print(f"  Batch size : {cfg.batch_size}")
        print(f"  LR         : {cfg.learning_rate} → {cfg.min_lr} (cosine)")
        print(f"  Out dir    : {cfg.out_dir}")
        print(f"{sep}\n")

        t_start      = time.time()
        running_loss = 0.0
        train_iter   = iter(self.train_loader)

        try:
            while self.iter_num < cfg.max_iters:

                # ── LR update ────────────────────────────────────────────────
                lr = self._get_lr()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # ── Periodic evaluation ───────────────────────────────────────
                if self.iter_num % cfg.eval_interval == 0:
                    losses    = self._estimate_loss()
                    elapsed   = time.time() - t_start
                    tr_l      = losses["train"]
                    va_l      = losses["val"]
                    improved  = va_l < self.best_val_loss

                    marker = "✓ NEW BEST" if improved else ""
                    print(
                        f"  [{self.iter_num:>5d}/{cfg.max_iters}] "
                        f"train={tr_l:.4f}  val={va_l:.4f}  "
                        f"lr={lr:.2e}  t={elapsed:.0f}s  {marker}"
                    )

                    self._log_f.write(
                        f"{self.iter_num},{tr_l:.4f},{va_l:.4f},{lr:.6f},{elapsed:.1f}\n"
                    )
                    self._log_f.flush()

                    if improved:
                        self.best_val_loss = va_l
                        best_path = os.path.join(cfg.out_dir, "best_model.pt")
                        model.save_checkpoint(
                            best_path,
                            extra={"iter": self.iter_num, "val_loss": va_l},
                        )

                # ── Fetch next batch ─────────────────────────────────────────
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

                # ── Forward pass ─────────────────────────────────────────────
                _, loss = model(x, targets=y)

                # ── Backward pass ────────────────────────────────────────────
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )

                self.optimizer.step()
                self.iter_num += 1
                running_loss  += loss.item()

                # ── Periodic console update ──────────────────────────────────
                log_every = max(1, cfg.eval_interval // 4)
                if self.iter_num % log_every == 0:
                    avg = running_loss / log_every
                    print(f"    iter {self.iter_num:5d} | "
                          f"loss={avg:.4f}  lr={lr:.2e}")
                    running_loss = 0.0

        except KeyboardInterrupt:
            print("\n  [!] Interrupted — saving checkpoint before exit…")

        # ── Save final checkpoint ─────────────────────────────────────────────
        final_path = os.path.join(cfg.out_dir, "final_model.pt")
        model.save_checkpoint(
            final_path,
            extra={"iter": self.iter_num, "val_loss": self.best_val_loss},
        )
        self._log_f.close()

        elapsed = time.time() - t_start
        print(f"\n{sep}")
        print(f"  Training finished in {elapsed:.0f}s")
        print(f"  Best val loss : {self.best_val_loss:.4f}")
        print(f"  Checkpoints   : {cfg.out_dir}/")
        print(f"{sep}\n")

        return self.best_val_loss
