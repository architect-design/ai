"""
dashboard.py — Live terminal dashboard for CustomLLM training.

Renders a real-time ASCII training dashboard that shows:
  • Live loss curve (sparkline + bar chart)
  • Current / best / average loss
  • Learning rate trace
  • Progress bar
  • Recent training log
  • A short live sample generated from the model

Launch alongside training by tailing the training_log.csv, or use
DashboardTrainer as a drop-in replacement for Trainer.

Usage:
    from dashboard import DashboardTrainer
    trainer = DashboardTrainer(model, config, train_loader, val_loader, device)
    trainer.train()
"""

import math
import os
import shutil
import sys
import time
from typing import List, Optional

import torch

from trainer import Trainer


# ─── Terminal helpers ─────────────────────────────────────────────────────────

def _term_size() -> tuple:
    s = shutil.get_terminal_size((100, 40))
    return s.columns, s.lines


def _clr():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _mv(row: int, col: int):
    sys.stdout.write(f"\033[{row};{col}H")


def _bold(s: str) -> str:  return f"\033[1m{s}\033[0m"
def _dim(s:  str) -> str:  return f"\033[2m{s}\033[0m"
def _grn(s:  str) -> str:  return f"\033[32m{s}\033[0m"
def _yel(s:  str) -> str:  return f"\033[33m{s}\033[0m"
def _cyn(s:  str) -> str:  return f"\033[36m{s}\033[0m"
def _red(s:  str) -> str:  return f"\033[31m{s}\033[0m"
def _mag(s:  str) -> str:  return f"\033[35m{s}\033[0m"


# ─── Sparkline ────────────────────────────────────────────────────────────────

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 40) -> str:
    if not values:
        return "─" * width
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return SPARK_CHARS[3] * len(vals)
    norm = [(v - mn) / (mx - mn) for v in vals]
    return "".join(SPARK_CHARS[int(n * (len(SPARK_CHARS) - 1))] for n in norm)


def bar_chart(values: List[float], width: int = 52, height: int = 8) -> List[str]:
    """Return a list of `height` strings forming a vertical bar chart."""
    if not values:
        return ["─" * width] * height

    n    = min(len(values), width)
    vals = values[-n:]
    mn   = max(0, min(vals) * 0.95)
    mx   = max(vals) * 1.05
    bars = []

    for row in range(height - 1, -1, -1):
        threshold = mn + (mx - mn) * row / (height - 1)
        line      = ""
        for v in vals:
            if v >= threshold:
                line += "█"
            elif v >= threshold - (mx - mn) / (height * 2):
                line += "▄"
            else:
                line += " "
        # Right-pad to full width
        line = line.ljust(n).ljust(width)
        bars.append(line)

    # Axis labels
    axis = f"{mn:.3f}" + " " * (width - 10) + f"{mx:.3f}"
    bars.append(axis)
    return bars


def progress_bar(done: int, total: int, width: int = 40) -> str:
    pct   = done / max(total, 1)
    filled = int(pct * width)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%  {done}/{total}"


# ─── DashboardTrainer ─────────────────────────────────────────────────────────

class DashboardTrainer(Trainer):
    """
    Extends Trainer with a live terminal dashboard.

    Every eval_interval iterations:
      • Clears the screen and redraws the dashboard
      • Generates a short sample from the current model

    Falls back gracefully to plain Trainer output when the terminal
    is too narrow (<60 columns) or not a TTY.
    """

    def __init__(self, model, config, train_loader, val_loader, device,
                 sample_len: int = 120, sample_temp: float = 0.8,
                 sample_prompt: str = ""):
        super().__init__(model, config, train_loader, val_loader, device)
        self.sample_len    = sample_len
        self.sample_temp   = sample_temp
        self.sample_prompt = sample_prompt
        self._losses_all:  List[float] = []
        self._val_losses:  List[float] = []
        self._lrs:         List[float] = []
        self._log_lines:   List[str]   = []
        self._last_sample: str = ""
        self._use_dash     = sys.stdout.isatty()

    def _sample_text(self) -> str:
        """Generate a short sample from the current model state."""
        try:
            from generate import Generator
            gen = Generator(self.model, self._tokenizer_ref, self.device)
            return gen.generate(
                prompt         = self.sample_prompt,
                max_new_tokens = self.sample_len,
                temperature    = self.sample_temp,
                top_k          = 40,
                top_p          = 0.95,
            )
        except Exception:
            return "(sample unavailable)"

    def _log(self, msg: str, kind: str = ""):
        ts = time.strftime("%H:%M:%S")
        prefix = {"ok": "✓", "warn": "!", "err": "✗"}.get(kind, "·")
        self._log_lines.append(f"  {ts}  {prefix}  {msg}")
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]

    def _draw(self, iter_num: int, max_iters: int, elapsed: float):
        W, H = _term_size()
        if W < 60:
            return

        _clr()

        # ── Header ───────────────────────────────────────────────────────────
        spec   = self.config.spec_name
        params = self.model.num_params()
        header = (f"  CustomLLM  ·  spec={spec}  ·  "
                  f"params={params/1e6:.2f}M  ·  device={self.device}")
        print(_bold(_cyn(header)))
        print(_dim("  " + "─" * (W - 4)))

        # ── Progress ─────────────────────────────────────────────────────────
        pbar = progress_bar(iter_num, max_iters, width=min(W - 30, 50))
        elap = f"{elapsed:.0f}s"
        eta  = ""
        if iter_num > 0:
            per_it = elapsed / iter_num
            rem    = per_it * (max_iters - iter_num)
            eta    = f"  ETA {rem:.0f}s"
        print(f"  {_grn(pbar)}{_dim(eta)}  ⏱ {elap}")

        # ── Stats row ─────────────────────────────────────────────────────────
        cur  = self._losses_all[-1]  if self._losses_all  else 0.0
        best = min(self._val_losses) if self._val_losses  else 0.0
        val  = self._val_losses[-1]  if self._val_losses  else 0.0
        lr   = self._lrs[-1]         if self._lrs         else 0.0
        ppl  = math.exp(min(val, 15)) if val > 0 else 0.0
        bpc  = val / math.log(2)      if val > 0 else 0.0

        col_w = max(12, (W - 6) // 5)
        stats = [
            ("Train Loss",  f"{cur:.4f}",  _grn),
            ("Val Loss",    f"{val:.4f}",  _yel),
            ("Best Val",    f"{best:.4f}", _mag),
            ("PPL",         f"{ppl:.2f}",  _cyn),
            ("BPC",         f"{bpc:.4f}",  _cyn),
        ]
        row = ""
        for label, val_s, col_fn in stats:
            cell = f"  {_dim(label+':')} {col_fn(_bold(val_s))}"
            row += cell.ljust(col_w + 20)
        print(row[:W])
        print(f"  {_dim('LR:')} {_yel(f'{lr:.2e}')}")

        print(_dim("  " + "─" * (W - 4)))

        # ── Loss chart ───────────────────────────────────────────────────────
        chart_w = min(W - 12, 70)
        print(f"  {_bold('Training Loss')}  {_dim(sparkline(self._losses_all, chart_w))}")
        print(f"  {_bold('Val Loss     ')}  {_dim(sparkline(self._val_losses, chart_w))}")

        print(_dim("  " + "─" * (W - 4)))

        # ── Bar chart ─────────────────────────────────────────────────────────
        bar_h = 6
        bars  = bar_chart(self._val_losses, width=min(chart_w, 60), height=bar_h)
        print(f"  {_bold('Val Loss Curve:')}")
        for b in bars:
            print(f"  {_grn(b)}")

        print(_dim("  " + "─" * (W - 4)))

        # ── Recent log ───────────────────────────────────────────────────────
        log_rows = min(6, H - 30)
        print(f"  {_bold('Training Log:')}")
        for line in self._log_lines[-log_rows:]:
            print(_dim(line[:W - 2]))

        print(_dim("  " + "─" * (W - 4)))

        # ── Live sample ──────────────────────────────────────────────────────
        if self._last_sample:
            print(f"  {_bold('Live Sample:')}")
            sample_preview = self._last_sample[:min(W * 3, 300)]
            # Wrap to terminal width
            while sample_preview:
                print(f"  {_grn(sample_preview[:W-4])}")
                sample_preview = sample_preview[W-4:]

        sys.stdout.flush()

    # ── Override eval hook from Trainer ───────────────────────────────────────

    def train(self) -> float:
        """
        Override Trainer.train() to inject dashboard rendering at each
        eval checkpoint.
        """
        cfg   = self.config
        model = self.model
        model.train()

        # We need a tokenizer for sample generation — peek from the loader
        self._tokenizer_ref = None
        for attr in ("dataset", "_dataset"):
            ds = getattr(self.train_loader, attr, None)
            if ds is not None and hasattr(ds, "tokenizer"):
                self._tokenizer_ref = ds.tokenizer
                break

        t_start    = time.time()
        train_iter = iter(self.train_loader)
        best_val   = float("inf")

        if not self._use_dash:
            # Fall back to plain trainer output
            return super().train()

        try:
            while self.iter_num < cfg.max_iters:

                lr = self._get_lr()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                self._lrs.append(lr)

                # Evaluation
                if self.iter_num % cfg.eval_interval == 0:
                    losses  = self._estimate_loss()
                    elapsed = time.time() - t_start
                    tl      = losses["train"]
                    vl      = losses["val"]
                    self._val_losses.append(vl)

                    improved = vl < best_val
                    if improved:
                        best_val = vl
                        path     = os.path.join(cfg.out_dir, "best_model.pt")
                        model.save_checkpoint(
                            path, extra={"iter": self.iter_num, "val_loss": vl}
                        )
                        self._log(f"New best val loss: {vl:.4f} → saved", "ok")
                    else:
                        self._log(
                            f"iter {self.iter_num:5d}  "
                            f"train={tl:.4f}  val={vl:.4f}  lr={lr:.2e}"
                        )

                    # Generate a live sample (requires tokenizer)
                    if self._tokenizer_ref:
                        self._last_sample = self._sample_text()

                    self._draw(self.iter_num, cfg.max_iters, elapsed)

                # Training step
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)
                _, loss = model(x, targets=y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )

                self.optimizer.step()
                self.iter_num  += 1
                self._losses_all.append(loss.item())

        except KeyboardInterrupt:
            print("\n\n  [!] Interrupted — saving…")

        # Final checkpoint
        final_path = os.path.join(cfg.out_dir, "final_model.pt")
        model.save_checkpoint(
            final_path,
            extra={"iter": self.iter_num, "val_loss": best_val}
        )
        self._log_f.close()

        elapsed = time.time() - t_start
        print(f"\n  Training complete in {elapsed:.0f}s  "
              f"best_val={best_val:.4f}")
        return best_val


# ─── Standalone log watcher ───────────────────────────────────────────────────

def watch_log(log_path: str, refresh: float = 2.0):
    """
    Tail a training_log.csv file and redraw a live ASCII dashboard.
    Run this in a separate terminal while train.py runs.

    Usage:
        python dashboard.py --log checkpoints/MY_SPEC/training_log.csv
    """
    import csv

    losses_train, losses_val, lrs = [], [], []
    print(f"Watching {log_path} (Ctrl-C to quit)…")

    while True:
        try:
            if os.path.exists(log_path):
                with open(log_path) as f:
                    rows = list(csv.DictReader(f))
                losses_train = [float(r["train_loss"]) for r in rows]
                losses_val   = [float(r["val_loss"])   for r in rows]
                lrs          = [float(r["lr"])          for r in rows]

                W, _ = _term_size()
                _clr()
                print(_bold(_cyn(f"  CustomLLM — Live Training Monitor  |  {log_path}")))
                print(_dim("  " + "─" * (W - 4)))
                cur  = losses_train[-1] if losses_train else 0.0
                val  = losses_val[-1]   if losses_val   else 0.0
                best = min(losses_val)  if losses_val   else 0.0
                lr   = lrs[-1]          if lrs          else 0.0
                print(f"  Iterations: {len(rows)}  "
                      f"{_dim('Train:')} {_grn(f'{cur:.4f}')}  "
                      f"{_dim('Val:')} {_yel(f'{val:.4f}')}  "
                      f"{_dim('Best:')} {_mag(f'{best:.4f}')}  "
                      f"{_dim('LR:')} {_cyn(f'{lr:.2e}')}")
                print()
                chart_w = min(W - 6, 72)
                print(f"  {_bold('Train')}  {sparkline(losses_train, chart_w)}")
                print(f"  {_bold('Val  ')}  {sparkline(losses_val,   chart_w)}")
                print()
                bars = bar_chart(losses_val, width=min(chart_w, 60), height=7)
                for b in bars:
                    print(f"  {_grn(b)}")
            else:
                print(f"  Waiting for {log_path}…")

            time.sleep(refresh)
        except KeyboardInterrupt:
            print("\nDone.")
            break


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Watch a CustomLLM training_log.csv live")
    p.add_argument("--log",     required=True,  help="Path to training_log.csv")
    p.add_argument("--refresh", type=float, default=2.0, help="Refresh interval (seconds)")
    args = p.parse_args()
    watch_log(args.log, args.refresh)
