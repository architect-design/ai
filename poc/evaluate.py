"""
evaluate.py — Evaluation suite for CustomLLM

Metrics computed
────────────────
  • Loss (cross-entropy)
  • Perplexity (PPL) — exp(loss)
  • Bits-Per-Character (BPC) — loss / ln(2)
  • Bits-Per-Token (BPT)
  • Vocabulary coverage — % of test tokens that appear in vocab
  • Format consistency score — structural pattern overlap with training data
  • Generation diversity — unique n-grams / total n-grams (dist-1, dist-2, dist-3)
  • Repetition rate — fraction of n-grams that appear more than once

Usage (CLI):
    python evaluate.py \
        --checkpoint checkpoints/MY_SPEC/best_model.pt \
        --tokenizer  checkpoints/MY_SPEC/tokenizer.json \
        --test_data  data/processed/MY_SPEC_test.txt \
        --num_samples 5 --sample_len 300

Usage (API):
    from evaluate import Evaluator
    ev = Evaluator.from_checkpoint("checkpoints/MY_SPEC/best_model.pt",
                                   "checkpoints/MY_SPEC/tokenizer.json")
    report = ev.full_report("data/processed/MY_SPEC_test.txt")
    ev.print_report(report)
"""

import argparse
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─── n-gram helpers ───────────────────────────────────────────────────────────

def ngrams(tokens: List, n: int) -> List[Tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: List, n: int) -> float:
    """Distinct-N: unique n-grams / total n-grams (diversity metric)."""
    all_ng = ngrams(tokens, n)
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


def repetition_rate(tokens: List, n: int = 3) -> float:
    """Fraction of n-grams that appear more than once."""
    counts = Counter(ngrams(tokens, n))
    if not counts:
        return 0.0
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


# ─── format-consistency helpers ───────────────────────────────────────────────

def char_bigram_profile(text: str) -> Counter:
    """Return a counter of character bigrams."""
    return Counter(zip(text, text[1:]))


def profile_overlap(ref: Counter, hyp: Counter) -> float:
    """
    Cosine similarity between two bigram-frequency profiles.
    Returns a value in [0, 1].
    """
    keys  = set(ref) | set(hyp)
    dot   = sum(ref[k] * hyp[k] for k in keys)
    norm_r = math.sqrt(sum(v * v for v in ref.values()))
    norm_h = math.sqrt(sum(v * v for v in hyp.values()))
    if norm_r == 0 or norm_h == 0:
        return 0.0
    return dot / (norm_r * norm_h)


# ─── Evaluator class ──────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, model, tokenizer, device: str):
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.device    = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        tokenizer_path: str,
        device: str = "auto",
    ) -> "Evaluator":
        from model    import CustomLLM
        from tokenizer import load_tokenizer

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

        model     = CustomLLM.from_checkpoint(checkpoint, device=device)
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"[Evaluator] device={device}  params={model.num_params():,}")
        return cls(model, tokenizer, device)

    # ── loss / perplexity ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_loss(
        self,
        text: str,
        context_length: Optional[int] = None,
        batch_size: int = 8,
        stride: int = 1,
    ) -> Dict[str, float]:
        """
        Compute cross-entropy loss, perplexity, and BPC over `text`.

        Uses a sliding window of width `context_length` (defaults to model's
        context length) stepped by `stride` tokens to cover all tokens.
        """
        ctx_len = context_length or self.model.config.context_length
        tokens  = self.tokenizer.encode(text)

        if len(tokens) <= 1:
            return {"loss": 0.0, "ppl": 1.0, "bpc": 0.0, "bpt": 0.0, "n_tokens": 0}

        total_nll = 0.0
        total_tok = 0

        # Sliding window evaluation
        windows_x, windows_y = [], []
        for i in range(0, len(tokens) - 1, max(stride, 1)):
            end  = min(i + ctx_len, len(tokens))
            x    = tokens[i      : end]
            y    = tokens[i + 1  : end + 1]
            if len(x) < 2:
                break
            # Pad to same length for batching
            windows_x.append(x)
            windows_y.append(y)

        for i in range(0, len(windows_x), batch_size):
            batch_x = windows_x[i : i + batch_size]
            batch_y = windows_y[i : i + batch_size]
            max_l   = max(len(s) for s in batch_x)

            # Left-pad to max_l
            px = torch.zeros(len(batch_x), max_l, dtype=torch.long)
            py = torch.full((len(batch_x), max_l), -1, dtype=torch.long)
            for j, (sx, sy) in enumerate(zip(batch_x, batch_y)):
                L = len(sx)
                px[j, :L] = torch.tensor(sx)
                py[j, :L] = torch.tensor(sy)

            px, py = px.to(self.device), py.to(self.device)
            logits, _ = self.model(px)               # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                py.view(-1),
                ignore_index=-1,
                reduction="sum",
            )
            valid = (py != -1).sum().item()
            total_nll += loss.item()
            total_tok += valid

        if total_tok == 0:
            return {"loss": 0.0, "ppl": 1.0, "bpc": 0.0, "bpt": 0.0, "n_tokens": 0}

        avg_loss = total_nll / total_tok
        ppl      = math.exp(min(avg_loss, 20))
        bpt      = avg_loss / math.log(2)
        # BPC: bits per character (use tokenizer decode to estimate char count)
        n_chars  = len(self.tokenizer.decode(tokens))
        bpc      = (total_nll / math.log(2)) / max(n_chars, 1)

        return {
            "loss":     avg_loss,
            "ppl":      ppl,
            "bpc":      bpc,
            "bpt":      bpt,
            "n_tokens": total_tok,
            "n_chars":  n_chars,
        }

    # ── vocabulary coverage ───────────────────────────────────────────────────

    def vocab_coverage(self, text: str) -> Dict[str, float]:
        """What fraction of characters appear in the vocabulary?"""
        tokens = self.tokenizer.encode(text)
        unk    = sum(1 for t in tokens if t == self.tokenizer.UNK)
        total  = len(tokens)
        return {
            "coverage":  1.0 - unk / max(total, 1),
            "unk_count": unk,
            "n_tokens":  total,
        }

    # ── generation diversity metrics ──────────────────────────────────────────

    def generation_diversity(
        self,
        num_samples: int = 5,
        sample_len:  int = 200,
        temperature: float = 0.8,
        top_k: int   = 50,
        top_p: float = 0.95,
    ) -> Dict[str, float]:
        """
        Generate `num_samples` texts and compute diversity / repetition metrics.
        """
        from generate import Generator
        gen = Generator(self.model, self.tokenizer, self.device)

        samples = []
        for _ in range(num_samples):
            s = gen.generate(
                prompt="", max_new_tokens=sample_len,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            samples.append(s)

        # Token-level metrics
        all_tokens = []
        for s in samples:
            all_tokens.extend(self.tokenizer.encode(s))

        # Character-level metrics (for char models)
        all_chars = list("".join(samples))

        return {
            "dist1_tok": distinct_n(all_tokens, 1),
            "dist2_tok": distinct_n(all_tokens, 2),
            "dist3_tok": distinct_n(all_tokens, 3),
            "dist1_chr": distinct_n(all_chars, 1),
            "dist2_chr": distinct_n(all_chars, 2),
            "rep_rate":  repetition_rate(all_tokens, 3),
            "avg_len":   sum(len(s) for s in samples) / num_samples,
            "samples":   samples,
        }

    # ── format consistency ────────────────────────────────────────────────────

    def format_consistency(
        self,
        ref_text: str,
        num_samples: int = 5,
        sample_len: int  = 300,
    ) -> Dict[str, float]:
        """
        Generate text and compare its character-bigram profile to the
        reference corpus. Score in [0, 1]; 1.0 = identical structure.
        """
        from generate import Generator
        gen = Generator(self.model, self.tokenizer, self.device)

        ref_profile = char_bigram_profile(ref_text[:5000])
        scores = []
        for _ in range(num_samples):
            s = gen.generate(prompt="", max_new_tokens=sample_len)
            hyp_profile = char_bigram_profile(s)
            scores.append(profile_overlap(ref_profile, hyp_profile))

        return {
            "mean_consistency": sum(scores) / max(len(scores), 1),
            "min_consistency":  min(scores, default=0.0),
            "max_consistency":  max(scores, default=0.0),
        }

    # ── full report ───────────────────────────────────────────────────────────

    def full_report(
        self,
        test_data_path: str,
        num_samples: int = 5,
        sample_len:  int = 300,
        batch_size:  int = 8,
    ) -> Dict:
        with open(test_data_path, encoding="utf-8") as f:
            test_text = f.read()

        print("[Evaluator] Computing loss / perplexity…")
        loss_metrics = self.compute_loss(test_text, batch_size=batch_size)

        print("[Evaluator] Computing vocabulary coverage…")
        cov_metrics = self.vocab_coverage(test_text)

        print(f"[Evaluator] Generating {num_samples} samples for diversity…")
        div_metrics = self.generation_diversity(
            num_samples=num_samples, sample_len=sample_len
        )

        print("[Evaluator] Computing format consistency…")
        fmt_metrics = self.format_consistency(
            test_text, num_samples=num_samples, sample_len=sample_len
        )

        return {
            "loss":         loss_metrics,
            "coverage":     cov_metrics,
            "diversity":    div_metrics,
            "consistency":  fmt_metrics,
        }

    # ── pretty print ──────────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: Dict, show_samples: bool = True):
        sep = "═" * 58
        print(f"\n{sep}")
        print("  CustomLLM — Evaluation Report")
        print(sep)

        L = report["loss"]
        print(f"\n  ▸ LANGUAGE MODELLING")
        print(f"    Loss (cross-entropy) : {L['loss']:.4f}")
        print(f"    Perplexity (PPL)     : {L['ppl']:.2f}")
        print(f"    Bits-per-token (BPT) : {L['bpt']:.4f}")
        print(f"    Bits-per-char  (BPC) : {L['bpc']:.4f}")
        print(f"    Tokens evaluated     : {L['n_tokens']:,}")

        C = report["coverage"]
        print(f"\n  ▸ VOCABULARY COVERAGE")
        print(f"    Coverage             : {C['coverage']*100:.2f}%")
        print(f"    Unknown tokens       : {C['unk_count']:,}")

        D = report["diversity"]
        print(f"\n  ▸ GENERATION DIVERSITY")
        print(f"    Distinct-1 (tokens)  : {D['dist1_tok']:.4f}")
        print(f"    Distinct-2 (tokens)  : {D['dist2_tok']:.4f}")
        print(f"    Distinct-3 (tokens)  : {D['dist3_tok']:.4f}")
        print(f"    Distinct-1 (chars)   : {D['dist1_chr']:.4f}")
        print(f"    Distinct-2 (chars)   : {D['dist2_chr']:.4f}")
        print(f"    Repetition rate      : {D['rep_rate']:.4f}")
        print(f"    Avg generated length : {D['avg_len']:.0f} chars")

        F = report["consistency"]
        pct = F['mean_consistency'] * 100
        print(f"\n  ▸ FORMAT CONSISTENCY (vs reference)")
        print(f"    Mean score           : {pct:.2f}%")
        print(f"    Min / Max            : {F['min_consistency']*100:.1f}% / {F['max_consistency']*100:.1f}%")

        if show_samples and "samples" in D:
            print(f"\n  ▸ SAMPLE GENERATIONS (temperature=0.8)")
            print(f"  {'─'*54}")
            for i, s in enumerate(D["samples"][:2]):
                print(f"\n  [Sample {i+1}]")
                preview = s[:400].replace("\n", "↵\n    ")
                print(f"    {preview}")

        print(f"\n{sep}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate a trained CustomLLM checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--tokenizer",    required=True)
    p.add_argument("--test_data",    required=True,  help="Path to test text file")
    p.add_argument("--num_samples",  type=int, default=5)
    p.add_argument("--sample_len",   type=int, default=300)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       default="auto")
    p.add_argument("--no_samples",   action="store_true", help="Hide sample outputs")
    args = p.parse_args()

    ev = Evaluator.from_checkpoint(args.checkpoint, args.tokenizer, args.device)
    report = ev.full_report(
        args.test_data,
        num_samples = args.num_samples,
        sample_len  = args.sample_len,
        batch_size  = args.batch_size,
    )
    ev.print_report(report, show_samples=not args.no_samples)
