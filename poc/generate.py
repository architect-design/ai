"""
generate.py — Inference / text generation for CustomLLM

Usage (CLI):
    python generate.py \
        --checkpoint checkpoints/my_spec/best_model.pt \
        --tokenizer  checkpoints/my_spec/tokenizer.json \
        --prompt     "your seed text" \
        --max_tokens 300 \
        --temperature 0.8 \
        --top_k 50 \
        --top_p 0.95

Usage (API):
    from generate import Generator

    gen = Generator.from_checkpoint(
        checkpoint="checkpoints/my_spec/best_model.pt",
        tokenizer_path="checkpoints/my_spec/tokenizer.json",
    )
    text = gen.generate("seed text", max_new_tokens=200)
    print(text)
"""

import argparse
import sys
import torch
from typing import Optional

from model import CustomLLM
from tokenizer import load_tokenizer


# ─── Generator class ──────────────────────────────────────────────────────────

class Generator:
    """
    High-level wrapper around CustomLLM for convenient text generation.
    Handles device placement, prompt encoding, and decoding automatically.
    """

    def __init__(self, model: CustomLLM, tokenizer, device: str):
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = device

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        tokenizer_path: str,
        device: str = "auto",
    ) -> "Generator":
        if device == "auto":
            device = ("cuda"  if torch.cuda.is_available() else
                      "mps"   if torch.backends.mps.is_available() else
                      "cpu")
        print(f"[Generator] device={device}")
        model     = CustomLLM.from_checkpoint(checkpoint, device=device)
        tokenizer = load_tokenizer(tokenizer_path)
        return cls(model, tokenizer, device)

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        random_seed: Optional[int] = None,
    ) -> str:
        """
        Generate text continuation from `prompt`.

        Args:
            prompt             : seed text (empty → start from BOS token)
            max_new_tokens     : number of tokens to generate
            temperature        : sampling temperature (lower = more deterministic)
            top_k              : keep only the top-k most likely next tokens (0=off)
            top_p              : nucleus filtering probability (1.0 = off)
            repetition_penalty : penalise recently seen tokens (1.0 = off)
            random_seed        : set for reproducible outputs

        Returns:
            Generated string (includes the prompt prefix).
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Encode prompt
        if prompt:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [self.tokenizer.BOS]

        # Trim to context window (keep tail)
        max_ctx = self.model.config.context_length
        if len(ids) > max_ctx:
            print(f"[Generator] Prompt truncated from {len(ids)} to {max_ctx} tokens")
            ids = ids[-max_ctx:]

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        out_ids = self.model.generate(
            idx,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k  if top_k  > 0   else None,
            top_p              = top_p  if top_p  < 1.0 else None,
            repetition_penalty = repetition_penalty,
        )

        # Decode only the newly generated portion
        new_ids = out_ids[0, len(ids):].tolist()
        return prompt + self.tokenizer.decode(new_ids)

    def interactive(self, **kwargs):
        """Simple REPL for interactive generation."""
        print("\nCustomLLM Interactive Generator  (Ctrl-C to quit)")
        print("─" * 50)
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not prompt:
                continue
            result = self.generate(prompt, **kwargs)
            print("\n" + "─" * 50)
            print(result)
            print("─" * 50)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate text with a trained CustomLLM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint")
    p.add_argument("--tokenizer",    required=True,  help="Path to tokenizer.json")
    p.add_argument("--prompt",       default="",     help="Seed text (empty = BOS)")
    p.add_argument("--max_tokens",   type=int,   default=200,  help="Tokens to generate")
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top_k",        type=int,   default=50,   help="0 = disabled")
    p.add_argument("--top_p",        type=float, default=0.95, help="1.0 = disabled")
    p.add_argument("--rep_penalty",  type=float, default=1.1,  help="Repetition penalty")
    p.add_argument("--num_samples",  type=int,   default=1)
    p.add_argument("--seed",         type=int,   default=None, help="Random seed")
    p.add_argument("--device",       default="auto")
    p.add_argument("--interactive",  action="store_true",
                   help="Launch an interactive REPL instead of single generation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gen = Generator.from_checkpoint(
        checkpoint     = args.checkpoint,
        tokenizer_path = args.tokenizer,
        device         = args.device,
    )

    kwargs = dict(
        max_new_tokens     = args.max_tokens,
        temperature        = args.temperature,
        top_k              = args.top_k,
        top_p              = args.top_p,
        repetition_penalty = args.rep_penalty,
        random_seed        = args.seed,
    )

    if args.interactive:
        gen.interactive(**kwargs)
        sys.exit(0)

    sep = "─" * 60
    for i in range(args.num_samples):
        print(f"\n{sep}")
        if args.num_samples > 1:
            print(f"  Sample {i + 1} / {args.num_samples}")
            print(sep)
        seed_for_sample = None if args.seed is None else args.seed + i
        result = gen.generate(args.prompt, **{**kwargs, "random_seed": seed_for_sample})
        print(result)
    print(f"\n{sep}")
