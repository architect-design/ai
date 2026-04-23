"""
model.py — CustomLLM: GPT-style Decoder-only Transformer from scratch
         Built with PyTorch. No pretrained weights. No external LLMs.

Architecture
────────────
  TokenEmbedding + PositionalEmbedding
  → N × TransformerBlock
      ├── LayerNorm
      ├── CausalMultiHeadAttention  (masked self-attention)
      ├── LayerNorm
      └── FeedForward (GELU, 4× expansion)
  → LayerNorm
  → Linear head  (weight-tied with token embedding)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── Causal Multi-Head Self-Attention ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Scaled dot-product attention with a causal (autoregressive) mask.
    Keys, queries and values are computed in a single fused projection.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale    = math.sqrt(self.head_dim)

        # Fused Q, K, V projection
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Lower-triangular causal mask (registered as a non-parameter buffer)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) / self.scale          # (B, nh, T, T)
        # Apply causal mask — future tokens become -inf → 0 after softmax
        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values
        y = att @ v                                             # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # (B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


# ─── Feed-Forward Block ───────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP with GELU activation.
    Inner dimension is 4× the embedding size (as in the original Transformer).
    """

    def __init__(self, config):
        super().__init__()
        inner = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, inner, bias=config.bias),
            nn.GELU(),
            nn.Linear(inner, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One transformer block with pre-LayerNorm (more stable than post-norm):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ff   = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ff(self.ln2(x))     # residual + FFN
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class CustomLLM(nn.Module):
    """
    GPT-style decoder-only language model trained entirely from scratch.

    Usage:
        config = ModelConfig(vocab_size=tokenizer.vocab_size, ...)
        model  = CustomLLM(config).to(device)

        # Training
        logits, loss = model(x, targets=y)
        loss.backward()

        # Inference
        output = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8)
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size > 0,    "vocab_size must be set before building the model"
        assert config.context_length > 0
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),        # token embeddings
            wpe  = nn.Embedding(config.context_length, config.n_embd),   # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),         # final norm
        ))

        # Language modelling head (no bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share weights.
        # This halves the parameter count and often improves perplexity.
        self.transformer.wte.weight = self.lm_head.weight

        # Weight initialisation
        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

        n = self.num_params()
        print(f"[CustomLLM] Initialised — {n:,} parameters  "
              f"({n/1e6:.2f}M)  "
              f"[{config.n_layer}L·{config.n_head}H·{config.n_embd}D]")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def get_num_params(self) -> int:  # alias
        return self.num_params()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx     : (B, T) long tensor of token ids
            targets : (B, T) long tensor of target ids (optional; for loss)

        Returns:
            logits  : (B, T, vocab_size)  — or (B, 1, vocab_size) during inference
            loss    : cross-entropy scalar if targets given, else None
        """
        B, T = idx.shape
        assert T <= self.config.context_length, (
            f"Sequence length {T} > context_length {self.config.context_length}"
        )
        device = idx.device
        pos    = torch.arange(T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd) → broadcast
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Full logit matrix for training
            logits = self.lm_head(x)                              # (B, T, V)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Only the last position during auto-regressive inference
            logits = self.lm_head(x[:, [-1], :])                 # (B, 1, V)
            loss   = None

        return logits, loss

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Auto-regressive text generation.

        Args:
            idx               : (1, T) or (B, T) prompt token ids
            max_new_tokens    : how many tokens to generate
            temperature       : >1 → more random, <1 → sharper
            top_k             : keep only the top-k logits (0 = disabled)
            top_p             : nucleus filtering (1.0 = disabled)
            repetition_penalty: penalise previously seen tokens (1.0 = disabled)

        Returns:
            (B, T + max_new_tokens) tensor of token ids
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = (idx if idx.size(1) <= self.config.context_length
                        else idx[:, -self.config.context_length:])

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    for prev in set(idx[b].tolist()):
                        logits[b, prev] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k is not None and top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens past the nucleus threshold
                sorted_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx      = torch.cat([idx, idx_next], dim=1)

        return idx

    # ── serialisation ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        """Save model weights + config (and any extra metadata)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "config":      self.config,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[CustomLLM] Checkpoint saved → {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "CustomLLM":
        """Load a model from a checkpoint file."""
        ckpt   = torch.load(path, map_location=device)
        model  = cls(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model  = model.to(device)
        print(f"[CustomLLM] Loaded from {path}  ({model.num_params():,} params)")
        return model

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CustomLLM(\n"
            f"  layers  = {cfg.n_layer}\n"
            f"  heads   = {cfg.n_head}\n"
            f"  d_model = {cfg.n_embd}\n"
            f"  context = {cfg.context_length}\n"
            f"  vocab   = {cfg.vocab_size}\n"
            f"  params  = {self.num_params():,}\n"
            f")"
        )
