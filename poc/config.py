"""
config.py — ModelConfig for CustomLLM
"""
from dataclasses import dataclass, field
import json, os


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size:      int   = 256    # Set automatically after tokenizer is built
    context_length:  int   = 256    # Max tokens the model attends to
    n_embd:          int   = 384    # Embedding / hidden dimension
    n_head:          int   = 6      # Number of attention heads (n_embd % n_head == 0)
    n_layer:         int   = 6      # Number of stacked transformer blocks
    dropout:         float = 0.1    # Dropout probability
    bias:            bool  = False  # Bias in Linear/LayerNorm (False = faster, GPT-2 style)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:      int   = 32
    max_iters:       int   = 5000
    eval_interval:   int   = 200    # Evaluate every N iterations
    eval_iters:      int   = 50     # Batches used for loss estimation
    learning_rate:   float = 3e-4
    weight_decay:    float = 0.1
    beta1:           float = 0.9
    beta2:           float = 0.95
    grad_clip:       float = 1.0    # 0 = disabled
    warmup_iters:    int   = 200
    lr_decay_iters:  int   = 5000   # Should match max_iters
    min_lr:          float = 3e-5   # ~10% of learning_rate

    # ── Data ─────────────────────────────────────────────────────────────────
    train_split:     float = 0.9

    # ── Generation defaults ───────────────────────────────────────────────────
    temperature:     float = 0.8
    top_k:           int   = 50
    top_p:           float = 0.95

    # ── I/O ──────────────────────────────────────────────────────────────────
    out_dir:         str   = "checkpoints"
    spec_name:       str   = "custom"

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))

    def __repr__(self) -> str:
        lines = [f"ModelConfig(spec='{self.spec_name}')"]
        lines.append(f"  architecture : {self.n_layer}L × {self.n_head}H × {self.n_embd}D")
        lines.append(f"  context      : {self.context_length} tokens")
        lines.append(f"  vocab        : {self.vocab_size}")
        lines.append(f"  training     : bs={self.batch_size}, iters={self.max_iters}, lr={self.learning_rate}")
        return "\n".join(lines)
