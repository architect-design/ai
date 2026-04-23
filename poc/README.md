# CustomLLM — GPT-style Language Model From Scratch

A complete, production-quality implementation of a **decoder-only transformer**
language model written entirely in Python + PyTorch — no pretrained weights,
no external LLM APIs. Every component (attention, FFN, tokenizer, trainer,
generation) is implemented from scratch.

---

## Architecture

```
Input tokens (B, T)
       │
       ├─ TokenEmbedding      (vocab_size → n_embd)
       ├─ PositionalEmbedding (context_length → n_embd)
       └─ Dropout
              │
       ┌──────┴──────┐  × N layers
       │ TransformerBlock    │
       │   LayerNorm         │
       │   CausalMHAttention │  ← masked so each position
       │   (residual add)    │    can only attend to the past
       │   LayerNorm         │
       │   FeedForward       │  ← 4× expansion, GELU, dropout
       │   (residual add)    │
       └──────┬──────┘
              │
       LayerNorm (final)
              │
       Linear head (n_embd → vocab_size)
       Weight-tied with token embedding
              │
       Softmax → next-token probabilities
```

| Hyper-parameter        | Default | Description                         |
|------------------------|---------|-------------------------------------|
| `context_length`       | 256     | Tokens the model can attend to      |
| `n_embd`               | 384     | Embedding / hidden dimension        |
| `n_head`               | 6       | Attention heads (n_embd % n_head=0) |
| `n_layer`              | 6       | Transformer blocks                  |
| `dropout`              | 0.1     | Dropout probability                 |
| Default param count    | ~10 M   | Scales with d_model and n_layer     |

---

## Project Structure

```
custom_llm/
├── config.py       — ModelConfig dataclass (all hyperparameters)
├── tokenizer.py    — CharTokenizer + BPETokenizer (trained from scratch)
├── dataset.py      — TokenDataset + DataLoader factory
├── model.py        — CustomLLM (full transformer implementation)
├── trainer.py      — Training loop with cosine LR, grad clipping, logging
├── generate.py     — Generator class + interactive REPL + CLI
├── train.py        — Main training entry point (CLI)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch
```

### 2. Prepare your training data

Any plain-text file works — logs, code, prose, structured records,
gene sequences, custom domain formats, etc.

```bash
mkdir data
echo "your training data here" > data/myformat.txt
```

### 3. Train

```bash
python train.py \
    --data data/myformat.txt \
    --spec_name MY_FORMAT \
    --tokenizer char \
    --context_len 256 \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
    --max_iters 5000 \
    --batch_size 32
```

Checkpoints are saved to `checkpoints/MY_FORMAT/`.

### 4. Generate

```bash
python generate.py \
    --checkpoint checkpoints/MY_FORMAT/best_model.pt \
    --tokenizer  checkpoints/MY_FORMAT/tokenizer.json \
    --prompt     "your seed text" \
    --max_tokens 300 \
    --temperature 0.8
```

### 5. Interactive REPL

```bash
python generate.py \
    --checkpoint checkpoints/MY_FORMAT/best_model.pt \
    --tokenizer  checkpoints/MY_FORMAT/tokenizer.json \
    --interactive
```

---

## Using as a Python Library

```python
from config    import ModelConfig
from tokenizer import CharTokenizer
from dataset   import build_loaders
from model     import CustomLLM
from trainer   import Trainer
from generate  import Generator

# ── Build tokenizer ──────────────────────────────────────────────────────
text      = open("data/myformat.txt").read()
tokenizer = CharTokenizer().train(text)

# ── Configure model ───────────────────────────────────────────────────────
config = ModelConfig(
    vocab_size     = tokenizer.vocab_size,
    context_length = 256,
    n_embd         = 384,
    n_head         = 6,
    n_layer        = 6,
    max_iters      = 3000,
    out_dir        = "checkpoints/my_spec",
)

# ── Build dataloaders ────────────────────────────────────────────────────
train_loader, val_loader = build_loaders(text, tokenizer, config)

# ── Build and train model ─────────────────────────────────────────────────
model   = CustomLLM(config).to("cuda")
trainer = Trainer(model, config, train_loader, val_loader, "cuda")
trainer.train()

# ── Generate new data ─────────────────────────────────────────────────────
gen  = Generator.from_checkpoint(
    "checkpoints/my_spec/best_model.pt",
    "checkpoints/my_spec/tokenizer.json",
)
text = gen.generate(prompt="seed text", max_new_tokens=200, temperature=0.8)
print(text)
```

---

## Tokenizer Options

| Tokenizer      | Best for                                    | Vocab size    |
|----------------|---------------------------------------------|---------------|
| `CharTokenizer`| Structured / short-vocab formats, code, DNA | chars in corpus |
| `BPETokenizer` | Natural language, larger diverse corpora    | Configurable  |

```bash
# Character-level (default, recommended for custom formats)
python train.py --data data/myformat.txt --tokenizer char

# Byte-Pair Encoding with 2000 merges
python train.py --data data/myformat.txt --tokenizer bpe --bpe_vocab 2000
```

---

## Generation Parameters

| Parameter           | Effect                                              |
|---------------------|-----------------------------------------------------|
| `temperature`       | 0.1 = deterministic/strict, 1.0 = balanced, 2.0 = wild |
| `top_k`             | Keep only the K most likely next tokens (50 = good) |
| `top_p`             | Nucleus filtering — only tokens summing to P prob   |
| `repetition_penalty`| >1.0 discourages repeating the same tokens         |

---

## Model Size Presets

| Size      | n_embd | n_head | n_layer | ~Params |
|-----------|--------|--------|---------|---------|
| Tiny      | 128    | 4      | 4       | ~1 M    |
| Small     | 256    | 4      | 4       | ~3 M    |
| Medium    | 384    | 6      | 6       | ~10 M   |
| Large     | 512    | 8      | 8       | ~25 M   |
| XL        | 768    | 12     | 12      | ~85 M   |

---

## Resume Training

```bash
python train.py \
    --data data/myformat.txt \
    --spec_name MY_FORMAT \
    --resume checkpoints/MY_FORMAT/best_model.pt \
    --max_iters 10000
```

---

## Output Files

After training, `checkpoints/<spec_name>/` contains:

| File                | Description                                    |
|---------------------|------------------------------------------------|
| `best_model.pt`     | Checkpoint with lowest validation loss         |
| `final_model.pt`    | Checkpoint at the final training iteration     |
| `tokenizer.json`    | Vocabulary and merge rules                     |
| `config.json`       | All model & training hyperparameters           |
| `training_log.csv`  | iter, train_loss, val_loss, lr, elapsed per eval|

---

## Notes

- **No external LLMs are used.** Every weight is learned from your data.
- Training on CPU is supported but slow for large models. GPU recommended.
- Apple Silicon (MPS) is supported via `--device mps`.
- Gradient checkpointing is not implemented — reduce batch size if OOM.
