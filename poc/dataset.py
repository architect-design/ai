"""
dataset.py — Dataset helpers for CustomLLM training
"""
import os
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ─── Core Dataset ─────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Sliding-window dataset over a 1-D integer token sequence.

    Each sample is (x, y) where:
        x = tokens[i : i + context_length]
        y = tokens[i+1 : i + context_length + 1]   ← shifted by 1 (next-token targets)
    """

    def __init__(self, tokens: List[int], context_length: int):
        self.data           = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


# ─── Multi-file Dataset ───────────────────────────────────────────────────────

class MultiFileDataset(Dataset):
    """
    Loads and concatenates multiple text files into one token stream,
    then wraps it in a sliding-window view.
    Useful when training data spans many documents / spec examples.
    """

    def __init__(self, file_paths: List[str], tokenizer, context_length: int):
        tokens: List[int] = []
        for path in file_paths:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            tokens.extend(tokenizer.encode(text))
            tokens.append(tokenizer.EOS)          # document boundary
        self._inner = TokenDataset(tokens, context_length)
        self.context_length = context_length
        print(f"[MultiFileDataset] {len(file_paths)} files → "
              f"{len(tokens):,} tokens → {len(self._inner):,} samples")

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._inner[idx]


# ─── Factory function ─────────────────────────────────────────────────────────

def build_loaders(
    text: str,
    tokenizer,
    config,
    pin_memory: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tokenise `text`, split train/val, return DataLoaders.

    Args:
        text       : raw corpus string
        tokenizer  : a CharTokenizer or BPETokenizer instance
        config     : ModelConfig (uses config.train_split, context_length, batch_size)
        pin_memory : speed up GPU transfers
        num_workers: parallel workers (0 = main process, safest on Windows)

    Returns:
        (train_loader, val_loader)
    """
    tokens = tokenizer.encode(text)
    print(f"[Dataset] {len(text):,} chars → {len(tokens):,} tokens")

    split = int(len(tokens) * config.train_split)
    train_ds = TokenDataset(tokens[:split],  config.context_length)
    val_ds   = TokenDataset(tokens[split:],  config.context_length)

    print(f"[Dataset] train={len(train_ds):,}  val={len(val_ds):,}  "
          f"(split={config.train_split:.0%})")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty. "
            "Your text is too short for the chosen context_length. "
            f"Need at least {config.context_length + 1} tokens; "
            f"got {len(tokens)}."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )
    return train_loader, val_loader


def load_text_files(paths: List[str], separator: str = "\n\n") -> str:
    """Read and concatenate one or more text files."""
    parts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            parts.append(f.read())
        print(f"[load_text_files] {p}  ({os.path.getsize(p):,} bytes)")
    return separator.join(parts)
