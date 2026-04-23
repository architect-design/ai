"""
preprocess.py — Data preprocessing, cleaning, augmentation, and splitting
               for CustomLLM training corpora.

Usage (CLI):
    python preprocess.py \
        --input  data/raw/*.txt \
        --output data/processed/ \
        --spec_name MY_FORMAT \
        --dedup \
        --min_line_len 10 \
        --augment_factor 2 \
        --splits 0.9 0.05 0.05

Usage (API):
    from preprocess import Preprocessor
    pp   = Preprocessor(spec_name="MY_FORMAT", dedup=True)
    data = pp.run(["raw/file1.txt", "raw/file2.txt"])
    pp.save(data, "data/processed/")
"""

import argparse
import hashlib
import os
import random
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ─── Cleaning helpers ─────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC-normalize unicode, replace fancy quotes/dashes."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    return text


def normalize_whitespace(text: str, preserve_newlines: bool = True) -> str:
    """Collapse runs of spaces/tabs; optionally collapse newlines too."""
    text = re.sub(r"[ \t]+", " ", text)
    if not preserve_newlines:
        text = re.sub(r"\n+", "\n", text)
    return text.strip()


def remove_non_printable(text: str, keep_newline: bool = True) -> str:
    """Strip non-printable control characters."""
    result = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "\n" and keep_newline:
            result.append(ch)
        elif ch == "\t":
            result.append(" ")
        elif cat.startswith("C"):   # control / format / surrogate
            continue
        else:
            result.append(ch)
    return "".join(result)


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def random_slice(text: str, rng: random.Random,
                 min_ratio: float = 0.5, max_ratio: float = 0.95) -> str:
    """Extract a random substring of the text."""
    n = len(text)
    ratio  = rng.uniform(min_ratio, max_ratio)
    length = max(1, int(n * ratio))
    start  = rng.randint(0, max(0, n - length))
    return text[start : start + length]


def swap_lines(text: str, rng: random.Random,
               swap_prob: float = 0.05) -> str:
    """
    Randomly swap adjacent lines with probability `swap_prob`.
    Useful for teaching the model that line order matters (by seeing it broken).
    """
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if rng.random() < swap_prob:
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
    return "\n".join(lines)


def char_dropout(text: str, rng: random.Random,
                 drop_prob: float = 0.005) -> str:
    """Randomly drop individual characters (mild corruption for robustness)."""
    return "".join(ch for ch in text if rng.random() > drop_prob)


def repeat_section(text: str, rng: random.Random,
                   max_len: int = 500) -> str:
    """
    Extract a random block and duplicate it — teaches the model about
    repetitive structural patterns common in custom formats.
    """
    n = len(text)
    if n < 20:
        return text
    start  = rng.randint(0, n // 2)
    end    = min(start + rng.randint(10, max_len), n)
    block  = text[start:end]
    insert = rng.randint(0, n)
    return text[:insert] + block + text[insert:]


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_lines(text: str, case_sensitive: bool = True) -> Tuple[str, int]:
    """
    Remove exact duplicate lines.
    Returns (deduplicated text, number of duplicates removed).
    """
    seen   = set()
    result = []
    dupes  = 0
    for line in text.split("\n"):
        key = line if case_sensitive else line.lower()
        if key not in seen:
            seen.add(key)
            result.append(line)
        else:
            dupes += 1
    return "\n".join(result), dupes


def deduplicate_blocks(texts: List[str], block_size: int = 512) -> List[str]:
    """
    Remove near-duplicate chunks (by MD5 hash of `block_size`-char windows).
    Applied across the entire corpus list.
    """
    seen = set()
    out  = []
    for text in texts:
        blocks  = [text[i : i + block_size]
                   for i in range(0, len(text), block_size)]
        hashes  = {hashlib.md5(b.encode()).hexdigest() for b in blocks}
        overlap = hashes & seen
        if len(overlap) / max(len(hashes), 1) < 0.5:
            out.append(text)
            seen.update(hashes)
    removed = len(texts) - len(out)
    if removed:
        print(f"  [dedup-blocks] Removed {removed} near-duplicate documents")
    return out


# ─── Statistics ───────────────────────────────────────────────────────────────

def corpus_stats(text: str) -> Dict:
    lines  = text.split("\n")
    tokens = text.split()
    chars  = Counter(text)
    return {
        "characters":    len(text),
        "lines":         len(lines),
        "words":         len(tokens),
        "unique_chars":  len(chars),
        "avg_line_len":  sum(len(l) for l in lines) / max(len(lines), 1),
        "top_chars":     chars.most_common(10),
    }


def print_stats(stats: Dict, label: str = ""):
    sep = "─" * 52
    print(f"\n  {sep}")
    if label:
        print(f"  {label}")
    print(f"  Characters   : {stats['characters']:>10,}")
    print(f"  Lines        : {stats['lines']:>10,}")
    print(f"  Words        : {stats['words']:>10,}")
    print(f"  Unique chars : {stats['unique_chars']:>10,}")
    print(f"  Avg line len : {stats['avg_line_len']:>10.1f}")
    print(f"  {sep}")


# ─── Split helpers ────────────────────────────────────────────────────────────

def split_text(text: str,
               train: float = 0.9,
               val: float   = 0.05,
               test: float  = 0.05) -> Tuple[str, str, str]:
    """Split text into train / val / test at character level."""
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    n  = len(text)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return text[:t1], text[t1:t2], text[t2:]


# ─── Preprocessor class ───────────────────────────────────────────────────────

class Preprocessor:
    """
    Full preprocessing pipeline for CustomLLM training data.

    Steps (in order):
        1. Load raw files
        2. Unicode normalisation
        3. Remove non-printable characters
        4. Whitespace normalisation
        5. Minimum line length filtering
        6. Deduplication (optional)
        7. Augmentation (optional)
        8. Train / val / test split
    """

    def __init__(
        self,
        spec_name:        str   = "custom",
        dedup:            bool  = True,
        min_line_len:     int   = 0,
        max_line_len:     int   = 0,          # 0 = unlimited
        preserve_newlines:bool  = True,
        augment_factor:   int   = 1,           # 1 = no augmentation
        aug_swap_prob:    float = 0.02,
        aug_drop_prob:    float = 0.003,
        seed:             int   = 42,
    ):
        self.spec_name         = spec_name
        self.dedup             = dedup
        self.min_line_len      = min_line_len
        self.max_line_len      = max_line_len
        self.preserve_newlines = preserve_newlines
        self.augment_factor    = augment_factor
        self.aug_swap_prob     = aug_swap_prob
        self.aug_drop_prob     = aug_drop_prob
        self.rng               = random.Random(seed)

    # ── internal ──────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = normalize_unicode(text)
        text = remove_non_printable(text, keep_newline=self.preserve_newlines)
        text = normalize_whitespace(text, self.preserve_newlines)

        if self.min_line_len > 0 or self.max_line_len > 0:
            lines = []
            for line in text.split("\n"):
                if self.min_line_len and len(line) < self.min_line_len:
                    continue
                if self.max_line_len and len(line) > self.max_line_len:
                    line = line[:self.max_line_len]
                lines.append(line)
            text = "\n".join(lines)

        return text.strip()

    def _augment(self, text: str) -> List[str]:
        """Return `augment_factor - 1` augmented variants of `text`."""
        variants = []
        for _ in range(self.augment_factor - 1):
            v = text
            # random combination of augmentations
            choice = self.rng.random()
            if choice < 0.3:
                v = random_slice(v, self.rng)
            elif choice < 0.6:
                v = swap_lines(v, self.rng, self.aug_swap_prob)
            elif choice < 0.8:
                v = char_dropout(v, self.rng, self.aug_drop_prob)
            else:
                v = repeat_section(v, self.rng)
            variants.append(v)
        return variants

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Load, clean, optionally augment, and split.

        Returns:
            {"train": ..., "val": ..., "test": ...}
        """
        print(f"\n[Preprocessor] spec='{self.spec_name}'  files={len(file_paths)}")

        # 1. Load
        docs: List[str] = []
        for p in file_paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            docs.append(raw)
            print(f"  Loaded: {p}  ({len(raw):,} chars)")

        # 2. Clean each document
        docs = [self._clean(d) for d in docs]
        docs = [d for d in docs if d]          # drop empty

        # 3. Block-level dedup across files
        if self.dedup and len(docs) > 1:
            docs = deduplicate_blocks(docs)

        # 4. Concatenate into one corpus
        corpus = "\n\n".join(docs)

        # 5. Line-level dedup
        if self.dedup:
            corpus, n_dupes = deduplicate_lines(corpus)
            print(f"  [dedup-lines] Removed {n_dupes:,} duplicate lines")

        print_stats(corpus_stats(corpus), label="After cleaning")

        # 6. Augmentation
        if self.augment_factor > 1:
            print(f"  [augment] factor={self.augment_factor}  "
                  f"generating {self.augment_factor - 1} synthetic variants…")
            extras = self._augment(corpus)
            corpus = corpus + "\n\n" + "\n\n".join(extras)
            print(f"  [augment] corpus grew to {len(corpus):,} chars")

        return corpus

    def save(
        self,
        corpus: str,
        out_dir: str,
        splits: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    ) -> Dict[str, str]:
        """
        Split corpus and write train/val/test text files to `out_dir`.
        Returns the split texts as a dict.
        """
        os.makedirs(out_dir, exist_ok=True)
        train_t, val_t, test_t = split_text(corpus, *splits)

        files = {
            "train": train_t,
            "val":   val_t,
            "test":  test_t,
        }
        for name, text in files.items():
            path = os.path.join(out_dir, f"{self.spec_name}_{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Wrote {name}: {path}  ({len(text):,} chars)")

        return files


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess text data for CustomLLM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      nargs="+", required=True, help="Input text files")
    p.add_argument("--output",     default="data/processed",  help="Output directory")
    p.add_argument("--spec_name",  default="custom")
    p.add_argument("--dedup",      action="store_true",       help="Deduplicate lines/blocks")
    p.add_argument("--min_line_len", type=int, default=0)
    p.add_argument("--max_line_len", type=int, default=0)
    p.add_argument("--augment_factor", type=int, default=1,
                   help="Number of total copies (1 = no augmentation)")
    p.add_argument("--splits", type=float, nargs=3,
                   default=[0.9, 0.05, 0.05],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pp = Preprocessor(
        spec_name       = args.spec_name,
        dedup           = args.dedup,
        min_line_len    = args.min_line_len,
        max_line_len    = args.max_line_len,
        augment_factor  = args.augment_factor,
        seed            = args.seed,
    )
    corpus = pp.run(args.input)
    pp.save(corpus, args.output, splits=tuple(args.splits))
    print("\n[Preprocessor] Done.")
