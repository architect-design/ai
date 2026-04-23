"""
tokenizer.py — CharTokenizer and BPETokenizer for CustomLLM
"""
import json, os, re
from collections import Counter
from typing import List, Dict, Tuple, Optional


# ─── Base ─────────────────────────────────────────────────────────────────────

class BaseTokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    SPECIALS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError


# ─── Character-Level ──────────────────────────────────────────────────────────

class CharTokenizer(BaseTokenizer):
    """
    Simplest possible tokenizer: every unique character is one token.
    Fast to train, perfectly lossless, good for small/structured corpora.
    """

    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0

    def train(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        offset = len(self.SPECIALS)
        self.stoi = dict(self.SPECIALS)
        self.itos = {v: k for k, v in self.SPECIALS.items()}
        for i, ch in enumerate(chars):
            self.stoi[ch] = i + offset
            self.itos[i + offset] = ch
        self.vocab_size = len(self.stoi)
        print(f"[CharTokenizer] vocab_size={self.vocab_size}  "
              f"unique_chars={len(chars)}")
        return self

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.UNK) for ch in text]

    def decode(self, ids: List[int]) -> str:
        skip = {self.PAD, self.BOS, self.EOS}
        return "".join(self.itos.get(i, "?") for i in ids if i not in skip)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": "char",
                 "stoi": self.stoi,
                 "itos": {str(k): v for k, v in self.itos.items()}},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ─── Byte-Pair Encoding ───────────────────────────────────────────────────────

class BPETokenizer(BaseTokenizer):
    """
    Minimal Byte-Pair Encoding tokenizer trained from scratch.
    Learns the most frequent character-pair merges from the corpus.

    Suitable for larger / natural-language datasets where CharTokenizer
    produces too many tokens per sentence.
    """

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}   # pair -> merged
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _word_freq(text: str) -> Counter:
        """Count word frequencies (space-split, basic)."""
        return Counter(text.split())

    @staticmethod
    def _get_pairs(vocab: Dict[Tuple[str, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for seq, freq in vocab.items():
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge(vocab: Dict[Tuple[str, ...], int],
               pair: Tuple[str, str],
               merged: str) -> Dict[Tuple[str, ...], int]:
        new_vocab: Dict[Tuple[str, ...], int] = {}
        for seq, freq in vocab.items():
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_vocab[tuple(new_seq)] = freq
        return new_vocab

    # ── public API ───────────────────────────────────────────────────────────

    def train(self, text: str) -> "BPETokenizer":
        word_freq = self._word_freq(text)
        # Represent each word as a char tuple with end-of-word marker
        vocab: Dict[Tuple[str, ...], int] = {
            tuple(list(w) + ["</w>"]): f for w, f in word_freq.items()
        }

        # Seed vocab from unique characters
        chars: set = set()
        for seq in vocab:
            chars.update(seq)

        self.stoi = dict(self.SPECIALS)
        for ch in sorted(chars):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}

        num_merges = self.target_vocab_size - len(self.stoi)
        print(f"[BPETokenizer] base_vocab={len(self.stoi)}  "
              f"merges_planned={num_merges}  target={self.target_vocab_size}")

        for step in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merged = "".join(best)
            vocab = self._merge(vocab, best, merged)
            self.merges[best] = merged
            new_id = len(self.stoi)
            self.stoi[merged] = new_id
            self.itos[new_id] = merged

            if (step + 1) % 200 == 0 or step == num_merges - 1:
                print(f"  step {step+1}/{num_merges}: "
                      f"'{best[0]}'+'{best[1]}' → '{merged}'  "
                      f"freq={pairs[best]}")

        self.vocab_size = len(self.stoi)
        print(f"[BPETokenizer] final vocab_size={self.vocab_size}")
        return self

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word."""
        symbols = list(word) + ["</w>"]
        changed = True
        while changed and len(symbols) > 1:
            changed = False
            new_syms: List[str] = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                        (symbols[i], symbols[i + 1]) in self.merges):
                    new_syms.append(self.merges[(symbols[i], symbols[i + 1])])
                    i += 2
                    changed = True
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        return symbols

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.split():
            for tok in self._tokenize_word(word):
                tokens.append(self.stoi.get(tok, self.UNK))
        return tokens

    def decode(self, ids: List[int]) -> str:
        skip = {self.PAD, self.BOS, self.EOS}
        raw = "".join(self.itos.get(i, "?") for i in ids if i not in skip)
        return raw.replace("</w>", " ").strip()

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": "bpe",
                 "vocab_size": self.vocab_size,
                 "target_vocab_size": self.target_vocab_size,
                 "merges": {f"{a}\t{b}": m for (a, b), m in self.merges.items()},
                 "stoi": self.stoi,
                 "itos": {str(k): v for k, v in self.itos.items()}},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("target_vocab_size", 1000))
        tok.merges = {
            tuple(k.split("\t")): v for k, v in data["merges"].items()
        }
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok


# ─── Factory ──────────────────────────────────────────────────────────────────

def load_tokenizer(path: str) -> BaseTokenizer:
    """Auto-detect and load the correct tokenizer from a saved JSON file."""
    with open(path, encoding="utf-8") as f:
        kind = json.load(f).get("type", "char")
    if kind == "char":
        return CharTokenizer.load(path)
    elif kind == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"Unknown tokenizer type: {kind!r}")
