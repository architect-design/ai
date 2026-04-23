# tokenizer.py
import json
import os


class FormatTokenizer:
    def __init__(self):
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def fit(self, raw_text_data):
        """Builds vocabulary from the raw text content."""
        unique_chars = sorted(list(set(raw_text_data)))
        self.char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text):
        return [self.char_to_int[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.int_to_char[i] for i in indices])

    def get_start_char(self):
        if self.vocab_size > 0:
            return list(self.char_to_int.keys())[0]
        return ""

    def save(self, filepath):
        data = {
            "char_to_int": self.char_to_int,
            "int_to_char": self.int_to_char,
            "vocab_size": self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        # FIX: JSON saves keys as strings. We must convert them back to integers.
        # int_to_char keys must be integers (e.g., 11 -> 'a')
        self.int_to_char = {int(k): v for k, v in data["int_to_char"].items()}

        # char_to_int values must be integers
        self.char_to_int = data["char_to_int"]

        self.vocab_size = data["vocab_size"]