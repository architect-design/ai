"""
Custom LLM Spec Engine
======================
A from-scratch rule-learning and inference engine that learns financial
file format specifications (NACHA, VISA VCF, Oracle GL, Custom) from
uploaded documents and stores them as structured knowledge bases.

No external LLM is used. The engine uses:
  - Statistical token analysis
  - Regex-based pattern extraction
  - TF-IDF-style term weighting
  - Rule inference and validation
"""

import re
import json
import os
import math
import hashlib
import logging
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_STORE = os.path.join(BASE_DIR, "models", "trained_specs")
os.makedirs(SPEC_STORE, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. TOKENISER
# ══════════════════════════════════════════════════════════════════════════
class FinancialTokenizer:
    """Tokenises financial-spec text into structured tokens."""

    # Patterns that signal a field definition line
    FIELD_PATTERNS = [
        r'(?P<name>[\w\s]+?)\s+(?P<pos>\d+[-–]\d+)\s+(?P<len>\d+)\s+(?P<type>\w+)',
        r'(?P<pos>\d+)\s+(?P<name>[\w\s]+?)\s+(?P<len>\d+)\s+(?P<type>\w+)',
        r'Field\s+(?P<num>\d+)[\s:]+(?P<name>[^\n]+)',
        r'(?P<name>[A-Z][A-Z\s_]+)\s*[:=]\s*(?P<desc>[^\n]+)',
    ]

    DATA_TYPE_MAP = {
        r'\b(numeric|num|n)\b':         'numeric',
        r'\b(alpha|alphanumeric|an|a)\b':'alphanumeric',
        r'\b(date)\b':                   'date',
        r'\b(amount|amt)\b':             'amount',
        r'\b(boolean|bool|flag)\b':      'boolean',
        r'\b(blank|filler|reserved)\b':  'filler',
        r'\b(routing|aba|rtn)\b':        'routing_number',
        r'\b(account)\b':                'account_number',
    }

    def tokenize(self, text: str) -> list[dict]:
        tokens = []
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            tok = {
                'line':     i,
                'raw':      stripped,
                'words':    self._words(stripped),
                'numbers':  self._numbers(stripped),
                'data_type': self._infer_type(stripped),
                'is_field': self._is_field_line(stripped),
            }
            tokens.append(tok)
        return tokens

    def _words(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r'[A-Za-z_]+', text)]

    def _numbers(self, text: str) -> list[int]:
        return [int(n) for n in re.findall(r'\d+', text)]

    def _infer_type(self, text: str) -> str:
        lower = text.lower()
        for pattern, dtype in self.DATA_TYPE_MAP.items():
            if re.search(pattern, lower, re.IGNORECASE):
                return dtype
        return 'alphanumeric'

    def _is_field_line(self, text: str) -> bool:
        for pat in self.FIELD_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return True
        has_numbers = bool(re.search(r'\d+', text))
        has_words   = bool(re.search(r'[A-Za-z]{3,}', text))
        return has_numbers and has_words and len(text) < 200


# ══════════════════════════════════════════════════════════════════════════
# 2. PATTERN EXTRACTOR (learns from tokens)
# ══════════════════════════════════════════════════════════════════════════
class PatternExtractor:
    """Extracts structured field definitions from tokenised spec documents."""

    def extract_fields(self, tokens: list[dict], format_type: str) -> list[dict]:
        fields = []
        for tok in tokens:
            if not tok['is_field']:
                continue
            field = self._parse_field(tok['raw'], tok['data_type'])
            if field:
                field['source_line'] = tok['line']
                fields.append(field)

        # Deduplicate and clean
        fields = self._deduplicate(fields)
        fields = self._infer_missing(fields)
        return fields

    # ── Low-level parsers ─────────────────────────────────────────────
    def _parse_field(self, line: str, inferred_type: str) -> dict | None:
        # Try fixed-width pattern:  Name   1-10   10   AN
        m = re.search(
            r'(?P<name>[\w][\w\s\-_/]+?)'
            r'\s+(?P<start>\d+)[-–to ]+(?P<end>\d+)'
            r'\s+(?P<len>\d+)'
            r'(?:\s+(?P<type>\w+))?',
            line, re.IGNORECASE
        )
        if m:
            start = int(m.group('start'))
            end   = int(m.group('end'))
            length = int(m.group('len')) if m.group('len') else end - start + 1
            return {
                'name':       self._clean_name(m.group('name')),
                'start':      start,
                'end':        end,
                'length':     length,
                'data_type':  self._resolve_type(m.group('type'), inferred_type),
                'required':   self._is_required(line),
                'validation': self._extract_validation(line),
            }

        # Positional: pos  name  length  type
        m = re.search(
            r'(?P<pos>\d+)\s+'
            r'(?P<name>[A-Za-z][\w\s\-_/]+?)\s+'
            r'(?P<len>\d+)'
            r'(?:\s+(?P<type>\w+))?',
            line, re.IGNORECASE
        )
        if m:
            pos    = int(m.group('pos'))
            length = int(m.group('len'))
            return {
                'name':      self._clean_name(m.group('name')),
                'start':     pos,
                'end':       pos + length - 1,
                'length':    length,
                'data_type': self._resolve_type(m.group('type'), inferred_type),
                'required':  self._is_required(line),
                'validation': self._extract_validation(line),
            }

        # Key-value:  FIELD_NAME: description / type
        m = re.search(r'^([A-Z][A-Z0-9_\s]{2,}?):\s+(.+)$', line)
        if m:
            return {
                'name':      self._clean_name(m.group(1)),
                'start':     None,
                'end':       None,
                'length':    self._guess_length(m.group(2), inferred_type),
                'data_type': inferred_type,
                'required':  True,
                'validation': {},
            }

        return None

    # ── Helpers ──────────────────────────────────────────────────────
    def _clean_name(self, name: str) -> str:
        return re.sub(r'\s+', '_', name.strip().upper().strip('_-'))

    def _resolve_type(self, raw_type: str | None, fallback: str) -> str:
        if not raw_type:
            return fallback
        mapping = {
            'N': 'numeric', 'AN': 'alphanumeric', 'A': 'alpha',
            'NUM': 'numeric', 'ALPHA': 'alpha', 'DT': 'date',
            'AMT': 'amount', 'DATE': 'date', 'BOOL': 'boolean',
        }
        return mapping.get(raw_type.upper(), fallback)

    def _is_required(self, line: str) -> bool:
        lower = line.lower()
        if re.search(r'\b(required|mandatory|must)\b', lower):
            return True
        if re.search(r'\b(optional|conditional|if applicable)\b', lower):
            return False
        return True   # default: required

    def _extract_validation(self, line: str) -> dict:
        v = {}
        # Numeric range: 0-999999
        m = re.search(r'(?:range|value|values?)[\s:]+(\d+)\s*[-–to]+\s*(\d+)', line, re.I)
        if m:
            v['min'] = int(m.group(1))
            v['max'] = int(m.group(2))
        # Allowed values
        m = re.search(r'(?:one of|values?)[\s:]+([A-Z0-9,\s/|]+)', line, re.I)
        if m:
            vals = re.split(r'[,|/\s]+', m.group(1))
            v['allowed'] = [x.strip() for x in vals if x.strip()]
        # Format pattern
        m = re.search(r'format[\s:]+([YYYYMMDD0-9X]+)', line, re.I)
        if m:
            v['format'] = m.group(1)
        return v

    def _guess_length(self, desc: str, dtype: str) -> int:
        defaults = {
            'numeric': 10, 'alphanumeric': 35, 'date': 8,
            'amount': 12, 'routing_number': 9, 'account_number': 17,
            'boolean': 1, 'filler': 1,
        }
        nums = re.findall(r'\d+', desc)
        if nums:
            return int(nums[0])
        return defaults.get(dtype, 20)

    def _deduplicate(self, fields: list[dict]) -> list[dict]:
        seen, unique = set(), []
        for f in fields:
            key = f['name']
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def _infer_missing(self, fields: list[dict]) -> list[dict]:
        """Fill gaps in positional fields."""
        positioned = [f for f in fields if f['start'] is not None]
        if not positioned:
            pos = 1
            for f in fields:
                f['start'] = pos
                f['end']   = pos + f['length'] - 1
                pos        = f['end'] + 1
        return fields


# ══════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE BASE  (stores / retrieves specs)
# ══════════════════════════════════════════════════════════════════════════
class KnowledgeBase:
    """Persistent JSON store for learned spec knowledge."""

    def save(self, spec_name: str, spec: dict) -> str:
        safe = re.sub(r'[^\w\-]', '_', spec_name)
        path = os.path.join(SPEC_STORE, f"{safe}.json")
        spec['_saved_at'] = datetime.utcnow().isoformat()
        with open(path, 'w') as fh:
            json.dump(spec, fh, indent=2)
        return path

    def load(self, spec_name: str) -> dict | None:
        safe = re.sub(r'[^\w\-]', '_', spec_name)
        path = os.path.join(SPEC_STORE, f"{safe}.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    def list_specs(self) -> list[str]:
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(SPEC_STORE)
            if f.endswith('.json')
        ]

    def delete(self, spec_name: str) -> bool:
        safe = re.sub(r'[^\w\-]', '_', spec_name)
        path = os.path.join(SPEC_STORE, f"{safe}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════
# 4. INFERENCE ENGINE  (TF-IDF style term weighting + rule lookup)
# ══════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """Matches an unknown file against known specs using statistical scoring."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def identify_format(self, sample_text: str, top_k: int = 3) -> list[dict]:
        """Return ranked list of matching specs."""
        specs = self.kb.list_specs()
        if not specs:
            return []

        scores = []
        for name in specs:
            spec = self.kb.load(name)
            if spec:
                score = self._score(sample_text, spec)
                scores.append({'spec': name, 'score': score})

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def _score(self, text: str, spec: dict) -> float:
        """Cosine-similarity-like score between text and spec vocabulary."""
        spec_terms = self._spec_terms(spec)
        text_terms = Counter(re.findall(r'\b\w+\b', text.lower()))

        if not spec_terms or not text_terms:
            return 0.0

        total = sum(
            spec_terms.get(t, 0) * text_terms.get(t, 0)
            for t in set(spec_terms) | set(text_terms)
        )
        norm = (
            math.sqrt(sum(v**2 for v in spec_terms.values())) *
            math.sqrt(sum(v**2 for v in text_terms.values()))
        )
        return total / norm if norm else 0.0

    def _spec_terms(self, spec: dict) -> Counter:
        terms = Counter()
        for field in spec.get('fields', []):
            for word in re.findall(r'\w+', field.get('name', '').lower()):
                terms[word] += 2
            terms[field.get('data_type', '')] += 1
        for word in re.findall(r'\w+', spec.get('description', '').lower()):
            terms[word] += 1
        return terms


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN  SpecEngine  (public API)
# ══════════════════════════════════════════════════════════════════════════
class SpecEngine:
    """
    The top-level Custom LLM engine.

    train()          – learn a spec from uploaded document text
    get_spec()       – retrieve a learned spec
    list_specs()     – list all known specs
    identify()       – auto-detect format of unknown file
    """

    def __init__(self):
        self.tokenizer  = FinancialTokenizer()
        self.extractor  = PatternExtractor()
        self.kb         = KnowledgeBase()
        self.inference  = InferenceEngine(self.kb)

    # ── Training ──────────────────────────────────────────────────────
    def train(
        self,
        spec_name: str,
        spec_text: str,
        format_type: str = 'custom',
        description: str = '',
        metadata: dict | None = None,
    ) -> dict:
        """
        Learn a spec from raw spec-document text.
        Returns the learned spec dict.
        """
        logger.info("Training spec: %s", spec_name)

        tokens = self.tokenizer.tokenize(spec_text)
        fields = self.extractor.extract_fields(tokens, format_type)

        # Vocabulary (TF-IDF style, for later identification)
        all_words = []
        for tok in tokens:
            all_words.extend(tok['words'])
        vocab = dict(Counter(all_words).most_common(200))

        # Compute checksum for deduplication
        chk = hashlib.md5(spec_text.encode()).hexdigest()

        spec = {
            'name':        spec_name,
            'format_type': format_type,
            'description': description,
            'fields':      fields,
            'vocabulary':  vocab,
            'checksum':    chk,
            'token_count': len(tokens),
            'field_count': len(fields),
            'metadata':    metadata or {},
        }

        path = self.kb.save(spec_name, spec)
        spec['_saved_path'] = path
        logger.info("Saved spec with %d fields → %s", len(fields), path)
        return spec

    # ── Retrieval ─────────────────────────────────────────────────────
    def get_spec(self, spec_name: str) -> dict | None:
        return self.kb.load(spec_name)

    def list_specs(self) -> list[str]:
        return self.kb.list_specs()

    def delete_spec(self, spec_name: str) -> bool:
        return self.kb.delete(spec_name)

    def identify(self, sample_text: str) -> list[dict]:
        return self.inference.identify_format(sample_text)
