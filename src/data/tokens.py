"""
Deterministic word -> index tokenization shared by the Layer-1 surrogate and
its trainer.

Python's built-in hash() is salted per process (PYTHONHASHSEED), so
abs(hash(w)) yields a different vocab index in every run and any trained
weights are not reproducible or transferable. CRC32 is stable across
processes, platforms, and Python versions.
"""

import zlib


def stable_token_index(word: str, vocab_size: int = 10000) -> int:
    """Map a word to a stable vocabulary index in [0, vocab_size)."""
    return zlib.crc32(word.lower().encode("utf-8")) % vocab_size
