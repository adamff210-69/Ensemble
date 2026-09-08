"""
Layer 3: Output-Verification Lightweight LLM (<150ms Latency Budget)
Performs contextual safety verification over (Prompt, Model Output, L1 Diagnostics, L2 Diagnostics)
and executes weighted ensemble fallback for uncertain cases.
"""

from .verifier import Layer3Verifier, Layer3Result
from .fallback_ensemble import WeightedEnsembleFallback, EnsembleResult

__all__ = [
    "Layer3Verifier",
    "Layer3Result",
    "WeightedEnsembleFallback",
    "EnsembleResult",
]
