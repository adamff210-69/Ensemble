"""
Layer 2: Hidden-State Representation Analysis (<100ms Latency Budget)
Extracts last-token hidden state trajectories from target LLMs, computes layer-wise class prototype distances,
and classifies representation shifts using a lightweight MLP probe.
"""

from .hidden_extractor import TargetLLMHiddenExtractor, HiddenStateOutput
from .prototype_engine import PrototypeEngine, PrototypeFeatures
from .mlp_probe import Layer2MLPProbe, Layer2Result
from .trainer import Layer2Trainer

__all__ = [
    "TargetLLMHiddenExtractor",
    "HiddenStateOutput",
    "PrototypeEngine",
    "PrototypeFeatures",
    "Layer2MLPProbe",
    "Layer2Result",
    "Layer2Trainer",
]
