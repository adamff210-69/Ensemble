"""
Layer 2 MLP Probe Classifier (~12.6M parameters) and Result data structures.
Classifies target LLM hidden state representation features.
"""

from dataclasses import dataclass
from typing import Optional, List
import time
import torch
import torch.nn as nn
from .hidden_extractor import TargetLLMHiddenExtractor
from .prototype_engine import PrototypeEngine, PrototypeFeatures
from ..device import resolve_device


@dataclass
class Layer2Result:
    """Detection outcome from Layer 2 hidden state probe."""
    prob_attack_hidden: float
    confidence: float
    divergence_score: float
    latency_ms: float = 0.0
    action: str = "PENDING"  # PASS, BLOCK, ROUTE_TO_L3


class Layer2MLPProbe(nn.Module):
    """
    MLP Classifier for Layer 2 representation probing.
    Architecture: [Input -> 512 -> 256 -> 128 -> Output] with ReLU, BatchNorm, Dropout.
    """

    def __init__(self, input_dim: int = 95, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)


class Layer2Analyzer:
    """End-to-end wrapper combining TargetLLMHiddenExtractor, PrototypeEngine, and Layer2MLPProbe."""

    def __init__(
        self,
        extractor: TargetLLMHiddenExtractor,
        prototype_engine: PrototypeEngine,
        probe: Optional[Layer2MLPProbe] = None,
        device: str = "auto",
    ):
        self.extractor = extractor
        self.prototype_engine = prototype_engine
        self.device = resolve_device(device)

        # Estimate feature dimension: 4 * num_layers - 1
        num_layers = extractor.num_layers
        feature_dim = num_layers * 3 + (num_layers - 1)

        self.probe = probe or Layer2MLPProbe(input_dim=feature_dim).to(self.device)
        self.probe.eval()

    def predict(self, prompt: str, is_attack_hint: bool = False) -> Layer2Result:
        """Run hidden-state extraction, prototype feature computation, and MLP probe classification."""
        start_time = time.perf_counter()

        hidden_output = self.extractor.extract(prompt, is_attack_hint=is_attack_hint)
        features = self.prototype_engine.extract_features(hidden_output.hidden_states)

        feat_tensor = features.feature_vector.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob_attack = self.probe(feat_tensor).item()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        confidence = abs(prob_attack - 0.5) * 2.0

        return Layer2Result(
            prob_attack_hidden=float(prob_attack),
            confidence=float(confidence),
            divergence_score=features.total_divergence,
            latency_ms=latency_ms,
        )
