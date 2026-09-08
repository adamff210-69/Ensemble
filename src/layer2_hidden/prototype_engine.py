"""
Prototype Engine for Layer 2 Hidden-State Representation Analysis.
Computes layer-wise class prototypes (mu_attack_l, mu_benign_l), distance vectors,
divergence scores, and trajectory metrics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch

from ..device import resolve_device


@dataclass
class PrototypeFeatures:
    """Extracted prototype distance and trajectory feature vectors."""
    attack_distances: torch.Tensor   # shape: [num_layers]
    benign_distances: torch.Tensor   # shape: [num_layers]
    divergence_scores: torch.Tensor  # shape: [num_layers]
    total_divergence: float
    trajectory_deltas: torch.Tensor # shape: [num_layers - 1]
    feature_vector: torch.Tensor     # Concatenated feature representation tensor for MLP probe


class PrototypeEngine:
    """Computes attack and benign class prototypes and calculates layer-wise representation metrics."""

    def __init__(self, num_layers: int = 24, d_model: int = 2560, device: str = "auto"):
        self.num_layers = num_layers
        self.d_model = d_model
        self.device = resolve_device(device)
        # Prototypes initialized to zeros until compute_prototypes is invoked
        self.mu_attack = torch.zeros(num_layers, d_model, device=self.device)
        self.mu_benign = torch.zeros(num_layers, d_model, device=self.device)
        self.is_fitted = False

    def compute_prototypes(
        self,
        attack_states: List[torch.Tensor],  # list of [num_layers, d_model] tensors
        benign_states: List[torch.Tensor],  # list of [num_layers, d_model] tensors
    ) -> None:
        """Compute mean hidden states for attack and benign classes across all layers."""
        if attack_states:
            stacked_attack = torch.stack(attack_states, dim=0)  # [N_attack, num_layers, d_model]
            self.mu_attack = stacked_attack.mean(dim=0).to(self.device)

        if benign_states:
            stacked_benign = torch.stack(benign_states, dim=0)  # [N_benign, num_layers, d_model]
            self.mu_benign = stacked_benign.mean(dim=0).to(self.device)

        # Default fallback non-zero initialization if dataset was empty
        if not attack_states or not benign_states:
            self.mu_attack = torch.ones(self.num_layers, self.d_model, device=self.device) * 0.5
            self.mu_benign = torch.ones(self.num_layers, self.d_model, device=self.device) * -0.5

        self.is_fitted = True

    def extract_features(self, hidden_matrix: torch.Tensor) -> PrototypeFeatures:
        """
        Compute layer-wise prototype distances and trajectory features.
        hidden_matrix shape: [num_layers, d_model]
        """
        num_layers, d_model = hidden_matrix.shape

        # Ensure prototypes live on the same device as the incoming hidden states
        mu_attack = self.mu_attack.to(hidden_matrix.device)
        mu_benign = self.mu_benign.to(hidden_matrix.device)

        # 1. Per-layer attack prototype distance: ||h_l - mu_attack_l||2
        attack_dists = torch.norm(hidden_matrix - mu_attack[:num_layers, :d_model], p=2, dim=1)

        # 2. Per-layer benign prototype distance: ||h_l - mu_benign_l||2
        benign_dists = torch.norm(hidden_matrix - mu_benign[:num_layers, :d_model], p=2, dim=1)

        # 3. Layer-wise divergence score: (distance_to_attack - distance_to_benign)
        divergence = attack_dists - benign_dists
        total_divergence = divergence.sum().item()

        # 4. Trajectory features: rate of change across layers ||h_{l+1} - h_l||2
        trajectory_deltas = torch.norm(hidden_matrix[1:] - hidden_matrix[:-1], p=2, dim=1)

        # Concatenate into unified feature vector for MLP probe:
        # [attack_dists, benign_dists, divergence, trajectory_deltas]
        feature_vector = torch.cat([attack_dists, benign_dists, divergence, trajectory_deltas], dim=0)

        return PrototypeFeatures(
            attack_distances=attack_dists,
            benign_distances=benign_dists,
            divergence_scores=divergence,
            total_divergence=total_divergence,
            trajectory_deltas=trajectory_deltas,
            feature_vector=feature_vector,
        )
