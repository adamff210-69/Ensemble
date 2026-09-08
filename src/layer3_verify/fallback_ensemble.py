"""
Weighted Ensemble Fallback Module.
Executes multi-layer logit fusion when Layer 3 yields UNCERTAIN or low-confidence verdict.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class EnsembleResult:
    """Outcome of weighted ensemble fallback computation."""
    final_score: float
    decision_threshold: float
    action: str  # BLOCK or PASS
    layer_weights: List[float]


class WeightedEnsembleFallback:
    """Calculates weighted linear combination of probabilities across all three layers."""

    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.30, 0.40, 0.30),
        decision_threshold: float = 0.50,
    ):
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.decision_threshold = decision_threshold

    def compute_fallback(
        self, prob_l1: float, prob_l2: float, prob_l3: float
    ) -> EnsembleResult:
        """Compute final score: w1*prob_l1 + w2*prob_l2 + w3*prob_l3."""
        w1, w2, w3 = self.weights
        final_score = w1 * prob_l1 + w2 * prob_l2 + w3 * prob_l3
        action = "BLOCK" if final_score >= self.decision_threshold else "PASS"

        return EnsembleResult(
            final_score=float(final_score),
            decision_threshold=self.decision_threshold,
            action=action,
            layer_weights=self.weights,
        )
