"""
Handoff Policy Engine executing confidence thresholding across Layer 1, Layer 2, and Layer 3.
"""

from enum import Enum
from typing import Tuple


class HandoffAction(str, Enum):
    BLOCK = "BLOCK"
    PASS = "PASS"
    ROUTE_TO_L2 = "ROUTE_TO_L2"
    ROUTE_TO_L3 = "ROUTE_TO_L3"
    FALLBACK_ENSEMBLE = "FALLBACK_ENSEMBLE"


class HandoffPolicyEngine:
    """Evaluates multi-tier handoff policies between cascade layers."""

    def __init__(
        self,
        l1_attack_thresh: float = 0.85,
        l1_benign_thresh: float = 0.15,
        l2_attack_thresh: float = 0.80,
        l2_benign_thresh: float = 0.20,
        l3_confidence_thresh: float = 0.70,
    ):
        self.l1_attack_thresh = l1_attack_thresh
        self.l1_benign_thresh = l1_benign_thresh
        self.l2_attack_thresh = l2_attack_thresh
        self.l2_benign_thresh = l2_benign_thresh
        self.l3_confidence_thresh = l3_confidence_thresh

    def evaluate_layer1(self, prob_attack: float) -> HandoffAction:
        """
        Layer 1 Handoff Policy:
        if prob_attack > 0.85 -> BLOCK
        elif prob_attack < 0.15 -> PASS
        else -> ROUTE_TO_L2
        """
        if prob_attack > self.l1_attack_thresh:
            return HandoffAction.BLOCK
        elif prob_attack < self.l1_benign_thresh:
            return HandoffAction.PASS
        else:
            return HandoffAction.ROUTE_TO_L2

    def evaluate_layer2(self, prob_attack_hidden: float) -> HandoffAction:
        """
        Layer 2 Handoff Policy:
        if prob_attack_hidden > 0.80 -> BLOCK
        elif prob_attack_hidden < 0.20 -> PASS
        else -> ROUTE_TO_L3
        """
        if prob_attack_hidden > self.l2_attack_thresh:
            return HandoffAction.BLOCK
        elif prob_attack_hidden < self.l2_benign_thresh:
            return HandoffAction.PASS
        else:
            return HandoffAction.ROUTE_TO_L3

    def evaluate_layer3(self, verdict: str, confidence: float) -> HandoffAction:
        """
        Layer 3 Decision Policy:
        if verdict == "UNSAFE" and confidence > 0.7 -> BLOCK
        elif verdict == "SAFE" and confidence > 0.7 -> PASS
        else -> FALLBACK_ENSEMBLE
        """
        if verdict == "UNSAFE" and confidence > self.l3_confidence_thresh:
            return HandoffAction.BLOCK
        elif verdict == "SAFE" and confidence > self.l3_confidence_thresh:
            return HandoffAction.PASS
        else:
            return HandoffAction.FALLBACK_ENSEMBLE
