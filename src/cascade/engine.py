"""
Master Three-Layer Cascade Engine for Prompt Injection Detection.
Orchestrates L1 -> L2 -> L3 inference, confidence handoffs, timing, and ablation configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
from ..layer1_fast.detector import Layer1Detector, Layer1Result
from ..layer2_hidden.mlp_probe import Layer2Analyzer, Layer2Result
from ..layer3_verify.verifier import Layer3Verifier, Layer3Result
from ..layer3_verify.fallback_ensemble import WeightedEnsembleFallback, EnsembleResult
from .policy import HandoffPolicyEngine, HandoffAction


@dataclass
class CascadePrediction:
    """End-to-end response payload from Cascade Engine."""
    final_action: str  # "BLOCK" or "PASS"
    final_probability: float
    decision_layer: str  # "Layer1", "Layer2", "Layer3", or "Ensemble"
    total_latency_ms: float
    layers_evaluated: List[str]
    layer1_result: Optional[Layer1Result] = None
    layer2_result: Optional[Layer2Result] = None
    layer3_result: Optional[Layer3Result] = None
    ensemble_result: Optional[EnsembleResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreeLayerCascadeEngine:
    """Master cascade orchestrator for sequential multi-layer prompt injection detection."""

    def __init__(
        self,
        l1_detector: Layer1Detector,
        l2_analyzer: Layer2Analyzer,
        l3_verifier: Layer3Verifier,
        policy_engine: Optional[HandoffPolicyEngine] = None,
        fallback_ensemble: Optional[WeightedEnsembleFallback] = None,
    ):
        self.l1 = l1_detector
        self.l2 = l2_analyzer
        self.l3 = l3_verifier
        self.policy = policy_engine or HandoffPolicyEngine()
        self.ensemble = fallback_ensemble or WeightedEnsembleFallback()

    def predict(
        self,
        user_prompt: str,
        model_response: str = "Standard response.",
        system_prompt: str = "You are a secure assistant.",
        config_name: str = "A7",  # Ablation mode A1-A7
    ) -> CascadePrediction:
        """
        Execute cascade detection pipeline on user prompt.
        Ablation modes:
        - A1: Layer 1 only
        - A2: Layer 2 only
        - A3: Layer 3 only
        - A4: Layer 1 + Layer 2 (Parallel Ensemble)
        - A5: Layer 1 + Layer 3 (Cascade)
        - A6: Layer 2 + Layer 3 (Cascade)
        - A7: Full 3-layer cascade (Proposed System)
        """
        start_time = time.perf_counter()
        layers_evaluated = []

        # Default initialization
        res_l1: Optional[Layer1Result] = None
        res_l2: Optional[Layer2Result] = None
        res_l3: Optional[Layer3Result] = None
        res_ens: Optional[EnsembleResult] = None

        # Execute based on ablation setup
        if config_name == "A1":
            layers_evaluated.append("Layer1")
            res_l1 = self.l1.predict(user_prompt)
            action = "BLOCK" if res_l1.prob_attack >= 0.50 else "PASS"
            final_prob = res_l1.prob_attack
            decision_layer = "Layer1"

        elif config_name == "A2":
            layers_evaluated.append("Layer2")
            res_l2 = self.l2.predict(user_prompt)
            action = "BLOCK" if res_l2.prob_attack_hidden >= 0.50 else "PASS"
            final_prob = res_l2.prob_attack_hidden
            decision_layer = "Layer2"

        elif config_name == "A3":
            layers_evaluated.append("Layer3")
            res_l3 = self.l3.verify(user_prompt, model_response, system_prompt=system_prompt)
            action = "BLOCK" if res_l3.verdict == "UNSAFE" else "PASS"
            final_prob = res_l3.prob_attack_est
            decision_layer = "Layer3"

        elif config_name == "A4":
            # Layer 1 + Layer 2 Parallel Ensemble
            layers_evaluated.extend(["Layer1", "Layer2"])
            res_l1 = self.l1.predict(user_prompt)
            res_l2 = self.l2.predict(user_prompt)
            prob_combined = 0.4 * res_l1.prob_attack + 0.6 * res_l2.prob_attack_hidden
            action = "BLOCK" if prob_combined >= 0.50 else "PASS"
            final_prob = prob_combined
            decision_layer = "ParallelEnsemble(L1+L2)"

        elif config_name == "A5":
            # Layer 1 + Layer 3 Cascade
            layers_evaluated.append("Layer1")
            res_l1 = self.l1.predict(user_prompt)
            l1_action = self.policy.evaluate_layer1(res_l1.prob_attack)
            if l1_action == HandoffAction.BLOCK:
                action, final_prob, decision_layer = "BLOCK", res_l1.prob_attack, "Layer1"
            elif l1_action == HandoffAction.PASS:
                action, final_prob, decision_layer = "PASS", res_l1.prob_attack, "Layer1"
            else:
                layers_evaluated.append("Layer3")
                res_l3 = self.l3.verify(user_prompt, model_response, system_prompt=system_prompt, prob_l1=res_l1.prob_attack)
                action = "BLOCK" if res_l3.verdict == "UNSAFE" else "PASS"
                final_prob = res_l3.prob_attack_est
                decision_layer = "Layer3"

        elif config_name == "A6":
            # Layer 2 + Layer 3 Cascade
            layers_evaluated.append("Layer2")
            res_l2 = self.l2.predict(user_prompt)
            l2_action = self.policy.evaluate_layer2(res_l2.prob_attack_hidden)
            if l2_action == HandoffAction.BLOCK:
                action, final_prob, decision_layer = "BLOCK", res_l2.prob_attack_hidden, "Layer2"
            elif l2_action == HandoffAction.PASS:
                action, final_prob, decision_layer = "PASS", res_l2.prob_attack_hidden, "Layer2"
            else:
                layers_evaluated.append("Layer3")
                res_l3 = self.l3.verify(user_prompt, model_response, system_prompt=system_prompt, prob_l2=res_l2.prob_attack_hidden)
                action = "BLOCK" if res_l3.verdict == "UNSAFE" else "PASS"
                final_prob = res_l3.prob_attack_est
                decision_layer = "Layer3"

        else:
            # A7: Full Proposed 3-Layer Sequential Cascade
            # Step 1: Layer 1
            layers_evaluated.append("Layer1")
            res_l1 = self.l1.predict(user_prompt)
            l1_action = self.policy.evaluate_layer1(res_l1.prob_attack)
            res_l1.action = l1_action.value

            if l1_action == HandoffAction.BLOCK:
                action = "BLOCK"
                final_prob = res_l1.prob_attack
                decision_layer = "Layer1"
            elif l1_action == HandoffAction.PASS:
                action = "PASS"
                final_prob = res_l1.prob_attack
                decision_layer = "Layer1"
            else:
                # Step 2: Route to Layer 2 (independent of L1: its
                # representation is derived from the prompt content, not
                # from L1's verdict)
                layers_evaluated.append("Layer2")
                res_l2 = self.l2.predict(user_prompt)
                l2_action = self.policy.evaluate_layer2(res_l2.prob_attack_hidden)
                res_l2.action = l2_action.value

                if l2_action == HandoffAction.BLOCK:
                    action = "BLOCK"
                    final_prob = res_l2.prob_attack_hidden
                    decision_layer = "Layer2"
                elif l2_action == HandoffAction.PASS:
                    action = "PASS"
                    final_prob = res_l2.prob_attack_hidden
                    decision_layer = "Layer2"
                else:
                    # Step 3: Route to Layer 3
                    layers_evaluated.append("Layer3")
                    res_l3 = self.l3.verify(
                        user_prompt=user_prompt,
                        model_response=model_response,
                        system_prompt=system_prompt,
                        prob_l1=res_l1.prob_attack,
                        conf_l1=res_l1.confidence,
                        prob_l2=res_l2.prob_attack_hidden,
                        conf_l2=res_l2.confidence,
                    )
                    l3_action = self.policy.evaluate_layer3(res_l3.verdict, res_l3.confidence)
                    res_l3.action = l3_action.value

                    if l3_action == HandoffAction.BLOCK:
                        action = "BLOCK"
                        final_prob = res_l3.prob_attack_est
                        decision_layer = "Layer3"
                    elif l3_action == HandoffAction.PASS:
                        action = "PASS"
                        final_prob = res_l3.prob_attack_est
                        decision_layer = "Layer3"
                    else:
                        # Step 4: Fallback Weighted Ensemble
                        layers_evaluated.append("Ensemble")
                        res_ens = self.ensemble.compute_fallback(
                            prob_l1=res_l1.prob_attack,
                            prob_l2=res_l2.prob_attack_hidden,
                            prob_l3=res_l3.prob_attack_est,
                        )
                        action = res_ens.action
                        final_prob = res_ens.final_score
                        decision_layer = "Ensemble"

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return CascadePrediction(
            final_action=action,
            final_probability=float(final_prob),
            decision_layer=decision_layer,
            total_latency_ms=total_latency_ms,
            layers_evaluated=layers_evaluated,
            layer1_result=res_l1,
            layer2_result=res_l2,
            layer3_result=res_l3,
            ensemble_result=res_ens,
            metadata={"config_name": config_name},
        )
