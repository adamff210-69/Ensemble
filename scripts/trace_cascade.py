"""
Cascade Decision Trace: run two ablation configs on the SAME dataset and
report every sample where the final action differs, with the full per-layer
decision path. Answers "where does the extra layer flip a correct decision".

Usage:
    python scripts/trace_cascade.py            # A5 vs A7 (default)
    python scripts/trace_cascade.py A1 A7 25   # custom configs, max 25 flips
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.data.synthetic_generator import SyntheticDataGenerator
from src.layer1_fast.detector import Layer1Detector
from src.layer2_hidden.hidden_extractor import TargetLLMHiddenExtractor
from src.layer2_hidden.prototype_engine import PrototypeEngine
from src.layer2_hidden.mlp_probe import Layer2Analyzer
from src.layer2_hidden.trainer import Layer2Trainer
from src.layer3_verify.verifier import Layer3Verifier
from src.cascade.engine import ThreeLayerCascadeEngine, CascadePrediction
from src.device import resolve_device, load_system_device, device_summary


def path_summary(pred: CascadePrediction) -> str:
    """One-line per-layer decision path for a cascade prediction."""
    parts = []
    if pred.layer1_result:
        r = pred.layer1_result
        parts.append(f"L1 p={r.prob_attack:.3f} [{r.action}]")
    if pred.layer2_result:
        r = pred.layer2_result
        parts.append(f"L2 p={r.prob_attack_hidden:.3f} div={r.divergence_score:.2f} [{r.action}]")
    if pred.layer3_result:
        r = pred.layer3_result
        parts.append(f"L3 {r.verdict} conf={r.confidence:.2f} p={r.prob_attack_est:.3f} [{r.action}]")
    if pred.ensemble_result:
        r = pred.ensemble_result
        parts.append(f"Ensemble score={r.final_score:.3f} [{r.action}]")
    return " -> ".join(parts)


def build_engine(device: torch.device) -> ThreeLayerCascadeEngine:
    """Same engine construction as run_ablations.py (trained L2 probe)."""
    fit_dataset = SyntheticDataGenerator.get_dataset(n_attacks=150, n_benign=150, n_not_inject=50, seed=7)

    l1 = Layer1Detector(device=str(device), use_surrogate=True)
    extractor = TargetLLMHiddenExtractor(device=str(device), use_surrogate=True)
    proto = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model, device=str(device))
    l2_trainer = Layer2Trainer(extractor=extractor, prototype_engine=proto, device=str(device))
    X_fit, y_fit = l2_trainer.fit_prototypes_and_prepare_features(fit_dataset)
    l2_trainer.train_probe(X_fit, y_fit, epochs=15, batch_size=64)
    l2 = Layer2Analyzer(extractor, proto, probe=l2_trainer.probe, device=str(device))
    l3 = Layer3Verifier(device=str(device), use_surrogate=True)
    return ThreeLayerCascadeEngine(l1_detector=l1, l2_analyzer=l2, l3_verifier=l3)


def main():
    code_a = sys.argv[1] if len(sys.argv) > 1 else "A5"
    code_b = sys.argv[2] if len(sys.argv) > 2 else "A7"
    max_flips = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    device = resolve_device(load_system_device())
    print(f"Compute device: {device} ({device_summary()})")
    torch.manual_seed(42)

    dataset = SyntheticDataGenerator.get_dataset(n_attacks=200, n_benign=200, n_not_inject=100, seed=42)
    engine = build_engine(device)

    print(f"\n=== CASCADE DECISION TRACE: {code_a} vs {code_b} over {len(dataset)} samples ===\n")

    flips = []
    for i, s in enumerate(dataset.samples):
        common = dict(
            user_prompt=s.prompt,
            model_response=s.model_response,
            system_prompt=s.system_prompt,
        )
        pred_a = engine.predict(config_name=code_a, **common)
        pred_b = engine.predict(config_name=code_b, **common)
        if pred_a.final_action != pred_b.final_action:
            flips.append((i, s, pred_a, pred_b))

    # Classification of flips
    b_worse = [f for f in flips if is_better(f[2], f[1]) and not is_better(f[3], f[1])]
    a_worse = [f for f in flips if is_better(f[3], f[1]) and not is_better(f[2], f[1])]
    other = [f for f in flips if f not in b_worse and f not in a_worse]

    print(f"Total decision flips: {len(flips)}")
    print(f"  {code_b} worse than {code_a} (A correct, B wrong): {len(b_worse)}")
    print(f"  {code_a} worse than {code_b} (B correct, A wrong): {len(a_worse)}")
    print(f"  both wrong / tie: {len(other)}\n")

    def show(tag, flips_list):
        if not flips_list:
            return
        print(f"--- {tag} (showing up to {max_flips}) ---")
        for i, s, pred_a, pred_b in flips_list[:max_flips]:
            label = "ATTACK" if s.label == 1 else "benign"
            print(f"[{tag}] sample {i} ({label})")
            print(f"  prompt: {s.prompt[:100]}")
            print(f"  {code_a}: {pred_a.final_action}  {path_summary(pred_a)}")
            print(f"  {code_b}: {pred_b.final_action}  {path_summary(pred_b)}")
            print()

    show(f"{code_b}-WORSE", b_worse)
    show(f"{code_a}-WORSE", a_worse)

    if not flips:
        print("No decision flips: both configurations agree on every sample.")


def is_better(pred: CascadePrediction, sample) -> bool:
    """True if the prediction is correct for the sample's label."""
    return (pred.final_action == "BLOCK") == (sample.label == 1)


if __name__ == "__main__":
    main()
