"""
Script to execute full Ablation Matrix (A1-A7) and generate structured Research Report.

Methodology notes (integrity of the study):
- The Layer-2 probe and prototypes are fit on a DEDICATED dataset (seed=7)
  that never overlaps the evaluation dataset (seed=42).
- Feature extraction is label-free (the surrogate derives its representation
  from prompt content), so the probe cannot memorize ground truth.
- The McNemar significance test uses the SAME per-sample predictions that
  produced the table (stored by the runner) — no stochastic re-runs.
- Global torch seed fixed for reproducible surrogate noise.
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
from src.cascade.engine import ThreeLayerCascadeEngine
from src.evaluation.ablations import AblationStudyRunner
from src.evaluation.statistical import StatisticalSignificanceTest
from src.device import resolve_device, load_system_device, device_summary


def main():
    print("==========================================================================")
    print("      RESEARCH ABLATION STUDY: THREE-LAYER CASCADE ENSEMBLE SYSTEM")
    print("==========================================================================")

    device = resolve_device(load_system_device())
    print(f"Compute device: {device} ({device_summary()})")
    torch.manual_seed(42)  # reproducible surrogate trajectory noise

    # Evaluation dataset — nothing below is fit on these samples
    dataset = SyntheticDataGenerator.get_dataset(
        n_attacks=200, n_benign=200, n_not_inject=100, seed=42
    )
    print(f"Evaluation Dataset: {len(dataset)} samples (200 attacks, 200 benign, 100 NotInject), seed=42")

    # Dedicated fit dataset for L2 prototypes + probe (disjoint from eval)
    fit_dataset = SyntheticDataGenerator.get_dataset(
        n_attacks=150, n_benign=150, n_not_inject=50, seed=7
    )
    print(f"Fit Dataset (L2 prototypes + probe): {len(fit_dataset)} samples, seed=7 (disjoint from eval)")

    # Initialize Engine Components (all share the same compute device)
    l1 = Layer1Detector(device=str(device), use_surrogate=True)
    extractor = TargetLLMHiddenExtractor(device=str(device), use_surrogate=True)
    proto = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model, device=str(device))

    # Fit prototypes and train the probe on the fit set only
    l2_trainer = Layer2Trainer(extractor=extractor, prototype_engine=proto, device=str(device))
    X_fit, y_fit = l2_trainer.fit_prototypes_and_prepare_features(fit_dataset)
    l2_trainer.train_probe(X_fit, y_fit, epochs=15, batch_size=64)
    fit_metrics = l2_trainer.evaluate(X_fit, y_fit)
    print(f"Layer-2 probe fit-set quality (diagnostic only, NOT generalization): {fit_metrics}")

    l2 = Layer2Analyzer(extractor, proto, probe=l2_trainer.probe, device=str(device))
    l3 = Layer3Verifier(device=str(device), use_surrogate=True)
    engine = ThreeLayerCascadeEngine(l1_detector=l1, l2_analyzer=l2, l3_verifier=l3)

    # Run Ablation Matrix
    runner = AblationStudyRunner(engine)
    results = runner.run_ablations(dataset)

    # Print Table
    table_md = runner.generate_report_table(results)
    print("\n" + table_md + "\n")
    print("Note: A1 baseline = untrained regex-boosted surrogate (its signal comes")
    print("from trigger patterns, not learned weights). L3 alone (A3) can only see")
    print("the response text and default 0.5/0.5 layer scores.")

    # Statistical Significance Testing (A7 vs A1) — using the SAME predictions
    # as the table above (no stochastic re-run)
    print("\n--- Statistical Significance (A7 Proposed vs A1 Baseline) ---")
    preds_a1 = runner.get_predictions("A1")
    preds_a7 = runner.get_predictions("A7")
    targets = dataset.get_labels()

    mcnemar = StatisticalSignificanceTest.mcnemar_test(preds_a7, preds_a1, targets)
    discordant = mcnemar["b01_a_correct_b_wrong"] + mcnemar["b10_a_wrong_b_correct"]
    print(f"Discordant pairs: {discordant} (A7-only correct: {mcnemar['b01_a_correct_b_wrong']}, A1-only correct: {mcnemar['b10_a_wrong_b_correct']})")
    print(f"McNemar Statistic: {mcnemar['statistic']:.4f}")
    print(f"McNemar p-value: {mcnemar['p_value']:.4e}")
    print(f"Statistically Significant (p < 0.05): {mcnemar['is_significant_p05']}")

    print("\n==========================================================================")
    print("                    ABLATION STUDY COMPLETED CLEANLY")
    print("==========================================================================")


if __name__ == "__main__":
    main()
