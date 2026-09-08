"""
Script to execute full Ablation Matrix (A1-A7) and generate structured Research Report.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.layer1_fast.detector import Layer1Detector
from src.layer2_hidden.hidden_extractor import TargetLLMHiddenExtractor
from src.layer2_hidden.prototype_engine import PrototypeEngine
from src.layer2_hidden.mlp_probe import Layer2Analyzer
from src.layer3_verify.verifier import Layer3Verifier
from src.cascade.engine import ThreeLayerCascadeEngine
from src.evaluation.ablations import AblationStudyRunner
from src.evaluation.statistical import StatisticalSignificanceTest


def main():
    print("==========================================================================")
    print("      RESEARCH ABLATION STUDY: THREE-LAYER CASCADE ENSEMBLE SYSTEM")
    print("==========================================================================")

    dataset = SyntheticDataGenerator.get_dataset(
        n_attacks=200, n_benign=200, n_not_inject=100, seed=42
    )
    print(f"Evaluation Dataset Loaded: {len(dataset)} samples (200 attacks, 200 benign, 100 NotInject).")

    # Initialize Engine Components
    l1 = Layer1Detector(use_surrogate=True)
    extractor = TargetLLMHiddenExtractor(use_surrogate=True)
    proto = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model)

    # Initialize prototypes with synthetic states
    attack_samples = [s for s in dataset.samples if s.label == 1][:30]
    benign_samples = [s for s in dataset.samples if s.label == 0][:30]
    proto.compute_prototypes(
        [extractor.extract(s.prompt, is_attack_hint=True).hidden_states for s in attack_samples],
        [extractor.extract(s.prompt, is_attack_hint=False).hidden_states for s in benign_samples],
    )

    l2 = Layer2Analyzer(extractor, proto)
    l3 = Layer3Verifier(use_surrogate=True)
    engine = ThreeLayerCascadeEngine(l1_detector=l1, l2_analyzer=l2, l3_verifier=l3)

    # Run Ablation Matrix
    runner = AblationStudyRunner(engine)
    results = runner.run_ablations(dataset)

    # Print Table
    table_md = runner.generate_report_table(results)
    print("\n" + table_md + "\n")

    # Statistical Significance Testing (A7 vs A1 Baseline)
    print("\n--- Statistical Significance (A7 Proposed vs A1 Baseline) ---")
    preds_a1 = [
        1 if engine.predict(s.prompt, config_name="A1").final_action == "BLOCK" else 0
        for s in dataset.samples
    ]
    preds_a7 = [
        1 if engine.predict(s.prompt, config_name="A7").final_action == "BLOCK" else 0
        for s in dataset.samples
    ]
    targets = dataset.get_labels()

    mcnemar = StatisticalSignificanceTest.mcnemar_test(preds_a7, preds_a1, targets)
    print(f"McNemar Statistic: {mcnemar['statistic']:.4f}")
    print(f"McNemar p-value: {mcnemar['p_value']:.4e}")
    print(f"Statistically Significant (p < 0.05): {mcnemar['is_significant_p05']}")

    print("\n==========================================================================")
    print("                    ABLATION STUDY COMPLETED CLEANLY")
    print("==========================================================================")


if __name__ == "__main__":
    main()
