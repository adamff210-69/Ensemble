"""
Ablation Study Runner for evaluating Configurations A1 through A7.
Generates comprehensive comparative performance tables and statistical significance analysis.
"""

from typing import Dict, List, Any
import pandas as pd
from ..cascade.engine import ThreeLayerCascadeEngine, CascadePrediction
from ..data.dataset_loader import InjectionDataset
from .metrics import Evaluator, PerformanceMetrics
from .statistical import StatisticalSignificanceTest


class AblationStudyRunner:
    """Runner executing full ablation matrix A1-A7 over benchmark evaluation datasets."""

    CONFIGS = {
        "A1": "Layer 1 Only (Fast Pre-filter)",
        "A2": "Layer 2 Only (Hidden-State Analysis)",
        "A3": "Layer 3 Only (Output Verification)",
        "A4": "Layer 1 + Layer 2 (Parallel Ensemble)",
        "A5": "Layer 1 + Layer 3 (Cascade)",
        "A6": "Layer 2 + Layer 3 (Cascade)",
        "A7": "Full 3-Layer Cascade (Proposed System)",
    }

    def __init__(self, engine: ThreeLayerCascadeEngine):
        self.engine = engine

    def run_ablations(self, dataset: InjectionDataset) -> Dict[str, PerformanceMetrics]:
        """Run evaluation over all ablation configurations A1 to A7."""
        results: Dict[str, PerformanceMetrics] = {}

        for code in sorted(self.CONFIGS.keys()):
            predictions: List[CascadePrediction] = []
            for sample in dataset.samples:
                pred = self.engine.predict(
                    user_prompt=sample.prompt,
                    model_response=sample.model_response,
                    system_prompt=sample.system_prompt,
                    config_name=code,
                )
                predictions.append(pred)

            metrics = Evaluator.evaluate_predictions(predictions, dataset)
            results[code] = metrics

        return results

    def generate_report_table(self, results: Dict[str, PerformanceMetrics]) -> str:
        """Generate formatted Markdown research summary table of ablation results."""
        lines = [
            "| Exp Code | Configuration Description | Accuracy | F1 Score | AUROC | FPR | Over-Defense Rate | Latency (p95) | QPS |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for code, description in self.CONFIGS.items():
            if code in results:
                m = results[code]
                lines.append(
                    f"| **{code}** | {description} | {m.accuracy:.4f} | {m.f1:.4f} | {m.auroc:.4f} | "
                    f"{m.fpr:.4f} | {m.over_defense_rate:.4f} | {m.latency_p95:.2f}ms | {m.throughput_qps:.1f} |"
                )

        return "\n".join(lines)
