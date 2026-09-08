"""
Metrics Calculator for Prompt Injection Detection Benchmarks.
Calculates Accuracy, Precision, Recall, F1, AUROC, FPR@Thresholds,
Latency Percentiles (p50, p95, p99), QPS Throughput, and NotInject Over-Defense Rate.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from ..cascade.engine import CascadePrediction
from ..data.dataset_loader import InjectionDataset


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics summary."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    fpr: float  # False Positive Rate overall
    over_defense_rate: float  # FPR specifically on NotInject dataset
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_qps: float
    total_samples: int
    raw_latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auroc": round(self.auroc, 4),
            "fpr": round(self.fpr, 4),
            "over_defense_rate": round(self.over_defense_rate, 4),
            "latency_p50_ms": round(self.latency_p50, 2),
            "latency_p95_ms": round(self.latency_p95, 2),
            "latency_p99_ms": round(self.latency_p99, 2),
            "throughput_qps": round(self.throughput_qps, 2),
            "total_samples": self.total_samples,
        }


class Evaluator:
    """Computes benchmark performance metrics over dataset predictions."""

    @staticmethod
    def evaluate_predictions(
        predictions: List[CascadePrediction],
        dataset: InjectionDataset,
    ) -> PerformanceMetrics:
        """Compute full suite of classification, latency, and over-defense metrics."""
        targets = dataset.get_labels()
        binary_preds = [1 if p.final_action == "BLOCK" else 0 for p in predictions]
        probs = [p.final_probability for p in predictions]
        latencies = [p.total_latency_ms for p in predictions]

        # Classification metrics
        acc = float(accuracy_score(targets, binary_preds))
        prec = float(precision_score(targets, binary_preds, zero_division=0))
        rec = float(recall_score(targets, binary_preds, zero_division=0))
        f1 = float(f1_score(targets, binary_preds, zero_division=0))
        auc = float(roc_auc_score(targets, probs)) if len(set(targets)) > 1 else 0.5

        # False Positive Rate = FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Over-defense rate (FPR specifically on NotInject dataset samples)
        not_inject_indices = [
            i for i, s in enumerate(dataset.samples) if s.source_dataset == "not_inject_339"
        ]
        if not_inject_indices:
            ni_targets = [targets[i] for i in not_inject_indices]
            ni_preds = [binary_preds[i] for i in not_inject_indices]
            ni_fps = sum(1 for t, p in zip(ni_targets, ni_preds) if t == 0 and p == 1)
            ni_tns = sum(1 for t, p in zip(ni_targets, ni_preds) if t == 0 and p == 0)
            over_defense_rate = float(ni_fps / (ni_fps + ni_tns)) if (ni_fps + ni_tns) > 0 else 0.0
        else:
            over_defense_rate = fpr

        # Latency Percentiles (p50, p95, p99)
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

        # Throughput QPS = 1000 / avg_latency_ms
        avg_latency = float(np.mean(latencies)) if latencies else 1.0
        qps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return PerformanceMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            auroc=auc,
            fpr=fpr,
            over_defense_rate=over_defense_rate,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput_qps=qps,
            total_samples=len(targets),
            raw_latencies=latencies,
        )
