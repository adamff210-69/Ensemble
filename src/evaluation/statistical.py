"""
Statistical Significance Testing Module.
Provides McNemar's Test for model comparisons and Bootstrap 95% Confidence Intervals.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import stats


class StatisticalSignificanceTest:
    """Statistical methods to validate performance improvements and statistical significance."""

    @staticmethod
    def mcnemar_test(
        preds_a: List[int], preds_b: List[int], targets: List[int]
    ) -> Dict[str, float]:
        """
        Execute McNemar's Test on paired model binary predictions.
        Contingency Table:
                     Model B Correct  Model B Incorrect
        Model A Correct      b00              b01
        Model A Incorrect    b10              b11
        """
        correct_a = [1 if p == t else 0 for p, t in zip(preds_a, targets)]
        correct_b = [1 if p == t else 0 for p, t in zip(preds_b, targets)]

        b01 = sum(1 for ca, cb in zip(correct_a, correct_b) if ca == 1 and cb == 0)
        b10 = sum(1 for ca, cb in zip(correct_a, correct_b) if ca == 0 and cb == 1)

        # Continuity corrected McNemar statistic: (|b01 - b10| - 1)^2 / (b01 + b10)
        if (b01 + b10) == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(b01 - b10) - 1.0) ** 2 / (b01 + b10)
            p_value = float(stats.chi2.sf(statistic, df=1))

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "b01_a_correct_b_wrong": b01,
            "b10_a_wrong_b_correct": b10,
            "is_significant_p05": p_value < 0.05,
        }

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        metric_func=np.mean,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Compute non-parametric bootstrap percentile confidence interval.
        Returns (mean, lower_bound, upper_bound).
        """
        np.random.seed(seed)
        n = len(data)
        if n == 0:
            return 0.0, 0.0, 0.0

        boot_metrics = []
        data_arr = np.array(data)

        for _ in range(n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            sample = data_arr[indices]
            boot_metrics.append(metric_func(sample))

        alpha = (1.0 - ci) / 2.0
        lower = float(np.percentile(boot_metrics, alpha * 100))
        upper = float(np.percentile(boot_metrics, (1.0 - alpha) * 100))
        mean_val = float(np.mean(boot_metrics))

        return mean_val, lower, upper
