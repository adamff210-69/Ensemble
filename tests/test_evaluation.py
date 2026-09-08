"""
Unit tests for Metrics, Statistical Significance, and Ablation Study.
"""

import pytest
from src.data.synthetic_generator import SyntheticDataGenerator
from src.layer1_fast.detector import Layer1Detector
from src.layer2_hidden.hidden_extractor import TargetLLMHiddenExtractor
from src.layer2_hidden.prototype_engine import PrototypeEngine
from src.layer2_hidden.mlp_probe import Layer2Analyzer
from src.layer3_verify.verifier import Layer3Verifier
from src.cascade.engine import ThreeLayerCascadeEngine
from src.evaluation.metrics import Evaluator, PerformanceMetrics
from src.evaluation.statistical import StatisticalSignificanceTest
from src.evaluation.ablations import AblationStudyRunner


def test_evaluator_metrics():
    dataset = SyntheticDataGenerator.get_dataset(n_attacks=10, n_benign=10, n_not_inject=5)
    l1 = Layer1Detector(use_surrogate=True)
    extractor = TargetLLMHiddenExtractor(use_surrogate=True)
    proto = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model)
    l2 = Layer2Analyzer(extractor, proto)
    l3 = Layer3Verifier(use_surrogate=True)
    engine = ThreeLayerCascadeEngine(l1, l2, l3)

    preds = [engine.predict(s.prompt) for s in dataset.samples]
    metrics = Evaluator.evaluate_predictions(preds, dataset)

    assert isinstance(metrics, PerformanceMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert metrics.latency_p50 > 0.0
    assert metrics.throughput_qps > 0.0


def test_mcnemar_test():
    preds_a = [1, 0, 1, 1, 0]
    preds_b = [1, 1, 1, 0, 0]
    targets = [1, 0, 1, 1, 0]

    res = StatisticalSignificanceTest.mcnemar_test(preds_a, preds_b, targets)
    assert "statistic" in res
    assert "p_value" in res


def test_ablation_runner_stores_predictions():
    """Significance tests must reuse the table's own per-sample predictions,
    not re-run the (stochastic) pipeline."""
    dataset = SyntheticDataGenerator.get_dataset(n_attacks=10, n_benign=10, n_not_inject=5)
    l1 = Layer1Detector(use_surrogate=True)
    extractor = TargetLLMHiddenExtractor(use_surrogate=True)
    proto = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model)
    l2 = Layer2Analyzer(extractor, proto)
    l3 = Layer3Verifier(use_surrogate=True)
    engine = ThreeLayerCascadeEngine(l1, l2, l3)

    runner = AblationStudyRunner(engine)
    results = runner.run_ablations(dataset)

    assert set(results.keys()) == set(runner.CONFIGS.keys())
    for code in runner.CONFIGS:
        stored = runner.get_predictions(code)
        assert len(stored) == len(dataset.samples)
        assert set(stored) <= {0, 1}
