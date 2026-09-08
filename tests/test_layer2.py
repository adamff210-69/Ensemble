"""
Unit tests for Layer 2 Hidden-State Analysis and MLP Probe.
"""

import pytest
import torch
from src.layer2_hidden.hidden_extractor import TargetLLMHiddenExtractor
from src.layer2_hidden.prototype_engine import PrototypeEngine
from src.layer2_hidden.mlp_probe import Layer2MLPProbe, Layer2Analyzer
from src.layer2_hidden.trainer import Layer2Trainer
from src.data.synthetic_generator import SyntheticDataGenerator


def test_hidden_extractor():
    extractor = TargetLLMHiddenExtractor(num_layers=12, d_model=128, use_surrogate=True)
    out = extractor.extract("Test prompt", is_attack_hint=True)

    assert out.hidden_states.shape == (12, 128)
    assert out.num_layers == 12
    assert out.d_model == 128
    assert out.extraction_time_ms > 0.0


def test_prototype_engine():
    proto = PrototypeEngine(num_layers=12, d_model=128)
    attack_states = [torch.randn(12, 128) for _ in range(5)]
    benign_states = [torch.randn(12, 128) for _ in range(5)]

    proto.compute_prototypes(attack_states, benign_states)
    assert proto.is_fitted

    test_matrix = torch.randn(12, 128)
    feats = proto.extract_features(test_matrix)

    assert feats.attack_distances.shape[0] == 12
    assert feats.benign_distances.shape[0] == 12
    assert feats.feature_vector.shape[0] == (12 * 3 + 11)


def test_layer2_analyzer():
    extractor = TargetLLMHiddenExtractor(num_layers=12, d_model=128, use_surrogate=True)
    proto = PrototypeEngine(num_layers=12, d_model=128)
    analyzer = Layer2Analyzer(extractor, proto)

    res = analyzer.predict("System override attack prompt", is_attack_hint=True)
    assert 0.0 <= res.prob_attack_hidden <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert res.latency_ms > 0.0
