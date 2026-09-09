"""
Training script for training Layer 1 lightweight detector and Layer 2 hidden state MLP probe.
"""

import sys
import os
import torch

# Ensure src module importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.layer1_fast.detector import Layer1Detector
from src.layer1_fast.trainer import Layer1Trainer
from src.layer2_hidden.hidden_extractor import TargetLLMHiddenExtractor
from src.layer2_hidden.prototype_engine import PrototypeEngine
from src.layer2_hidden.trainer import Layer2Trainer
from src.device import resolve_device, load_system_device, device_summary


def main():
    print("=== Training Three-Layer Cascade Models ===")
    device = resolve_device(load_system_device())
    print(f"Compute device: {device} ({device_summary()})")
    dataset = SyntheticDataGenerator.get_dataset(n_attacks=150, n_benign=150, n_not_inject=50, seed=42)
    print(f"Loaded dataset with {len(dataset)} total samples.")

    # 1. Train Layer 1 Detector
    # NOTE: this trains the toy surrogate ENCODER only (a few hundred k params,
    # average-pooled hashed tokens) — a demonstration of the trainer, NOT the
    # A1 ablation baseline (that is deliberately the untrained regex-boosted
    # heuristic, which is what actually pre-filters in the cascade). Metrics
    # are computed on the training set after 5 passes.
    print("\n--- Training Layer 1 Fast Pre-filter (surrogate encoder demo, 5 epochs) ---")
    l1_detector = Layer1Detector(device=str(device), use_surrogate=True)
    l1_trainer = Layer1Trainer(model=l1_detector.surrogate_model, device=str(device))
    for epoch in range(1, 6):
        loss = l1_trainer.train_epoch(dataset, batch_size=32)
        print(f"  epoch {epoch}: loss {loss:.4f}")
    l1_metrics = l1_trainer.evaluate(dataset)
    print(f"Layer 1 Train-Set Metrics (demo only): {l1_metrics}")

    # 2. Train Layer 2 MLP Probe & Compute Prototypes
    print("\n--- Training Layer 2 Hidden State MLP Probe ---")
    extractor = TargetLLMHiddenExtractor(device=str(device), use_surrogate=True)
    proto_engine = PrototypeEngine(num_layers=extractor.num_layers, d_model=extractor.d_model, device=str(device))
    l2_trainer = Layer2Trainer(extractor=extractor, prototype_engine=proto_engine, device=str(device))

    print("Fitting prototypes and building feature vectors...")
    X, y = l2_trainer.fit_prototypes_and_prepare_features(dataset)
    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    print("Training MLP probe...")
    losses = l2_trainer.train_probe(X, y, epochs=10, batch_size=32)
    l2_metrics = l2_trainer.evaluate(X, y)
    print(f"Layer 2 Final Epoch Loss: {losses[-1]:.4f}")
    print(f"Layer 2 Probe Metrics: {l2_metrics}")

    print("\n=== Model Training Completed Successfully ===")


if __name__ == "__main__":
    main()
