"""
Layer 2 Probe Trainer module.
Trains MLP classifier on aggregated layer-wise hidden-state prototype feature vectors.
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from .mlp_probe import Layer2MLPProbe
from .prototype_engine import PrototypeEngine
from .hidden_extractor import TargetLLMHiddenExtractor
from ..data.dataset_loader import InjectionDataset
from ..device import resolve_device


class Layer2Trainer:
    """Trainer pipeline for Layer 2 representation MLP probe."""

    def __init__(
        self,
        extractor: TargetLLMHiddenExtractor,
        prototype_engine: PrototypeEngine,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.device = resolve_device(device)
        self.extractor = extractor
        self.prototype_engine = prototype_engine

        # Calculate feature dimension
        num_layers = extractor.num_layers
        self.feature_dim = num_layers * 3 + (num_layers - 1)

        self.probe = Layer2MLPProbe(input_dim=self.feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()

    def fit_prototypes_and_prepare_features(
        self, dataset: InjectionDataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden states, calculate class prototypes, and create training feature matrices."""
        attack_states = []
        benign_states = []
        all_states = []
        labels = []

        for sample in dataset.samples:
            # Label-free extraction: the representation must come from the
            # prompt content, never from the ground-truth label (leaking the
            # label into features makes the probe perfectly fit and blind).
            hidden_out = self.extractor.extract(sample.prompt)
            if sample.label == 1:
                attack_states.append(hidden_out.hidden_states)
            else:
                benign_states.append(hidden_out.hidden_states)

            all_states.append(hidden_out.hidden_states)
            labels.append(sample.label)

        # Fit prototype engine
        self.prototype_engine.compute_prototypes(attack_states, benign_states)

        # Extract feature vectors for each sample
        feature_list = []
        for state in all_states:
            feat_struct = self.prototype_engine.extract_features(state)
            feature_list.append(feat_struct.feature_vector)

        X = torch.stack(feature_list, dim=0)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        return X, y

    def train_probe(
        self, X: torch.Tensor, y: torch.Tensor, epochs: int = 15, batch_size: int = 64
    ) -> List[float]:
        """Train MLP probe model on feature tensors."""
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        self.probe.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.probe(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(dataloader))
            losses.append(avg_loss)

        return losses

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate probe model performance."""
        self.probe.eval()
        with torch.no_grad():
            preds = self.probe(X.to(self.device)).cpu().numpy().flatten()
            targets = y.numpy().flatten()

        binary_preds = (preds >= 0.5).astype(int)
        auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.5
        f1 = f1_score(targets, binary_preds, zero_division=0)
        precision = precision_score(targets, binary_preds, zero_division=0)
        recall = recall_score(targets, binary_preds, zero_division=0)

        return {
            "auroc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
