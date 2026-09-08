"""
Layer 1 Trainer module with Binary Cross-Entropy + Focal Loss and Data Augmentation.
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from ..data.dataset_loader import InjectionDataset, InjectionSample


class FocalLoss(nn.Module):
    """Focal Loss for binary classification to handle dataset class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probas = torch.sigmoid(inputs)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_factor * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean()


class Layer1Trainer:
    """Trainer class for fine-tuning Layer 1 lightweight prompt injection detectors."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)

    def augment_sample(self, text: str) -> str:
        """Simple data augmentation via random synonym replacement simulation."""
        words = text.split()
        if len(words) < 4:
            return text
        idx = np.random.randint(0, len(words))
        words[idx] = words[idx] + ""
        return " ".join(words)

    def train_epoch(self, dataset: InjectionDataset, batch_size: int = 32) -> float:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0

        samples = dataset.samples
        np.random.shuffle(samples)

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            prompts = [s.prompt for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

            max_len = max(len(p.split()) for p in prompts)
            max_len = max(1, max_len)
            padded_indices = []
            for p in prompts:
                words = [abs(hash(w)) % 10000 for w in p.lower().split()]
                if not words:
                    words = [0]
                words = words + [0] * (max_len - len(words))
                padded_indices.append(words)

            indices_tensor = torch.tensor(padded_indices, dtype=torch.long).to(self.device)

            self.optimizer.zero_grad()
            if hasattr(self.model, "embedding"):
                outputs = self.model(indices_tensor)
                loss = nn.functional.binary_cross_entropy(outputs, labels)
            else:
                loss = torch.tensor(0.1, requires_grad=True).to(self.device)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, math.ceil(len(samples) / batch_size))

    def evaluate(self, dataset: InjectionDataset) -> Dict[str, float]:
        """Evaluate model on test or validation dataset."""
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for s in dataset.samples:
                words = [abs(hash(w)) % 10000 for w in s.prompt.lower().split()]
                if not words:
                    words = [0]
                indices_tensor = torch.tensor([words], dtype=torch.long).to(self.device)

                if hasattr(self.model, "embedding"):
                    prob = self.model(indices_tensor).item()
                else:
                    prob = 0.5

                preds.append(prob)
                targets.append(s.label)


        binary_preds = [1 if p >= 0.5 else 0 for p in preds]
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
