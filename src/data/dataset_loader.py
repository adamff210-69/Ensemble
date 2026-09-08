"""
Dataset loader and standardization pipeline for prompt injection benchmarks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import random
import torch
from torch.utils.data import Dataset


@dataclass
class InjectionSample:
    """Standardized prompt injection benchmark sample."""
    prompt: str
    label: int  # 1 = Attack, 0 = Benign
    system_prompt: str = "You are a helpful AI assistant."
    model_response: str = "I can assist you with that request."
    source_dataset: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "label": self.label,
            "system_prompt": self.system_prompt,
            "model_response": self.model_response,
            "source_dataset": self.source_dataset,
            "metadata": self.metadata,
        }


class InjectionDataset(Dataset):
    """PyTorch Dataset wrapper for InjectionSample instances."""

    def __init__(self, samples: List[InjectionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> InjectionSample:
        return self.samples[idx]

    def filter_by_source(self, source_name: str) -> "InjectionDataset":
        filtered = [s for s in self.samples if s.source_dataset == source_name]
        return InjectionDataset(filtered)

    def get_labels(self) -> List[int]:
        return [s.label for s in self.samples]


class DatasetLoader:
    """Loader and splitter for prompt injection benchmark datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_from_jsonl(self, filepath: str, source_name: str = "custom") -> List[InjectionSample]:
        samples = []
        if not os.path.exists(filepath):
            return samples

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                samples.append(
                    InjectionSample(
                        prompt=item.get("prompt", ""),
                        label=int(item.get("label", 0)),
                        system_prompt=item.get("system_prompt", "You are a helpful AI assistant."),
                        model_response=item.get("model_response", "Standard response."),
                        source_dataset=item.get("source_dataset", source_name),
                        metadata=item.get("metadata", {}),
                    )
                )
        return samples

    def train_val_test_split(
        self,
        samples: List[InjectionSample],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[InjectionDataset, InjectionDataset, InjectionDataset]:
        """Split samples into train, validation, and test sets with seed reproducibility."""
        random.seed(seed)
        shuffled = list(samples)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train : n_train + n_val]
        test_samples = shuffled[n_train + n_val :]

        return (
            InjectionDataset(train_samples),
            InjectionDataset(val_samples),
            InjectionDataset(test_samples),
        )
