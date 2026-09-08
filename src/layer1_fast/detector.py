"""
Layer 1: Lightweight LLM Detector (<50ms latency budget).
Provides fast pre-filtering for prompt injection attacks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..device import resolve_device


@dataclass
class Layer1Result:
    """Detection outcome from Layer 1 detector."""
    prob_attack: float
    confidence: float
    trigger_flags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    action: str = "PENDING"  # PASS, BLOCK, ROUTE_TO_L2


class SurrogateLayer1Classifier(nn.Module):
    """Surrogate lightweight classifier used for sub-50ms execution and CPU fallback."""

    TRIGGER_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instruction",
        r"system\s+override",
        r"disregard\s+(ethical\s+)?constraint",
        r"do\s+anything\s+now",
        r"dan\s+mode",
        r"developer\s+mode",
        r"emergency\s+override",
        r"bypass\s+security",
        r"sudo\s+mode",
        r"reveal\s+(the\s+)?(secret|system\s+prompt|password)",
        r"instruction\s+override",
    ]

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(text_indices))


class Layer1Detector:
    """Layer 1 Fast Pre-Filtering LLM Detector."""

    def __init__(
        self,
        model_name_or_path: str = "protectai/deberta-v3-base-prompt-injection-v2",
        device: str = "auto",
        use_surrogate: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = resolve_device(device)
        self.use_surrogate = use_surrogate

        self.tokenizer = None
        self.model = None
        self.surrogate_model = SurrogateLayer1Classifier().to(self.device)

        if not use_surrogate:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
                self.model.eval()
            except Exception:
                # Fallback to surrogate model if remote HuggingFace checkpoint is unavailable
                self.use_surrogate = True

    def scan_trigger_flags(self, prompt: str) -> List[str]:
        """Scan prompt text for known injection trigger regex patterns."""
        flags = []
        prompt_lower = prompt.lower()
        for pattern in SurrogateLayer1Classifier.TRIGGER_PATTERNS:
            if re.search(pattern, prompt_lower):
                flags.append(pattern)
        return flags

    def predict(self, prompt: str) -> Layer1Result:
        """Execute fast inference on input prompt within <50ms latency budget."""
        start_time = time.perf_counter()
        trigger_flags = self.scan_trigger_flags(prompt)

        if not self.use_surrogate and self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Assume index 1 corresponds to attack class
                prob_attack = probs[0, 1].item() if probs.shape[-1] > 1 else torch.sigmoid(logits[0, 0]).item()
        else:
            # Fast surrogate scoring (Heuristic + lightweight embed score)
            words = [abs(hash(w)) % 10000 for w in prompt.lower().split()]
            if not words:
                words = [0]
            indices_tensor = torch.tensor([words], dtype=torch.long).to(self.device)
            with torch.no_grad():
                raw_score = self.surrogate_model(indices_tensor).item()

            # Boost score based on rule-based trigger flags for high-precision detection
            if trigger_flags:
                prob_attack = min(1.0, raw_score + 0.45 + 0.15 * len(trigger_flags))
            else:
                prob_attack = max(0.01, raw_score * 0.4)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        confidence = abs(prob_attack - 0.5) * 2.0  # Confidence scaled to [0, 1]

        return Layer1Result(
            prob_attack=float(prob_attack),
            confidence=float(confidence),
            trigger_flags=trigger_flags,
            latency_ms=latency_ms,
        )
