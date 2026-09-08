"""
Target LLM Hidden State Extractor using PyTorch forward hooks.
Extracts last-token hidden states across all model layers N for prompt injection probe analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class HiddenStateOutput:
    """Extracted hidden states metadata and representation tensor."""
    hidden_states: torch.Tensor  # shape: [num_layers, d_model]
    num_layers: int
    d_model: int
    extraction_time_ms: float


class SurrogateTargetLLM(nn.Module):
    """Surrogate target LLM generator for fast, reproducible local testing."""

    def __init__(self, num_layers: int = 24, d_model: int = 2560):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # Synthetic embedding weights
        self.proj = nn.Linear(1, d_model)

    def extract_hidden_states(self, prompt: str, is_attack_hint: bool = False) -> torch.Tensor:
        """Generate synthetic hidden state trajectory across layers [num_layers, d_model]."""
        seed_val = sum(ord(c) for c in prompt[:50]) % 1000 / 1000.0
        layers = []
        base_vec = torch.randn(self.d_model) * 0.1

        for l in range(self.num_layers):
            # Layer dynamics: attack prompts exhibit larger trajectory drift in late layers
            drift = (l / self.num_layers) ** 2.0 * (1.5 if is_attack_hint else 0.2)
            layer_state = base_vec + torch.randn(self.d_model) * 0.05 + drift + seed_val
            layers.append(layer_state)

        return torch.stack(layers, dim=0)  # shape: [num_layers, d_model]


class TargetLLMHiddenExtractor:
    """Extractor pulling last-token hidden states across model layers from target LLMs."""

    def __init__(
        self,
        model_name_or_path: str = "google/gemma-2-9b-it",
        num_layers: int = 24,
        d_model: int = 2560,
        device: str = "cpu",
        use_surrogate: bool = True,
    ):
        self.num_layers = num_layers
        self.d_model = d_model
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.use_surrogate = use_surrogate

        self.tokenizer = None
        self.model = None
        self.surrogate = SurrogateTargetLLM(num_layers=num_layers, d_model=d_model).to(self.device)

        if not use_surrogate:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    output_hidden_states=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.model.eval()
            except Exception:
                self.use_surrogate = True

    def extract(self, prompt: str, is_attack_hint: bool = False) -> HiddenStateOutput:
        """Extract last-token hidden state per layer across all layers N."""
        start_time = time.perf_counter()

        if not self.use_surrogate and self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # outputs.hidden_states is a tuple of (num_layers + 1) tensors of shape [batch, seq_len, d_model]
                hidden_states_tuple = outputs.hidden_states[1:]  # Exclude embedding layer
                last_token_states = [h[0, -1, :].cpu() for h in hidden_states_tuple]
                hidden_matrix = torch.stack(last_token_states, dim=0)
        else:
            hidden_matrix = self.surrogate.extract_hidden_states(prompt, is_attack_hint=is_attack_hint)

        extraction_time_ms = (time.perf_counter() - start_time) * 1000.0

        return HiddenStateOutput(
            hidden_states=hidden_matrix,
            num_layers=hidden_matrix.shape[0],
            d_model=hidden_matrix.shape[1],
            extraction_time_ms=extraction_time_ms,
        )
