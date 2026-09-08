"""
Central device resolution for the cascade pipeline.

All components resolve their compute device through this module so a single
`system.device` setting in config/cascade_config.yaml ("auto" | "cuda" |
"mps" | "cpu") governs the entire pipeline (L1 detector, L2 extractor +
prototypes + probe, L3 verifier, trainers, and the FastAPI service).
"""

import os
from typing import Optional

import torch

__all__ = ["resolve_device", "load_system_device", "device_summary"]


def resolve_device(preferred: Optional[str] = "auto") -> torch.device:
    """
    Resolve a requested device name into a torch.device.

    - "auto": CUDA if available, else MPS (Apple Silicon), else CPU.
    - "cuda": requires an available GPU; raises RuntimeError otherwise.
    - "mps":  requires an Apple Metal backend; raises RuntimeError otherwise.
    - "cpu":  always CPU.
    - anything else is passed through to torch.device (e.g. "cuda:1").
    """
    pref = (preferred or "auto").strip().lower()

    if pref in ("auto", ""):
        if torch.cuda.is_available():
            # Normalize to a concrete index: torch.device("cuda") (index=None)
            # is NOT == torch.device("cuda:0"), even though tensors always
            # report the concrete index.
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no GPU is available. "
                "Install a CUDA-enabled build of PyTorch and verify with `nvidia-smi`."
            )
        return torch.device("cuda", torch.cuda.current_device())

    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but the Apple Metal backend is unavailable."
            )
        return torch.device("mps")

    if pref == "cpu":
        return torch.device("cpu")

    # Pass-through for specific devices like "cuda:0", "cuda:1", ...
    return torch.device(pref)


def load_system_device(config_path: Optional[str] = None) -> str:
    """
    Read `system.device` from cascade_config.yaml.

    Falls back to "auto" if the file or the key is missing. The default
    search order is: explicit path, CWD-relative config/, and the repo-root
    config/ directory next to this package.
    """
    import yaml

    candidates = [
        config_path,
        os.path.join("config", "cascade_config.yaml"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "cascade_config.yaml",
        ),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                return str(data.get("system", {}).get("device", "auto"))
            except Exception:
                return "auto"

    return "auto"


def device_summary() -> str:
    """Human-readable summary of the compute environment."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA [{name}, {mem_gb:.1f} GB, capability {torch.cuda.get_device_capability(0)}]"
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU (no GPU detected)"
