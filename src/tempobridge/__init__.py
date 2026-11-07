"""TempoBridge: ECG-assisted, PPG-only representation learning."""

from .config import ModelConfig
from .model import TempoBridge

__all__ = ["ModelConfig", "TempoBridge"]
__version__ = "0.4.0"
