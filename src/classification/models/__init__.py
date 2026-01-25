"""Model architectures for classification."""

from .backbones import get_backbone, AVAILABLE_BACKBONES
from .heads import add_universal_head, attention_block
from .builder import build_model

__all__ = [
    "get_backbone",
    "AVAILABLE_BACKBONES",
    "add_universal_head",
    "attention_block",
    "build_model",
]
