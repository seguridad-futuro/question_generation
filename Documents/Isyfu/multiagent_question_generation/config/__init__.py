"""Configuration module for the question generation system."""

from .settings import Settings, get_settings
from .thresholds import QualityThresholds, DeduplicationConfig

__all__ = [
    "Settings",
    "get_settings",
    "QualityThresholds",
    "DeduplicationConfig",
]
