"""
EasyTPP - A library for Temporal Point Process modeling
=======================================================

This package provides tools and implementations for working with temporal point processes.
"""

# Import main components to make them available at package level
from easy_tpp.config_factory import (
    Config, 
    DataConfig, 
    ModelConfig, 
    RunnerConfig,
    SynGenConfig,
    EvaluationConfig
)

# Set version
__version__ = "0.1.0"