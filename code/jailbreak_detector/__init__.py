"""
Jailbreak Detector Package

A tool for detecting jailbreak patterns in LLM responses using another LLM as a judge.
"""

from .detector import JailbreakDetector
from .evaluator import JailbreakEvaluator
from .utils import (
    check_dependencies,
    get_available_device,
    create_sample_data,
    infer_detection_threshold,
    format_duration
)

__version__ = "0.1.0" 