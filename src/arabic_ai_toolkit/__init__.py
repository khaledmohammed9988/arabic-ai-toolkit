"""Arabic text utilities for reproducible AI evaluation."""

from .evaluation import CaseScore, EvaluationSummary, evaluate_case, evaluate_cases
from .normalization import NormalizationOptions, normalize_arabic

__all__ = [
    "CaseScore",
    "EvaluationSummary",
    "NormalizationOptions",
    "evaluate_case",
    "evaluate_cases",
    "normalize_arabic",
]
