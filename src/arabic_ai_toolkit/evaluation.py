"""Transparent metrics for Arabic generation evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from .normalization import NormalizationOptions, normalize_arabic


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float


@dataclass(frozen=True)
class EvaluationSummary:
    cases: int
    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float
    details: tuple[CaseScore, ...]

    def to_dict(self, include_details: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "cases": self.cases,
            "exact_match": round(self.exact_match, 6),
            "token_precision": round(self.token_precision, 6),
            "token_recall": round(self.token_recall, 6),
            "token_f1": round(self.token_f1, 6),
        }
        if include_details:
            payload["details"] = [asdict(item) for item in self.details]
        return payload


def _token_metrics(reference: str, prediction: str) -> tuple[float, float, float]:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    overlap = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())

    precision = overlap / len(pred_tokens) if pred_tokens else float(not ref_tokens)
    recall = overlap / len(ref_tokens) if ref_tokens else float(not pred_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_case(
    case_id: str,
    reference: str,
    prediction: str,
    options: NormalizationOptions | None = None,
) -> CaseScore:
    normalized_reference = normalize_arabic(reference, options)
    normalized_prediction = normalize_arabic(prediction, options)
    precision, recall, f1 = _token_metrics(normalized_reference, normalized_prediction)
    return CaseScore(
        case_id=str(case_id),
        exact_match=float(normalized_reference == normalized_prediction),
        token_precision=precision,
        token_recall=recall,
        token_f1=f1,
    )


def evaluate_cases(
    cases: Iterable[dict[str, object]],
    options: NormalizationOptions | None = None,
) -> EvaluationSummary:
    scores: list[CaseScore] = []
    for index, case in enumerate(cases, start=1):
        if "reference" not in case or "prediction" not in case:
            raise ValueError(f"case {index} must include reference and prediction")
        scores.append(
            evaluate_case(
                str(case.get("id", index)),
                str(case["reference"]),
                str(case["prediction"]),
                options,
            )
        )

    if not scores:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, ())

    size = len(scores)
    return EvaluationSummary(
        cases=size,
        exact_match=sum(score.exact_match for score in scores) / size,
        token_precision=sum(score.token_precision for score in scores) / size,
        token_recall=sum(score.token_recall for score in scores) / size,
        token_f1=sum(score.token_f1 for score in scores) / size,
        details=tuple(scores),
    )
