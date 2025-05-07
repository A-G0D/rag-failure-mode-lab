"""Stress taxonomy: failure-inducing query categories."""
from __future__ import annotations

from enum import Enum


class StressType(str, Enum):
    NORMAL = "normal"
    ADVERSARIAL = "adversarial"
    AMBIGUOUS = "ambiguous"
    LONG_CONTEXT_OVERLOAD = "long_context_overload"

    @classmethod
    def from_str(cls, value: str) -> "StressType":
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(t.value for t in cls)
            raise ValueError(f"unknown stress_type {value!r}; valid: {valid}") from exc


STRESS_TYPES: tuple[StressType, ...] = (
    StressType.ADVERSARIAL,
    StressType.AMBIGUOUS,
    StressType.LONG_CONTEXT_OVERLOAD,
)


def is_valid_stress_type(value: str) -> bool:
    return value in {t.value for t in StressType}
