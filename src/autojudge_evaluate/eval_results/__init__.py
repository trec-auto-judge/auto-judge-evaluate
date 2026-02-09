"""
EvalResult - Clean evaluation result container for IR experiments.

Components:
- EvalResult: Immutable read-only container (eval_result.py)
- IO: Load/write in various formats (io.py)
- Verification: Validation and checking (verification.py)
- Builder: Construction, filtering, aggregation (builder.py)
"""

from .eval_result import EvalResult, EvalEntry, MeasureSpecs, ALL_TOPIC_ID, MeasureDtype
from .builder import EvalResultBuilder
from .io import load, write, write_by_run

__all__ = [
    "EvalResult",
    "EvalEntry",
    "EvalResultBuilder",
    "MeasureSpecs",
    "ALL_TOPIC_ID",
    "MeasureDtype",
    "load",
    "write",
    "write_by_run",
]
