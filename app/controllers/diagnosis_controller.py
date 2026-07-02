"""GPU Insight Lab - Diagnosis Controller."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from diagnosis.engine import run_diagnosis
from diagnosis.scoring import compute_score, ScoreResult


def diagnose_session(session: Any) -> List[Dict[str, Any]]:
    return run_diagnosis(session)


def score_session(session: Any) -> Dict[str, Any]:
    result = compute_score(session)
    return dataclasses.asdict(result)
