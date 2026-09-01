from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskInput:
    species: str
    behaviour: str
    human_present: bool
    distance_m: float | None
    zone_risk: float = 0.0
    time_risk: float = 0.0
    persistence_s: float = 0.0
    confidence: float = 1.0


BEHAVIOUR_WEIGHT = {"RESTING": 0.0, "NORMAL_MOVEMENT": 0.05, "RUNNING": 0.20, "CHASING": 0.40, "AGGRESSIVE/ABNORMAL": 0.50, "UNKNOWN": 0.0}


def score_risk(x: RiskInput) -> dict[str, Any]:
    factors: list[dict[str, float | str]] = []
    score = 0.0
    b = BEHAVIOUR_WEIGHT.get(x.behaviour, 0.0)
    score += b; factors.append({"name": "behaviour", "value": x.behaviour, "contribution": b})
    hp = 0.25 if x.human_present else 0.0
    score += hp; factors.append({"name": "human_presence", "value": float(x.human_present), "contribution": hp})
    proximity = 0.0 if x.distance_m is None else max(0.0, min(0.25, 0.25 * (1.0 - x.distance_m / 100.0)))
    score += proximity; factors.append({"name": "proximity", "value": x.distance_m if x.distance_m is not None else -1.0, "contribution": proximity})
    persistence = min(0.10, max(0.0, x.persistence_s / 60.0 * 0.10))
    score += persistence; factors.append({"name": "persistence", "value": x.persistence_s, "contribution": persistence})
    score += max(0.0, min(0.10, x.zone_risk)); factors.append({"name": "zone", "value": x.zone_risk, "contribution": max(0.0, min(0.10, x.zone_risk))})
    score += max(0.0, min(0.10, x.time_risk)); factors.append({"name": "time", "value": x.time_risk, "contribution": max(0.0, min(0.10, x.time_risk))})
    uncertainty = max(0.0, min(0.20, 0.20 * (1.0 - x.confidence)))
    score -= uncertainty; factors.append({"name": "uncertainty_penalty", "value": x.confidence, "contribution": -uncertainty})
    score = max(0.0, min(1.0, score))
    if x.behaviour == "UNKNOWN" or x.confidence < 0.35:
        level = "MEDIUM" if x.human_present else "UNKNOWN"
    elif score >= 0.75: level = "CRITICAL"
    elif score >= 0.50: level = "HIGH"
    elif score >= 0.25: level = "MEDIUM"
    else: level = "LOW"
    return {"risk_score": round(score, 4), "risk_level": level, "factors": factors}
