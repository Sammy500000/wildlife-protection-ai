from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RiskEvent:
    species: str
    behaviour: str
    human_present: bool
    distance_m: float | None
    zone_type: str
    time_of_day: str
    persistence_s: float
    uncertainty: float
    risk_score: float
    risk_level: str
    factors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskEngine:
    """Transparent V1 decision-support scorer.

    Weights are engineering placeholders, not scientific truth. They should
    be calibrated against labelled scenarios and ranger/domain-expert review.
    """

    SPECIES = {"elephant": 18, "tiger": 22, "leopard": 20, "bear": 16}
    BEHAVIOUR = {
        "RESTING": 0, "NORMAL_MOVEMENT": 4, "RUNNING": 10,
        "CHASING": 22, "AGGRESSIVE_ABNORMAL": 30, "UNKNOWN": 8,
    }
    ZONE = {"forest_core": 0, "corridor": 5, "road": 10, "railway": 14,
            "village_boundary": 18, "settlement": 22}

    def score(
        self, species: str, behaviour: str, human_present: bool,
        distance_m: float | None = None, zone_type: str = "forest_core",
        time_of_day: str = "day", persistence_s: float = 0.0,
        confidence: float = 1.0,
    ) -> RiskEvent:
        factors = []
        def add(name, value, contribution):
            factors.append({"name": name, "value": value, "contribution": contribution})

        s = self.SPECIES.get(species.lower(), 8)
        b = self.BEHAVIOUR.get(behaviour, 8)
        z = self.ZONE.get(zone_type, 5)
        human = 20 if human_present else 0
        proximity = 0
        if distance_m is not None:
            if distance_m <= 5: proximity = 25
            elif distance_m <= 15: proximity = 18
            elif distance_m <= 30: proximity = 10
            elif distance_m <= 60: proximity = 4
        persistence = min(max(persistence_s / 30.0, 0.0), 1.0) * 10
        uncertainty_penalty = max(0.0, 1.0 - confidence) * 15

        add("species", species, s); add("behaviour", behaviour, b)
        add("human_presence", human_present, human); add("proximity_m", distance_m, proximity)
        add("zone", zone_type, z); add("persistence_s", persistence_s, persistence)
        add("uncertainty", confidence, -uncertainty_penalty)

        raw = s + b + human + proximity + z + persistence - uncertainty_penalty
        score = max(0.0, min(100.0, raw))
        level = "LOW" if score < 25 else "MEDIUM" if score < 50 else "HIGH" if score < 75 else "CRITICAL"
        return RiskEvent(species, behaviour, human_present, distance_m, zone_type,
                         time_of_day, persistence_s, confidence, score, level, factors)
