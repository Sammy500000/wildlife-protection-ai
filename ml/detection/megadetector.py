from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectorConfig:
    model_path: str | None = None
    confidence: float = 0.25


class MegaDetectorAdapter:
    """Thin adapter boundary for Microsoft's MegaDetector.

    The external model/runtime is intentionally isolated here so the rest of the
    application only consumes the project's Detection contract. Install and pin
    the exact MegaDetector release/runtime used by the experiment separately.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model: Any = None

    def load(self) -> None:
        if not self.config.model_path:
            raise RuntimeError("Set DETECTOR_MODEL_PATH to a downloaded MegaDetector model/runtime artifact.")
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(self.config.model_path)
        # Runtime-specific loading belongs here. We do not silently substitute
        # another detector because that would invalidate experiment provenance.
        raise NotImplementedError("Connect the pinned MegaDetector runtime in this adapter.")

    def predict(self, frame: Any) -> list[dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Detector is not loaded")
        raise NotImplementedError
