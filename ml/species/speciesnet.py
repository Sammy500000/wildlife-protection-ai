from __future__ import annotations

from typing import Any


class SpeciesNetAdapter:
    """Isolation boundary for Google Wildlife SpeciesNet.

    SpeciesNet releases/runtime APIs can change; keeping integration here lets
    the pipeline remain stable and makes the exact model version auditable.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model: Any = None

    def load(self) -> None:
        if not self.model_path:
            raise RuntimeError("Set SPECIESNET_MODEL_PATH to the pinned SpeciesNet artifact/runtime.")
        raise NotImplementedError("Connect the pinned SpeciesNet runtime in this adapter.")

    def predict(self, crop: Any) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("SpeciesNet is not loaded")
        raise NotImplementedError
