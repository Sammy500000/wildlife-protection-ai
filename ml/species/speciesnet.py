from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class SpeciesNetAdapter:
    """Adapter for Google's open-source SpeciesNet command-line runtime."""

    def __init__(self, country: str = "IND"):
        self.country = country

    @staticmethod
    def check_installation() -> None:
        result = subprocess.run(
            [sys.executable, "-m", "speciesnet.scripts.run_model", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("SpeciesNet is not installed. Run: python -m pip install speciesnet")

    def classify_folder(self, image_folder: Path, output_json: Path) -> dict[str, Any]:
        image_folder = image_folder.resolve()
        output_json = output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        self.check_installation()
        cmd = [sys.executable, "-m", "speciesnet.scripts.run_model", "--folders", str(image_folder), "--predictions_json", str(output_json)]
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SpeciesNet failed with exit code {result.returncode}")
        if not output_json.exists():
            raise RuntimeError(f"SpeciesNet completed but did not create {output_json}")
        return json.loads(output_json.read_text(encoding="utf-8"))
