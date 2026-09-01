from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class AnimalKingdomAdapter:
    """Parse Animal Kingdom action-recognition annotations into project manifests.

    The adapter preserves source multi-label actions. Mapping to the project's
    conservation behaviour ontology is intentionally a separate step.
    """

    def __init__(self, annotation_csv: str | Path, video_root: str | Path | None = None):
        self.annotation_csv = Path(annotation_csv)
        self.video_root = Path(video_root) if video_root else None

    @staticmethod
    def _labels(value: str) -> list[str]:
        value = (value or "").strip()
        if not value:
            return []
        for sep in (";", ","):
            if sep in value:
                return [x.strip() for x in value.split(sep) if x.strip()]
        return [value]

    def read(self) -> list[dict[str, Any]]:
        with self.annotation_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        out = []
        for row in rows:
            clip_path = row.get("clip_path") or row.get("video_path") or row.get("path") or ""
            if self.video_root and clip_path:
                clip_path = str((self.video_root / clip_path).resolve())
            out.append({
                "clip_id": row.get("clip_id", ""),
                "clip_number": row.get("clip_number", ""),
                "frame_number": row.get("frame_number", ""),
                "clip_path": clip_path,
                "action_labels": self._labels(row.get("action_labels", "")),
            })
        return out

    def to_manifest(self, output: str | Path) -> int:
        import json
        rows = self.read()
        payload = {
            "dataset": "Animal Kingdom",
            "task": "action_recognition",
            "items": rows,
            "source_annotation": str(self.annotation_csv.resolve()),
        }
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(rows)
