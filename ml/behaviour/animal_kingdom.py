from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

# Animal Kingdom action recognition annotations follow Charades-style CSVs.
# The official dataset contains 140 action labels and multi-label examples.
# We intentionally map only a conservative subset to the project's V1 ontology.

DEFAULT_LABEL_MAP = {
    "rest": "RESTING",
    "sleep": "RESTING",
    "sit": "RESTING",
    "sitting": "RESTING",
    "stand": "RESTING",
    "walk": "NORMAL_MOVEMENT",
    "walking": "NORMAL_MOVEMENT",
    "move": "NORMAL_MOVEMENT",
    "run": "RUNNING",
    "running": "RUNNING",
    "chase": "CHASING",
    "chasing": "CHASING",
    "attack": "AGGRESSIVE_ABNORMAL",
    "attacking": "AGGRESSIVE_ABNORMAL",
    "fight": "AGGRESSIVE_ABNORMAL",
    "fighting": "AGGRESSIVE_ABNORMAL",
    "charge": "AGGRESSIVE_ABNORMAL",
    "charging": "AGGRESSIVE_ABNORMAL",
}


def normalize_label(label: str) -> str | None:
    key = label.strip().lower().replace("-", "_")
    if key in DEFAULT_LABEL_MAP:
        return DEFAULT_LABEL_MAP[key]
    for source, target in DEFAULT_LABEL_MAP.items():
        if source in key:
            return target
    return None


def parse_action_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_raw = row.get("action_labels", "")
            labels = [x.strip() for x in labels_raw.replace(";", ",").split(",") if x.strip()]
            mapped = sorted({m for x in labels if (m := normalize_label(x))})
            if not mapped:
                continue
            rows.append({
                "clip_id": row.get("clip_id", ""),
                "clip_number": row.get("clip_number", ""),
                "frame_number": row.get("frame_number", ""),
                "clip_path": row.get("clip_path", ""),
                "source_labels": labels,
                "labels": mapped,
            })
    return rows


def build_manifest(annotation_root: Path, output_json: Path) -> dict:
    records = []
    for csv_path in sorted(annotation_root.rglob("*.csv")):
        records.extend(parse_action_csv(csv_path))

    manifest = {
        "dataset": "Animal Kingdom",
        "source": "CVPR2022 official dataset",
        "ontology": ["RESTING", "NORMAL_MOVEMENT", "RUNNING", "CHASING", "AGGRESSIVE_ABNORMAL", "UNKNOWN"],
        "records": records,
        "record_count": len(records),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", default="data/processed/animal_kingdom_manifest.json")
    a = p.parse_args()
    m = build_manifest(Path(a.annotations), Path(a.output))
    print(json.dumps({"record_count": m["record_count"], "output": a.output}, indent=2))
