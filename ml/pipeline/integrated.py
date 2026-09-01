from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from ml.pipeline.video import read_video
from ml.species.speciesnet import SpeciesNetAdapter
from ml.risk.engine import RiskInput, score_risk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _prediction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("predictions", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _best_species(rows: list[dict[str, Any]]) -> tuple[str, float]:
    candidates = []
    for row in rows:
        for key in ("prediction", "classification", "classifications"):
            value = row.get(key)
            if isinstance(value, dict):
                name = next((value.get(k) for k in ("label", "class", "species", "scientific_name", "common_name") if isinstance(value.get(k), str) and value.get(k)), None)
                if name:
                    score = next((float(value.get(k, 0.0)) for k in ("score", "confidence", "probability") if isinstance(value.get(k), (int, float, str))), 0.0)
                    candidates.append((name, score))
            elif isinstance(value, str) and value:
                candidates.append((value, 0.0))
        for k in ("label", "species", "scientific_name", "common_name"):
            if isinstance(row.get(k), str) and row[k]:
                try:
                    score = float(next((row.get(s, 0.0) for s in ("score", "confidence", "probability") if row.get(s) is not None), 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                candidates.append((row[k], score))
    return max(candidates, key=lambda x: x[1]) if candidates else ("UNKNOWN", 0.0)


def extract_track_crops(video: Path, tracks: list[dict[str, Any]], crop_root: Path, per_track: int = 8) -> dict[str, list[Path]]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        if row.get("track_id") is not None and row.get("class_name") == "animal":
            by_track[str(row["track_id"])].append(row)

    selected = {}
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: r["frame_index"])
        if len(rows) <= per_track:
            selected[tid] = rows
        else:
            step = (len(rows) - 1) / (per_track - 1)
            selected[tid] = [rows[round(i * step)] for i in range(per_track)]

    wanted = {(tid, int(r["frame_index"])) for tid, rows in selected.items() for r in rows}
    out: dict[str, list[Path]] = defaultdict(list)
    crop_root.mkdir(parents=True, exist_ok=True)

    for frame_index, _, frame in read_video(video, sample_every=1):
        for tid, fi in list(wanted):
            if fi != frame_index:
                continue
            row = next(r for r in selected[tid] if int(r["frame_index"]) == frame_index)
            x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            path = crop_root / f"track_{tid}" / f"frame_{frame_index:08d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame[y1:y2, x1:x2])
            out[tid].append(path)
        if selected and len(out) == len(selected) and all(len(out[k]) == len(selected[k]) for k in selected):
            break
    return dict(out)


def run_integrated(video: Path, output_dir: Path, sample_every: int = 3, species_samples: int = 8) -> dict[str, Any]:
    from scripts.run_video import run_video

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "video"
    summary = run_video(video, video_dir, sample_every=sample_every, confidence=0.25)
    tracks = json.loads((video_dir / "tracks.json").read_text(encoding="utf-8")).get("tracks", [])

    crop_root = output_dir / "track_crops"
    crops = extract_track_crops(video, tracks, crop_root, per_track=species_samples)

    species_root = output_dir / "species"
    species_root.mkdir(parents=True, exist_ok=True)
    species_json = species_root / "predictions.json"
    all_images = [p for paths in crops.values() for p in paths if p.suffix.lower() in IMAGE_EXTS]
    if all_images:
        payload = SpeciesNetAdapter(country="IND").classify_folder(crop_root, species_json)
    else:
        payload = {"predictions": []}
        species_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = _prediction_rows(payload)
    track_species = {}
    for tid, paths in crops.items():
        path_set = {str(p.resolve()) for p in paths}
        matching = [r for r in rows if isinstance(r.get("filepath") or r.get("file") or r.get("image") or r.get("path"), str) and str(Path(r.get("filepath") or r.get("file") or r.get("image") or r.get("path")).resolve()) in path_set]
        species, confidence = _best_species(matching)
        track_species[tid] = {"species": species, "confidence": confidence, "sample_count": len(paths), "classified_count": len(matching)}

    humans_present = any(r.get("class_name") == "person" for r in tracks)
    risk_events = []
    for tid, info in track_species.items():
        risk = score_risk(RiskInput(
            species=info["species"], behaviour="UNKNOWN", human_present=humans_present,
            distance_m=None, persistence_s=0.0, confidence=float(info["confidence"]),
        ))
        risk_events.append({
            "risk_event_id": str(uuid.uuid4()), "track_id": tid,
            "species": info["species"], "behaviour": "UNKNOWN",
            "human_present": humans_present, "risk": risk, "evidence_uri": None,
        })

    result = {
        "input": str(video), "summary": summary, "species": track_species,
        "risk_events": risk_events,
        "models": {"detector": "MegaDetectorV6 MDV6-yolov9-c", "tracker": "ByteTrack", "species": "SpeciesNet 5.x", "behavior": "ResNet18-LSTM (not trained yet)"},
        "outputs": {"tracks": str(video_dir / "tracks.json"), "species": str(species_json), "crops": str(crop_root)},
    }
    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
