from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "imageomics/KABR-mini-scene-raw-videos"
REPO_TYPE = "dataset"

BEHAVIORS = [
    "Walk",
    "Trot",
    "Run",
    "Graze",
    "Browse",
    "Head Up",
    "Auto-Groom",
    "Occluded",
]

VIDEO_RE = re.compile(r"^(\d+)\.mp4$", re.IGNORECASE)


def parse_behavior(xml_file: Path) -> str | None:
    try:
        root = ET.parse(xml_file).getroot()
    except (ET.ParseError, OSError):
        return None

    counts: Counter[str] = Counter()
    for attr in root.iter("attribute"):
        if attr.attrib.get("name", "").strip().lower() == "behavior":
            value = (attr.text or "").strip()
            for label in BEHAVIORS:
                if value.lower() == label.lower():
                    counts[label] += 1

    return counts.most_common(1)[0][0] if counts else None


def discover_pairs():
    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            recursive=True,
        )
    )

    action_paths = {
        item.path
        for item in entries
        if "/actions/" in item.path
        and item.path.lower().endswith(".xml")
    }
    video_paths = {
        item.path
        for item in entries
        if item.path.lower().endswith(".mp4")
        and VIDEO_RE.match(Path(item.path).name)
    }

    pairs = []
    for action in sorted(action_paths):
        stem = Path(action).stem
        parent = Path(action).parent.parent.as_posix()
        video = f"{parent}/{stem}.mp4"
        if video in video_paths:
            pairs.append((video, action))

    return pairs


def scan_one(pair, cache_dir: Path):
    video_path, action_path = pair
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=action_path,
        repo_type=REPO_TYPE,
        local_dir=cache_dir,
    )
    return video_path, action_path, parse_behavior(Path(local))


def scan_annotations(pairs, cache_dir: Path, workers: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one, pair, cache_dir): pair
            for pair in pairs
        }
        for i, future in enumerate(cf.as_completed(futures), 1):
            try:
                video, action, behavior = future.result()
                if behavior:
                    records.append(
                        {
                            "video": video,
                            "annotation": action,
                            "behavior": behavior,
                        }
                    )
            except Exception as exc:
                pair = futures[future]
                print(f"[WARN] {pair[1]}: {exc}")

            if i % 50 == 0 or i == len(futures):
                print(f"  scanned {i}/{len(futures)} annotations")

    return records


def select_balanced(records, per_class):
    selected = []
    counts = Counter()

    # Deterministic ordering keeps repeated runs reproducible.
    for record in sorted(records, key=lambda x: (x["behavior"], x["video"])):
        label = record["behavior"]
        if counts[label] < per_class:
            selected.append(record)
            counts[label] += 1

    print("\nAvailable / selected:")
    for label in BEHAVIORS:
        print(f"  {label:12s} {sum(r['behavior'] == label for r in records):4d} / {counts[label]:3d}")

    missing = [label for label in BEHAVIORS if counts[label] < per_class]
    if missing:
        print("\n[WARN] Not enough examples for: " + ", ".join(missing))

    return selected


def download_selected(records, output_dir: Path, workers: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_one(record):
        video_local = hf_hub_download(
            repo_id=REPO_ID,
            filename=record["video"],
            repo_type=REPO_TYPE,
            local_dir=output_dir,
        )
        xml_local = hf_hub_download(
            repo_id=REPO_ID,
            filename=record["annotation"],
            repo_type=REPO_TYPE,
            local_dir=output_dir,
        )
        return record, video_local, xml_local

    rows = []
    with cf.ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = [executor.submit(download_one, r) for r in records]
        for i, future in enumerate(cf.as_completed(futures), 1):
            record, video_local, xml_local = future.result()
            rows.append(
                {
                    "video": str(Path(video_local).relative_to(output_dir)),
                    "annotation": str(Path(xml_local).relative_to(output_dir)),
                    "label": record["behavior"],
                    "source_video": record["video"],
                    "source_annotation": record["annotation"],
                }
            )
            print(f"  downloaded {i}/{len(futures)}: {record['behavior']}")

    manifest = output_dir / "kabr_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "annotation",
                "label",
                "source_video",
                "source_annotation",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["label"], x["video"])))

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download a small balanced KABR mini-scene subset without downloading full drone videos."
    )
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default="data/raw/kabr")
    args = parser.parse_args()

    output = Path(args.output)

    print("[1/3] Discovering KABR mini-scene video/XML pairs...")
    pairs = discover_pairs()
    print(f"Discovered {len(pairs)} candidate pairs.")

    print(f"[2/3] Concurrently scanning XML annotations ({args.workers} workers)...")
    records = scan_annotations(pairs, output / "_annotation_cache", args.workers)
    selected = select_balanced(records, args.per_class)

    if not selected:
        raise RuntimeError("No labelled KABR mini-scenes were found.")

    print(f"\n[3/3] Downloading {len(selected)} selected MP4/XML pairs...")
    manifest = download_selected(selected, output, args.workers)

    print("\nKABR_DOWNLOAD_OK")
    print(f"Selected clips: {len(selected)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
