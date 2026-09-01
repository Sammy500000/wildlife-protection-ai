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
    "Walk", "Trot", "Run", "Graze",
    "Browse", "Head Up", "Auto-Groom", "Occluded",
]
VIDEO_RE = re.compile(r"^(\d+)\.mp4$", re.IGNORECASE)


def parse_behavior(xml_file: Path) -> str | None:
    try:
        root = ET.parse(xml_file).getroot()
    except (ET.ParseError, OSError):
        return None
    counts = Counter()
    for attr in root.iter("attribute"):
        if attr.attrib.get("name", "").strip().lower() != "behavior":
            continue
        value = (attr.text or "").strip().lower()
        for label in BEHAVIORS:
            if value == label.lower():
                counts[label] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def discover_pairs():
    api = HfApi()
    entries = list(api.list_repo_tree(
        repo_id=REPO_ID, repo_type=REPO_TYPE, recursive=True
    ))
    actions = {
        item.path for item in entries
        if "/actions/" in item.path and item.path.lower().endswith(".xml")
    }
    videos = {
        item.path for item in entries
        if item.path.lower().endswith(".mp4")
        and VIDEO_RE.match(Path(item.path).name)
    }
    pairs = []
    for action in sorted(actions):
        stem = Path(action).stem
        video = f"{Path(action).parent.parent.as_posix()}/{stem}.mp4"
        if video in videos:
            pairs.append((video, action))
    return pairs


def scan_one(pair, cache_dir):
    video, action = pair
    local = hf_hub_download(
        repo_id=REPO_ID, filename=action,
        repo_type=REPO_TYPE, local_dir=cache_dir
    )
    return video, action, parse_behavior(Path(local))


def scan_annotations(pairs, cache_dir, workers, per_class):
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    counts = Counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_one, p, cache_dir): p for p in pairs}
        for i, future in enumerate(cf.as_completed(futures), 1):
            try:
                video, action, behavior = future.result()
                if behavior and counts[behavior] < per_class:
                    records.append({
                        "video": video,
                        "annotation": action,
                        "behavior": behavior,
                    })
                    counts[behavior] += 1
            except Exception as exc:
                print(f"[WARN] {futures[future][1]}: {exc}")

            if i % 25 == 0 or i == len(futures):
                status = " ".join(
                    f"{b}={counts[b]}/{per_class}" for b in BEHAVIORS
                )
                print(f"  scanned {i}/{len(futures)} | {status}")

            if all(counts[b] >= per_class for b in BEHAVIORS):
                for f in futures:
                    f.cancel()
                print("  all requested classes found; stopping annotation scan early")
                break
    return records


def select_balanced(records, per_class):
    selected, counts = [], Counter()
    for r in sorted(records, key=lambda x: (x["behavior"], x["video"])):
        if counts[r["behavior"]] < per_class:
            selected.append(r)
            counts[r["behavior"]] += 1
    print("\nSelected:")
    for b in BEHAVIORS:
        print(f"  {b:12s}: {counts[b]}/{per_class}")
    missing = [b for b in BEHAVIORS if counts[b] < per_class]
    if missing:
        raise RuntimeError(
            "KABR pilot could not find enough examples for: "
            + ", ".join(missing)
        )
    return selected


def download_selected(records, output_dir, workers):
    output_dir.mkdir(parents=True, exist_ok=True)

    def one(record):
        video = hf_hub_download(
            repo_id=REPO_ID, filename=record["video"],
            repo_type=REPO_TYPE, local_dir=output_dir
        )
        xml = hf_hub_download(
            repo_id=REPO_ID, filename=record["annotation"],
            repo_type=REPO_TYPE, local_dir=output_dir
        )
        return record, video, xml

    rows = []
    with cf.ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = [executor.submit(one, r) for r in records]
        for i, f in enumerate(cf.as_completed(futures), 1):
            record, video, xml = f.result()
            rows.append({
                "video": str(Path(video).relative_to(output_dir)),
                "annotation": str(Path(xml).relative_to(output_dir)),
                "label": record["behavior"],
                "source_video": record["video"],
                "source_annotation": record["annotation"],
            })
            print(f"  downloaded {i}/{len(futures)}: {record['behavior']}")

    manifest = output_dir / "kabr_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "annotation", "label",
                        "source_video", "source_annotation"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["label"], x["video"])))
    return manifest


def main():
    p = argparse.ArgumentParser(
        description="Download a tiny balanced KABR mini-scene subset."
    )
    p.add_argument("--per-class", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", default="data/raw/kabr")
    args = p.parse_args()

    output = Path(args.output)
    print("[1/3] Discovering KABR mini-scene pairs...")
    pairs = discover_pairs()
    print(f"Discovered {len(pairs)} candidate pairs.")

    print(f"[2/3] Concurrent annotation scan ({args.workers} workers)...")
    records = scan_annotations(
        pairs, output / "_annotation_cache",
        args.workers, args.per_class
    )
    selected = select_balanced(records, args.per_class)

    print(f"\n[3/3] Downloading {len(selected)} selected MP4/XML pairs...")
    manifest = download_selected(selected, output, args.workers)

    print("\nKABR_DOWNLOAD_OK")
    print(f"Selected clips: {len(selected)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
