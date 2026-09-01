from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "imageomics/KABR"
REPO_TYPE = "dataset"
ANNOTATION_ROOT = "KABR/annotation"
IMAGE_ROOT = "KABR/dataset/image"

DEFAULT_CLASSES = {
    "Walk": 0,
    "Graze": 1,
    "Browse": 2,
    "Head Up": 3,
    "Auto-Groom": 4,
    "Trot": 5,
    "Run": 6,
    "Occluded": 7,
}


def download_annotation(filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{ANNOTATION_ROOT}/{filename}",
            repo_type=REPO_TYPE,
            local_dir=cache_dir,
        )
    )


def read_annotation(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        df = pd.read_csv(path)
    required = {"original_vido_id", "video_id", "frame_id", "path", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Unexpected KABR annotation format. Missing {sorted(missing)}; "
            f"found {list(df.columns)}"
        )
    return df


def select_sequences(df, classes, per_class, sequence_length, seed):
    rng = random.Random(seed)
    id_to_label = {int(v): k for k, v in classes.items()}
    df = df.copy()
    df["label_id"] = pd.to_numeric(df["labels"], errors="coerce")
    df = df[df["label_id"].isin(id_to_label)].copy()

    candidates = []
    for (source_id, video_id), group in df.groupby(
        ["original_vido_id", "video_id"], sort=True
    ):
        group = group.sort_values("frame_id")
        counts = Counter(group["label_id"].astype(int).tolist())
        if not counts:
            continue
        dominant_id, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / len(group)
        if len(group) < sequence_length or purity < 0.90:
            continue
        candidates.append(
            {
                "source_id": str(source_id),
                "video_id": int(video_id),
                "label": id_to_label[dominant_id],
                "label_id": int(dominant_id),
                "purity": round(purity, 4),
                "frames": group,
            }
        )

    by_label = {name: [] for name in classes}
    for item in candidates:
        by_label[item["label"]].append(item)

    selected = []
    for label in classes:
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        for item in pool[:per_class]:
            frames = item["frames"]
            indices = [
                round(i * (len(frames) - 1) / (sequence_length - 1))
                for i in range(sequence_length)
            ]
            sampled = frames.iloc[indices]
            selected.append(
                {
                    "sequence_id": f"{item['source_id']}_{item['video_id']}",
                    "source_id": item["source_id"],
                    "video_id": item["video_id"],
                    "label": item["label"],
                    "label_id": item["label_id"],
                    "purity": item["purity"],
                    "frames": sampled["path"].tolist(),
                }
            )
    return selected


def download_sequence_images(items, output_dir):
    downloaded = []
    for seq in items:
        seq_dir = output_dir / "frames" / seq["sequence_id"]
        seq_dir.mkdir(parents=True, exist_ok=True)
        local_frames = []
        for n, dataset_path in enumerate(seq["frames"]):
            target = seq_dir / f"{n:03d}.jpg"
            if not target.exists():
                source = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{IMAGE_ROOT}/{dataset_path}",
                    repo_type=REPO_TYPE,
                    local_dir=output_dir / "_hf_cache",
                )
                target.write_bytes(Path(source).read_bytes())
            local_frames.append(str(target.resolve()))
        downloaded.append({**seq, "frames": local_frames})
        print(f"  downloaded {seq['label']}: {seq['sequence_id']}")
    return downloaded


def write_manifest(items, classes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "KABR",
        "source": REPO_ID,
        "sequence_length": len(items[0]["frames"]) if items else 0,
        "classes": classes,
        "items": items,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8"
    )
    summary = {
        "dataset": "KABR",
        "sequences": len(items),
        "frames": sum(len(x["frames"]) for x in items),
        "class_counts": dict(Counter(x["label"] for x in items)),
        "classes": classes,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Download a tiny balanced KABR subset using annotations and selected frames only."
    )
    parser.add_argument("--sequences-per-class", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--output", default="data/raw/kabr")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    annotations = output / "_annotations"

    print("[1/4] Downloading small KABR annotation files...")
    train_path = download_annotation("train.csv", annotations)
    val_path = download_annotation("val.csv", annotations)
    try:
        classes_path = download_annotation("classes.json", annotations)
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
    except Exception:
        classes = DEFAULT_CLASSES

    print("[2/4] Reading frame annotations...")
    train = read_annotation(train_path)
    val = read_annotation(val_path)
    print(f"  train rows: {len(train):,}")
    print(f"  val rows:   {len(val):,}")
    print(f"  classes: {classes}")

    print("[3/4] Selecting balanced temporal sequences...")
    train_items = select_sequences(
        train, classes, args.sequences_per_class, args.sequence_length, args.seed
    )
    val_items = select_sequences(
        val, classes, max(1, args.sequences_per_class // 2),
        args.sequence_length, args.seed + 1
    )
    print(f"  selected train: {len(train_items)}")
    print(f"  selected val:   {len(val_items)}")

    if not train_items:
        raise RuntimeError("No suitable KABR training sequences were found.")

    print("[4/4] Downloading only selected image frames...")
    train_local = download_sequence_images(train_items, output / "train")
    val_local = download_sequence_images(val_items, output / "val")

    train_summary = write_manifest(train_local, classes, output / "train")
    val_summary = write_manifest(val_local, classes, output / "val")

    combined = {
        "dataset": "KABR",
        "classes": classes,
        "train": str((output / "train" / "manifest.json").resolve()),
        "val": str((output / "val" / "manifest.json").resolve()),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    print("\nKABR_DOWNLOAD_OK")
    print(json.dumps({"train": train_summary, "val": val_summary}, indent=2))


if __name__ == "__main__":
    main()
