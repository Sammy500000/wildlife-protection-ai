from __future__ import annotations

import argparse

from ml.behavior.animal_kingdom import AnimalKingdomAdapter


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare an internal manifest from Animal Kingdom action annotations.")
    p.add_argument("--annotations", required=True, help="Path to the Animal Kingdom action-recognition CSV.")
    p.add_argument("--video-root", default=None, help="Optional root containing the dataset videos.")
    p.add_argument("--output", default="data/processed/animal_kingdom_manifest.json")
    args = p.parse_args()

    n = AnimalKingdomAdapter(args.annotations, args.video_root).to_manifest(args.output)
    print(f"ANIMAL_KINGDOM_MANIFEST_OK: {n} annotation rows -> {args.output}")


if __name__ == "__main__":
    main()
