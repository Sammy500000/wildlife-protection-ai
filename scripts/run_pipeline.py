from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.pipeline.integrated import run_integrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildlife end-to-end surveillance pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/pipeline_test"))
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--species-samples", type=int, default=8)
    parser.add_argument("--behavior-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    if args.behavior_checkpoint and not args.behavior_checkpoint.exists():
        raise SystemExit(f"Behaviour checkpoint not found: {args.behavior_checkpoint}")

    result = run_integrated(
        args.input,
        args.output_dir,
        args.sample_every,
        args.species_samples,
        args.behavior_checkpoint,
    )
    print(json.dumps(result, indent=2))
    print("INTEGRATED_PIPELINE_OK")


if __name__ == "__main__":
    main()
