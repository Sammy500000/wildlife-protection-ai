from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.pipeline.integrated import run_integrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildlife end-to-end detection, tracking and species pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/pipeline_test"))
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--species-samples", type=int, default=8)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    result = run_integrated(args.input, args.output_dir, args.sample_every, args.species_samples)
    print(json.dumps(result, indent=2))
    print("INTEGRATED_PIPELINE_OK")


if __name__ == "__main__":
    main()
