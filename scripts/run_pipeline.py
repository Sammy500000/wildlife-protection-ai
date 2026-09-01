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
    parser.add_argument("--species-samples", type=int, default=16)
    parser.add_argument("--behavior-checkpoint", type=Path, default=None,
                         help="Legacy ResNet-LSTM checkpoint; retained for compatibility.")
    parser.add_argument("--kabr-python", type=Path, default=None,
                         help="Python executable from the CPU kabr-tools environment (Python 3.10/3.11).")
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
        args.kabr_python,
    )
    print(json.dumps(result, indent=2))
    print("INTEGRATED_PIPELINE_OK")


if __name__ == "__main__":
    main()
