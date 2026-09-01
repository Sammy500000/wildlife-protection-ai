from __future__ import annotations

import argparse
from pathlib import Path

from ml.behavior.inference import BehaviorInference

p = argparse.ArgumentParser(description="Smoke-test a trained behaviour checkpoint on image sequences.")
p.add_argument("--checkpoint", default="data/models/behavior/resnet18_lstm.pt")
p.add_argument("--frames", nargs="+", required=True)
a = p.parse_args()

result = BehaviorInference(a.checkpoint).predict([Path(x) for x in a.frames])
print(result)
print("BEHAVIOR_INFERENCE_OK")
