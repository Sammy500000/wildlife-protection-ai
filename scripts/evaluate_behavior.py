from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from ml.behavior.train_lstm import CLASSES, ResNetLSTM, SequenceDataset, _run_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output", default="data/outputs/behavior/evaluation.json")
    a = p.parse_args()

    ds = SequenceDataset(a.manifest)
    if not ds:
        raise RuntimeError("Evaluation manifest is empty.")
    payload = torch.load(a.checkpoint, map_location="cpu")
    classes = payload.get("classes", CLASSES)
    model = ResNetLSTM(num_classes=len(classes))
    model.load_state_dict(payload["state_dict"])
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    _, accuracy, y_true, y_pred = _run_epoch(
        model, loader, torch.nn.CrossEntropyLoss(), None, torch.device("cpu")
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0
    )
    result = {
        "model": str(Path(a.checkpoint)),
        "manifest": str(Path(a.manifest)),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes)))
        ).tolist(),
        "classes": classes,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": accuracy, "macro_f1": result["macro_f1"]}, indent=2))
    print(f"BEHAVIOR_EVALUATION_OK: {out}")


if __name__ == "__main__":
    main()
