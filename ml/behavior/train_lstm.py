from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASSES = ["RESTING", "NORMAL_MOVEMENT", "RUNNING", "CHASING", "AGGRESSIVE_ABNORMAL", "UNKNOWN"]


class SequenceDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int = 224):
        self.items = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("items", [])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.labels = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = []
        for path in item["frames"]:
            with Image.open(path).convert("RGB") as im:
                frames.append(self.transform(im))
        x = torch.stack(frames)
        y = self.labels.get(item["label"], self.labels["UNKNOWN"])
        return x, y


class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=len(CLASSES), hidden_size: int = 128, layers: int = 1):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.lstm = nn.LSTM(self.feature_dim, hidden_size, layers, batch_first=True,
                            dropout=0.0 if layers == 1 else 0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = self.encoder(x.reshape(b * t, c, h, w)).flatten(1)
        seq = feats.reshape(b, t, -1)
        out, _ = self.lstm(seq)
        return self.classifier(out[:, -1])


def _run_epoch(model, loader, loss_fn, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    ys, preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        count += y.size(0)
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
    return total_loss / max(count, 1), correct / max(count, 1), ys, preds


def macro_f1(y_true, y_pred, n):
    scores = []
    for k in range(n):
        tp = sum(a == k and b == k for a, b in zip(y_true, y_pred))
        fp = sum(a != k and b == k for a, b in zip(y_true, y_pred))
        fn = sum(a == k and b != k for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(scores) / n


def main(train_manifest, val_manifest, output, epochs=5, batch_size=2, lr=1e-4, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cpu")
    train_ds, val_ds = SequenceDataset(train_manifest), SequenceDataset(val_manifest)
    if not train_ds or not val_ds:
        raise RuntimeError("Train/validation manifests must contain items.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ResNetLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        tl, ta, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        with torch.no_grad():
            vl, va, vy, vp = _run_epoch(model, val_loader, loss_fn, None, device)
        vf1 = macro_f1(vy, vp, len(CLASSES))
        row = {"epoch": epoch, "train_loss": tl, "train_accuracy": ta,
               "val_loss": vl, "val_accuracy": va, "val_macro_f1": vf1}
        history.append(row)
        print(row)
        if vf1 > best_f1:
            best_f1 = vf1
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "model": "resnet18_lstm", "seed": seed, "best_val_macro_f1": best_f1}, output)
    Path(str(output) + ".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"BEHAVIOR_TRAIN_OK: best_val_macro_f1={best_f1:.4f} checkpoint={output}")
