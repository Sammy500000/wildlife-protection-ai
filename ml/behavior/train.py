from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from ml.behavior.model import ResNetLSTM


class SequenceDataset(Dataset):
    def __init__(self, manifest: str, class_names: list[str]):
        self.rows = json.loads(Path(manifest).read_text())
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frames = [self.transform(Image.open(p).convert("RGB")) for p in row["frame_paths"]]
        return torch.stack(frames), self.class_to_id[row["label"]]


def train(manifest: str, classes: list[str], epochs: int, output: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SequenceDataset(manifest, classes)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    model = ResNetLSTM(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step(); total += loss.item()
        print(f"epoch={epoch+1} loss={total/max(1,len(loader)):.4f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": classes}, output)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--classes",nargs="+",default=["RESTING","NORMAL_MOVEMENT","RUNNING","UNKNOWN"]); p.add_argument("--epochs",type=int,default=5); p.add_argument("--output",default="models/behavior/resnet18_lstm.pt")
    a=p.parse_args(); train(a.manifest,a.classes,a.epochs,a.output)
