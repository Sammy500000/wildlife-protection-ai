from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.behavior.train_lstm import ResNetLSTM, CLASSES


class BehaviorInference:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Behaviour checkpoint not found: {self.checkpoint_path}")
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.classes = payload.get("classes", CLASSES)
        self.model = ResNetLSTM(num_classes=len(self.classes))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, frame_paths: list[str | Path]) -> dict:
        if not frame_paths:
            return {"behaviour": "UNKNOWN", "confidence": 0.0, "frames": 0}
        tensors = []
        for path in frame_paths:
            with Image.open(path).convert("RGB") as image:
                tensors.append(self.transform(image))
        x = torch.stack(tensors).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probabilities.argmax())
        return {
            "behaviour": self.classes[idx],
            "confidence": float(probabilities[idx]),
            "frames": len(frame_paths),
            "model_version": self.checkpoint_path.name,
        }
