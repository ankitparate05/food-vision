"""Load exported artifacts and run inference. Used by the API and the CLI.

Artifacts directory layout (exported by training):
    model.pt            state_dict of the best checkpoint
    class_names.json    index -> dish name
    model_config.json   architecture + preprocessing settings
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageFile

from .model import build_model
from .transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Predictor:
    def __init__(self, artifacts_dir: str | Path, device: str = "cpu"):
        artifacts_dir = Path(artifacts_dir)
        self.config = json.loads((artifacts_dir / "model_config.json").read_text())
        self.class_names: list[str] = json.loads((artifacts_dir / "class_names.json").read_text())
        if len(self.class_names) != self.config["num_classes"]:
            raise ValueError("class_names.json does not match num_classes in model_config.json")

        self.device = torch.device(device)
        self.model = build_model(self.config["model_name"], len(self.class_names), pretrained=False)
        state = torch.load(artifacts_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.tfms = eval_transforms(
            img_size=self.config["img_size"],
            resize=self.config.get("resize"),
            mean=self.config.get("mean", IMAGENET_MEAN),
            std=self.config.get("std", IMAGENET_STD),
        )

    @staticmethod
    def _to_image(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        return Image.open(image).convert("RGB")

    @torch.inference_mode()
    def predict(self, image, top_k: int = 3) -> list[dict]:
        """Return the top-k dishes as ``[{"dish": str, "confidence": float}, ...]``."""
        x = self.tfms(self._to_image(image)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu()
        conf, idx = probs.topk(min(top_k, len(self.class_names)))
        return [{"dish": self.class_names[i], "confidence": round(float(c), 4)}
                for c, i in zip(conf.tolist(), idx.tolist())]
