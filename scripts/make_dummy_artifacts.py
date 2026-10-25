"""Create random-weight artifacts with the same format as a real training run.

Used by tests and CI so the API and Docker image can be exercised without
the real (gitignored) model weights. Predictions from these are meaningless.

    python scripts/make_dummy_artifacts.py artifacts [--model efficientnet_b0] [--classes a b c]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodvision.model import SUPPORTED_MODELS, build_model  # noqa: E402
from foodvision.transforms import IMAGENET_MEAN, IMAGENET_STD, default_resize  # noqa: E402

DEFAULT_CLASSES = ["dal_makhani", "pav_bhaji", "samosa"]


def make_dummy_artifacts(out_dir, model_name="efficientnet_b0", classes=None, img_size=224):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(classes or DEFAULT_CLASSES)
    torch.manual_seed(0)
    model = build_model(model_name, len(classes), pretrained=False)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "class_names.json").write_text(json.dumps(classes, indent=2))
    (out_dir / "model_config.json").write_text(json.dumps({
        "model_name": model_name, "num_classes": len(classes), "img_size": img_size,
        "resize": default_resize(img_size), "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    }, indent=2))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="efficientnet_b0", choices=SUPPORTED_MODELS)
    ap.add_argument("--classes", nargs="+", default=None)
    args = ap.parse_args()
    if (Path(args.out_dir) / "model.pt").exists():
        sys.exit(f"Refusing to overwrite existing {args.out_dir}/model.pt")
    print("Wrote dummy artifacts to", make_dummy_artifacts(args.out_dir, args.model, args.classes))
