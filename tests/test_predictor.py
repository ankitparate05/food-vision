import json

import pytest
import torch

from conftest import CLASSES, image_bytes, make_image
from foodvision.model import build_model
from foodvision.predictor import Predictor


@pytest.fixture(scope="module")
def predictor(artifacts_dir):
    return Predictor(artifacts_dir)


def test_returns_sorted_top_k(predictor):
    preds = predictor.predict(make_image(), top_k=3)
    assert len(preds) == 3
    confs = [p["confidence"] for p in preds]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert sum(confs) == pytest.approx(1.0, abs=1e-3)  # 3 classes -> top-3 covers everything
    assert {p["dish"] for p in preds} == set(CLASSES)


def test_top_k_is_clamped_to_number_of_classes(predictor):
    assert len(predictor.predict(make_image(), top_k=10)) == len(CLASSES)


@pytest.mark.parametrize("source", ["pil", "bytes", "path"])
def test_accepts_pil_bytes_and_path(predictor, tmp_path, source):
    img = make_image(seed=3)
    if source == "pil":
        x = img
    elif source == "bytes":
        x = image_bytes(seed=3, fmt="PNG")
    else:
        x = tmp_path / "x.png"
        img.save(x)
    assert len(predictor.predict(x)) == 3


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_handles_grayscale_and_alpha(predictor, mode):
    assert len(predictor.predict(make_image(mode=mode))) == 3


def test_deterministic(predictor):
    img = make_image(seed=7)
    assert predictor.predict(img) == predictor.predict(img)


def test_preprocessing_follows_model_config(predictor):
    cfg = predictor.config
    assert cfg["resize"] == 255 and cfg["img_size"] == 224
    x = predictor.tfms(make_image(w=500, h=300))
    assert tuple(x.shape) == (3, 224, 224)


def test_mismatched_class_names_rejected(artifacts_dir, tmp_path):
    for f in ("model.pt", "model_config.json"):
        (tmp_path / f).write_bytes((artifacts_dir / f).read_bytes())
    (tmp_path / "class_names.json").write_text(json.dumps(["only_one"]))
    with pytest.raises(ValueError):
        Predictor(tmp_path)


@pytest.mark.parametrize("name", ["efficientnet_b0", "convnext_tiny"])
def test_models_have_backbone_and_head(name):
    m = build_model(name, num_classes=5, pretrained=False).eval()
    assert hasattr(m, "features") and hasattr(m, "classifier")
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 64, 64)).shape == (1, 5)
