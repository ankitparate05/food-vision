import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_dummy_artifacts import make_dummy_artifacts  # noqa: E402

CLASSES = ["dal_makhani", "pav_bhaji", "samosa"]


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    return make_dummy_artifacts(tmp_path_factory.mktemp("artifacts"), classes=CLASSES)


def make_image(w=320, h=240, seed=0, mode="RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))
    return img.convert(mode)


def image_bytes(fmt="JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    img = make_image(**kw)
    img.save(buf, format=fmt)
    return buf.getvalue()
