import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from check_duplicates import dhash_bits, find_pairs, plan_exclusions
from conftest import make_image  # noqa: F401
from foodvision.data import load_splits, make_splits
from foodvision.nutrition import load_table, lookup

ROOT = Path(__file__).resolve().parents[1]
REAL_CLASSES = json.loads((ROOT / "reports/baseline/class_names.json").read_text())


# ---- nutrition --------------------------------------------------------------
def test_every_trained_class_has_nutrition():
    missing = [c for c in REAL_CLASSES if lookup(c) is None]
    assert not missing, f"No nutrition entry for: {missing}"


def test_nutrition_values_are_sane():
    for dish, info in load_table().items():
        assert 0 < info["kcal"] < 1500, dish
        assert all(info[k] >= 0 for k in ("protein_g", "carbs_g", "fat_g")), dish
        # macros should roughly explain the calories (4/4/9 kcal per gram), within 35%
        est = 4 * info["protein_g"] + 4 * info["carbs_g"] + 9 * info["fat_g"]
        assert abs(est - info["kcal"]) / info["kcal"] < 0.35, (dish, est, info["kcal"])


# ---- versioned splits ---------------------------------------------------------
def test_committed_splits_are_disjoint_and_stratified():
    splits = {s: pd.read_csv(ROOT / "data/splits" / f"{s}.csv") for s in ("train", "val", "test")}
    sets = {s: set(df.rel_path) for s, df in splits.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sum(len(v) for v in sets.values()) == 6269
    for name, df in splits.items():
        assert sorted(df.label.unique()) == REAL_CLASSES, name
    # each class keeps roughly the same share in every split
    share = {s: df.label.value_counts(normalize=True) for s, df in splits.items()}
    assert (share["train"] - share["test"]).abs().max() < 0.01


def test_make_splits_is_reproducible():
    df = pd.DataFrame({"path": [f"img{i}.jpg" for i in range(200)],
                       "label": [f"c{i % 4}" for i in range(200)], "target": [i % 4 for i in range(200)]})
    a, b = make_splits(df, seed=42), make_splits(df, seed=42)
    for k in a:
        assert a[k]["path"].tolist() == b[k]["path"].tolist()
    assert [len(a[k]) for k in ("train", "val", "test")] == [140, 30, 30]


def test_load_splits_resolves_paths(tmp_path):
    root = tmp_path / "Food Classification"
    for c in ("a", "b"):
        (root / c).mkdir(parents=True)
        make_image().save(root / c / "1.jpg")
    sd = tmp_path / "splits"
    sd.mkdir()
    rows = pd.DataFrame({"rel_path": ["a/1.jpg", "b/1.jpg"], "label": ["a", "b"], "target": [0, 1]})
    for s in ("train", "val", "test"):
        rows.to_csv(sd / f"{s}.csv", index=False)
    splits, classes = load_splits(sd, root)
    assert classes == ["a", "b"]
    assert Path(splits["train"]["path"][0]).exists()


# ---- duplicate detection ------------------------------------------------------
def photo_like(seed, w=320, h=240):
    """Large coloured shapes: closer to a real photo's structure than pixel noise."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), tuple(int(v) for v in rng.integers(0, 255, 3)))
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        r = int(rng.integers(30, 120))
        draw.ellipse([x, y, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return img


def recompress(img, quality=40):
    import io
    from PIL import Image
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue()))


@pytest.mark.parametrize("seed", range(5))
def test_dhash_matches_edited_copy_but_not_other_image(seed):
    img = photo_like(seed)
    h = dhash_bits(img)
    assert (h != dhash_bits(img.resize((160, 120)))).sum() <= 4
    assert (h != dhash_bits(recompress(img))).sum() <= 4
    assert (h != dhash_bits(photo_like(seed + 100))).sum() > 10


def test_find_pairs_and_exclusions():
    a, b = photo_like(1), photo_like(2)
    bits = np.stack([dhash_bits(x) for x in (a, recompress(a.resize((160, 120))), b)])
    assert [(i, j) for i, j, _ in find_pairs(bits, threshold=4)] == [(0, 1)]

    pairs = pd.DataFrame([
        {"rel_path_a": "x/train.jpg", "split_a": "train", "rel_path_b": "x/test.jpg", "split_b": "test"},
        {"rel_path_a": "y/val.jpg", "split_a": "val", "rel_path_b": "y/train.jpg", "split_b": "train"},
        {"rel_path_a": "z/val.jpg", "split_a": "val", "rel_path_b": "z/test.jpg", "split_b": "test"},
        {"rel_path_a": "w/t1.jpg", "split_a": "train", "rel_path_b": "w/t2.jpg", "split_b": "train"},
    ])
    assert plan_exclusions(pairs).rel_path.tolist() == ["x/test.jpg", "y/val.jpg", "z/test.jpg"]


@pytest.mark.parametrize("aug", ["basic", "strong"])
def test_train_transforms_output_shape(aug):
    from foodvision.transforms import train_transforms
    assert tuple(train_transforms(224, aug)(make_image()).shape) == (3, 224, 224)
