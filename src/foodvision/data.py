"""Dataset discovery, validation, splitting and the PyTorch Dataset."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


def find_image_root(base: str | Path) -> Path:
    """Return the directory with the most class sub-folders that contain images.

    Kaggle mounts datasets at paths that change between versions, so we search
    rather than hard-code ``.../Food Classification``.
    """
    base = Path(base)
    best, best_n = None, 0
    for d in [base, *(p for p in base.rglob("*") if p.is_dir())]:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        n = sum(any(f.suffix.lower() in IMG_EXTS for f in s.iterdir()) for s in subdirs)
        if n > best_n:
            best, best_n = d, n
    if best is None or best_n < 2:
        raise FileNotFoundError(f"No class folders with images found under {base}")
    return best


def is_valid_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def scan_images(image_root: str | Path, max_per_class: int = 0, workers: int = 32):
    """List images as a DataFrame (path, label, target), dropping corrupt files.

    Ordering (sorted classes, sorted files) is deterministic so that
    ``make_splits`` with the same seed reproduces the same split.
    """
    image_root = Path(image_root)
    candidates = []
    for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        if max_per_class:
            files = files[:max_per_class]
        candidates += [(f, class_dir.name) for f in files]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(lambda c: is_valid_image(c[0]), candidates))

    df = pd.DataFrame([{"path": str(f), "label": lbl}
                       for (f, lbl), good in zip(candidates, ok) if good])
    bad = [str(f) for (f, _), good in zip(candidates, ok) if not good]
    class_names = sorted(df["label"].unique())
    df["target"] = df["label"].map({c: i for i, c in enumerate(class_names)})
    return df, class_names, bad


def make_splits(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split (70/15/15 by default)."""
    train, temp = train_test_split(df, test_size=val_size + test_size,
                                   stratify=df["target"], random_state=seed)
    val, test = train_test_split(temp, test_size=test_size / (val_size + test_size),
                                 stratify=temp["target"], random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, image_root: str | Path) -> None:
    """Write splits with paths relative to the image root, so they work on any machine."""
    out_dir, image_root = Path(out_dir), Path(image_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        rel = part["path"].map(lambda p: Path(p).relative_to(image_root).as_posix())
        part.assign(rel_path=rel)[["rel_path", "label", "target"]].to_csv(out_dir / f"{name}.csv", index=False)


def load_splits(splits_dir: str | Path, image_root: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load versioned split CSVs (``rel_path,label,target``) and resolve full paths."""
    splits_dir, image_root = Path(splits_dir), Path(image_root)
    splits = {}
    for name in SPLITS:
        df = pd.read_csv(splits_dir / f"{name}.csv")
        df["path"] = df["rel_path"].map(lambda r: str(image_root / r))
        splits[name] = df
    pairs = pd.concat(splits.values())[["target", "label"]].drop_duplicates().sort_values("target")
    class_names = pairs["label"].tolist()
    if pairs["target"].tolist() != list(range(len(class_names))):
        raise ValueError("Split CSVs have inconsistent label/target mapping")
    missing = [p for p in splits["train"]["path"].head(20) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Split paths not found under {image_root}, e.g. {missing[0]}")
    return splits, class_names


class FoodDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tfms):
        self.paths = frame["path"].tolist()
        self.targets = frame["target"].tolist()
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # handles PNG alpha / grayscale
        return self.tfms(img), self.targets[i]
