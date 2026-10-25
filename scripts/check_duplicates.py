"""Find duplicate / near-duplicate images across train, val and test splits.

Web-scraped datasets often contain the same photo more than once. If a copy
lands in train and another in test, the test score is inflated (data leakage).

Method: 64-bit difference hash (dHash) per image; two images whose hashes differ
in <= --threshold bits are treated as near-duplicates (resized, re-compressed or
lightly edited copies).

    python scripts/check_duplicates.py --data-dir /kaggle/input --splits-dir data/splits --out reports/duplicates

Writes:
    pairs.csv          every near-duplicate pair with its splits and distance
    exclude.csv        val/test images to drop (pass to train.py --exclude-csv)
    summary.json       counts per split pair
    examples.png       side-by-side preview of leaking pairs
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodvision.data import SPLITS, find_image_root  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dhash_bits(path_or_image, size: int = 8) -> np.ndarray:
    """64 booleans: is each pixel brighter than its right neighbour (9x8 grayscale)."""
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    px = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances for an (n, 64) boolean matrix, via two matmuls."""
    b = bits.astype(np.float32)
    return (b @ (1 - b).T + (1 - b) @ b.T).astype(np.int16)


def find_pairs(bits: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    d = hamming_matrix(bits)
    i, j = np.where(np.triu(d <= threshold, k=1))
    return [(int(a), int(b), int(d[a, b])) for a, b in zip(i, j)]


def plan_exclusions(pairs: pd.DataFrame) -> pd.DataFrame:
    """For each leaking pair keep the train copy and drop the evaluation copy.

    train-val / train-test -> drop the val/test image
    val-test               -> drop the test image
    """
    rank = {"train": 0, "val": 1, "test": 2}
    drop = []
    for r in pairs.itertuples():
        if r.split_a == r.split_b:
            continue  # within-split duplicates don't leak
        drop.append(r.rel_path_a if rank[r.split_a] > rank[r.split_b] else r.rel_path_b)
    return pd.DataFrame({"rel_path": sorted(set(drop))})


def save_examples(pairs: pd.DataFrame, image_root: Path, out_png: Path, n: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leak = pairs[pairs.split_a != pairs.split_b].head(n)
    if leak.empty:
        return
    fig, axes = plt.subplots(len(leak), 2, figsize=(6, 3 * len(leak)), squeeze=False)
    for row, r in zip(axes, leak.itertuples()):
        for ax, rel, split in ((row[0], r.rel_path_a, r.split_a), (row[1], r.rel_path_b, r.split_b)):
            ax.imshow(Image.open(image_root / rel).convert("RGB"))
            ax.set_title(f"{split}: {rel}\n(distance {r.distance})", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detect duplicate images across dataset splits.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="reports/duplicates")
    ap.add_argument("--threshold", type=int, default=4, help="Max differing bits (of 64) to count as duplicate")
    args = ap.parse_args(argv)

    image_root = find_image_root(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(Path(args.splits_dir) / f"{s}.csv").assign(split=s) for s in SPLITS]
    df = pd.concat(frames, ignore_index=True)
    with ThreadPoolExecutor(max_workers=32) as ex:
        bits = np.stack(list(ex.map(lambda r: dhash_bits(image_root / r), df["rel_path"])))
    print(f"Hashed {len(df)} images")

    rows = []
    for a, b, dist in find_pairs(bits, args.threshold):
        rows.append({"rel_path_a": df.rel_path[a], "split_a": df.split[a], "label_a": df.label[a],
                     "rel_path_b": df.rel_path[b], "split_b": df.split[b], "label_b": df.label[b],
                     "distance": dist})
    pairs = pd.DataFrame(rows, columns=["rel_path_a", "split_a", "label_a",
                                        "rel_path_b", "split_b", "label_b", "distance"])
    pairs.to_csv(out / "pairs.csv", index=False)

    exclude = plan_exclusions(pairs)
    exclude.to_csv(out / "exclude.csv", index=False)

    key = pairs.apply(lambda r: "-".join(sorted([r.split_a, r.split_b])), axis=1) if len(pairs) else pd.Series(dtype=str)
    summary = {
        "images": int(len(df)),
        "threshold_bits": args.threshold,
        "duplicate_pairs": int(len(pairs)),
        "pairs_by_split": {k: int(v) for k, v in key.value_counts().items()},
        "label_conflicts": int((pairs.label_a != pairs.label_b).sum()) if len(pairs) else 0,
        "val_test_images_to_exclude": int(len(exclude)),
        "test_images_leaking": int(exclude.rel_path.isin(frames[2].rel_path).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_examples(pairs, image_root, out / "examples.png")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
