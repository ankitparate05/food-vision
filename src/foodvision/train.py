"""Train a dish classifier with two-phase fine-tuning and MLflow tracking.

Example (Kaggle, GPU):
    python -m foodvision.train --data-dir /kaggle/input --splits-dir data/splits \
        --model convnext_tiny --aug strong --finetune-lr 1e-4 \
        --output-dir /kaggle/working/runs/convnext_strong \
        --tracking-uri sqlite:////kaggle/working/mlflow.db \
        --mlflow-artifacts /kaggle/working/mlartifacts

Outputs (same format the API loads):
    model.pt, class_names.json, model_config.json, metrics.json, history.csv,
    classification_report.txt, confusion_matrix.csv, confusion_matrix.png
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import FoodDataset, find_image_root, load_splits, make_splits, save_splits, scan_images
from .model import SUPPORTED_MODELS, build_model, count_parameters, set_backbone_trainable
from .transforms import IMAGENET_MEAN, IMAGENET_STD, default_resize, eval_transforms, train_transforms


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train a Food Vision classifier.")
    ap.add_argument("--data-dir", required=True, help="Folder containing the class folders (searched recursively)")
    ap.add_argument("--splits-dir", default=None,
                    help="Versioned split CSVs (rel_path,label,target). If omitted, splits are created from the data.")
    ap.add_argument("--exclude-csv", default=None,
                    help="CSV with a rel_path column of images to drop from val/test (e.g. duplicates of train images)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="efficientnet_b0", choices=SUPPORTED_MODELS)
    ap.add_argument("--aug", default="basic", choices=["basic", "strong"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=7)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--max-per-class", type=int, default=0, help="Cap images per class (0 = all); for quick tests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="Random init (tests only)")
    ap.add_argument("--experiment", default="food-vision")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                    help="MLflow tracking URI (SQLite file by default)")
    ap.add_argument("--mlflow-artifacts", default="mlartifacts",
                    help="Folder for MLflow run artifacts (used when the experiment is first created)")
    ap.add_argument("--no-mlflow", action="store_true")
    return ap.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class Tracker:
    """Thin wrapper so training runs identically with or without MLflow."""

    def __init__(self, enabled: bool, tracking_uri: str | None, experiment: str, run_name: str,
                 artifact_location: str | None = None):
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            print("mlflow not installed; continuing without tracking")
            return
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_location)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        self.mlflow = mlflow

    def params(self, d):
        if self.mlflow:
            self.mlflow.log_params(d)

    def metrics(self, d, step=None):
        if self.mlflow:
            self.mlflow.log_metrics(d, step=step)

    def tags(self, d):
        if self.mlflow:
            self.mlflow.set_tags(d)

    def artifacts(self, path):
        if self.mlflow:
            self.mlflow.log_artifacts(str(path), artifact_path="artifacts")

    def end(self):
        if self.mlflow:
            self.mlflow.end_run()


def make_loader(frame, tfms, batch_size, workers, shuffle, device):
    return DataLoader(FoodDataset(frame, tfms), batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, num_workers=workers, pin_memory=device.type == "cuda",
                      timeout=300 if workers > 0 else 0)  # fail loudly instead of hanging forever


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.inference_mode()
def predict_probs(model, loader, device, use_amp):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def score(probs, y_true, prefix):
    y_pred = probs.argmax(1)
    k = min(3, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return {
        f"{prefix}_top1_acc": float((y_pred == y_true).mean()),
        f"{prefix}_top3_acc": float(np.mean([t in row for t, row in zip(y_true, topk)])),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_confusion(y_true, y_pred, class_names, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        n = len(class_names)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.4)))
        im = ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks(range(n), class_names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized confusion matrix (test set)")
        fig.colorbar(im, ax=ax, label="Fraction of true class")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=110)
        plt.close(fig)
    except ImportError:
        pass


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ---- data -------------------------------------------------------------
    image_root = find_image_root(args.data_dir)
    if args.splits_dir:
        splits, class_names = load_splits(args.splits_dir, image_root)
        if args.max_per_class:
            splits = {k: v.groupby("target").head(args.max_per_class).reset_index(drop=True)
                      for k, v in splits.items()}
    else:
        df, class_names, bad = scan_images(image_root, args.max_per_class)
        print(f"Scanned {len(df)} images ({len(bad)} corrupt skipped)")
        splits = make_splits(df, seed=args.seed)
    save_splits({k: v[["path", "label", "target"]] for k, v in splits.items()}, out / "splits", image_root)

    excluded = 0
    if args.exclude_csv:
        drop = set(pd.read_csv(args.exclude_csv)["rel_path"])
        for name in ("val", "test"):
            rel = splits[name]["path"].map(lambda p: Path(p).relative_to(image_root).as_posix())
            before = len(splits[name])
            splits[name] = splits[name][~rel.isin(drop)].reset_index(drop=True)
            excluded += before - len(splits[name])
        print(f"Excluded {excluded} val/test images listed in {args.exclude_csv}")

    sizes = {k: len(v) for k, v in splits.items()}
    print(f"Device: {device} | classes: {len(class_names)} | splits: {sizes}")

    eval_workers = min(args.num_workers, 2)
    ev_tfms = eval_transforms(args.img_size)
    train_loader = make_loader(splits["train"], train_transforms(args.img_size, args.aug),
                               args.batch_size, args.num_workers, True, device)
    val_loader = make_loader(splits["val"], ev_tfms, args.batch_size, eval_workers, False, device)
    test_loader = make_loader(splits["test"], ev_tfms, args.batch_size, eval_workers, False, device)

    # ---- model + tracking --------------------------------------------------
    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    run_name = args.run_name or f"{args.model}-{args.aug}"
    tracker = Tracker(not args.no_mlflow, args.tracking_uri, args.experiment, run_name,
                      artifact_location=Path(args.mlflow_artifacts).resolve().as_uri())
    tracker.params({**{k: v for k, v in vars(args).items() if k not in {"tracking_uri"}},
                    "num_classes": len(class_names), "n_params": count_parameters(model),
                    "n_train": sizes["train"], "n_val": sizes["val"], "n_test": sizes["test"],
                    "excluded_val_test": excluded, "device": str(device)})
    tracker.tags({"git_commit": git_commit() or "unknown", "model": args.model, "aug": args.aug})

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_val, step = [], -1.0, 0
    best_path = out / "model.pt"
    t_start = time.time()

    def train_phase(phase, epochs, lr, params):
        nonlocal best_val, step
        if epochs <= 0:
            return
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(1, epochs + 1):
            step += 1
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, use_amp, opt, scaler)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp)
            lr_now = opt.param_groups[0]["lr"]
            sched.step()
            improved = va_acc > best_val
            if improved:
                best_val = va_acc
                torch.save(model.state_dict(), best_path)
            row = dict(phase=phase, epoch=step, train_loss=tr_loss, train_acc=tr_acc,
                       val_loss=va_loss, val_acc=va_acc, lr=lr_now)
            history.append(row)
            tracker.metrics({k: v for k, v in row.items() if k not in {"phase", "epoch"}}, step=step)
            print(f"[{phase}] ep {ep}/{epochs} | train {tr_loss:.3f}/{tr_acc:.3f} | "
                  f"val {va_loss:.3f}/{va_acc:.3f} | {time.time() - t0:.0f}s"
                  + (" | saved" if improved else ""), flush=True)

    set_backbone_trainable(model, False)
    train_phase("head", args.head_epochs, args.head_lr, model.classifier.parameters())
    set_backbone_trainable(model, True)
    train_phase("finetune", args.finetune_epochs, args.finetune_lr, model.parameters())

    # ---- evaluate best checkpoint -----------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_probs, val_true = predict_probs(model, val_loader, device, use_amp)
    test_probs, test_true = predict_probs(model, test_loader, device, use_amp)
    metrics = {**score(val_probs, val_true, "val"), **score(test_probs, test_true, "test"),
               "best_val_acc": best_val, "train_minutes": (time.time() - t_start) / 60,
               "n_params": count_parameters(model)}
    tracker.metrics(metrics)

    y_pred = test_probs.argmax(1)
    (out / "classification_report.txt").write_text(classification_report(
        test_true, y_pred, labels=range(len(class_names)), target_names=class_names,
        digits=3, zero_division=0))
    save_confusion(test_true, y_pred, class_names, out)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (out / "model_config.json").write_text(json.dumps({
        "model_name": args.model, "num_classes": len(class_names), "img_size": args.img_size,
        "resize": default_resize(args.img_size), "mean": IMAGENET_MEAN, "std": IMAGENET_STD}, indent=2))
    (out / "metrics.json").write_text(json.dumps({**metrics, "args": vars(args)}, indent=2))
    tracker.artifacts(out)
    tracker.end()

    print("\n" + "\n".join(f"{k:>16}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
    print(f"Artifacts saved to {out}")
    return metrics


if __name__ == "__main__":
    main()
