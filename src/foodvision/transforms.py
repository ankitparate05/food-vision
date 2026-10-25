"""Image transforms shared by training, evaluation and the API.

The eval transform is the single source of truth for inference preprocessing:
the API builds it from ``model_config.json`` so serving matches training exactly.
"""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_resize(img_size: int) -> int:
    """Resize-before-center-crop size (255 for 224, matching the baseline run)."""
    return int(img_size * 1.14)


def train_transforms(img_size: int = 224, aug: str = "basic") -> transforms.Compose:
    """Training augmentation.

    basic  : random crop, flip, small rotation, colour jitter (the baseline).
    strong : basic + TrivialAugmentWide + RandomErasing, aimed at the
             train/val gap seen in the baseline (99% train vs 91% val).
    """
    if aug not in {"basic", "strong"}:
        raise ValueError(f"aug must be 'basic' or 'strong', got {aug!r}")

    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == "strong":
        ops.append(transforms.TrivialAugmentWide())
    else:
        ops += [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug == "strong":
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)


def eval_transforms(img_size: int = 224, resize: int | None = None,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    """Deterministic preprocessing for validation, test and serving."""
    return transforms.Compose([
        transforms.Resize(resize or default_resize(img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
