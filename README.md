# 🍛 Food Vision: Indian Dish Classifier with an MLOps Pipeline

[![CI](https://github.com/ankitparate05/food-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitparate05/food-vision/actions/workflows/ci.yml) [![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Ankitparate/food-vision)

**▶ Try it live:** [huggingface.co/spaces/Ankitparate/food-vision](https://huggingface.co/spaces/Ankitparate/food-vision): upload a food photo, get the dish and its nutrition.

Upload a photo of an Indian dish → get the **top-3 predicted dishes** with confidence and **approximate nutrition**.
The best model, a fine-tuned **ConvNeXt-Tiny**, recognises **20 dishes** with **94.3% top-1 / 98.8% top-3 accuracy** on a de-duplicated test set. Served by a **FastAPI** service in **Docker**, tested in **GitHub Actions CI**.

The focus is not just the model but the engineering around it: versioned data splits, a data-leakage check, experiment tracking, one shared preprocessing path for training and serving, and automated tests.

---

## Results

### Final comparison (clean test set, duplicates removed)

All runs use the same versioned splits, seed and epochs (3 head + 7 fine-tune), tracked in MLflow. The winner is chosen by **validation** accuracy; test is reported once.

| Run | Model | Augmentation | Val acc | Test acc | Test top-3 | Test macro-F1 | Params |
|---|---|---|---|---|---|---|---|
| **C** | **ConvNeXt-Tiny** | strong | **92.2%** | **94.3%** | **98.8%** | **0.945** | 27.8M |
| A | EfficientNet-B0 | basic | 90.4% | 93.3% | 97.8% | 0.935 | 4.0M |
| B | EfficientNet-B0 | strong | 90.5% | 92.5% | 97.9% | 0.925 | 4.0M |

**Findings**
- **ConvNeXt-Tiny wins on every metric** (+1.8 pts validation, +1.0 pt test over the baseline), at 7x the parameters (~110 MB vs 16 MB).
- **Stronger augmentation (TrivialAugment + RandomErasing) did not help EfficientNet-B0**: validation was flat (90.4 to 90.5%). The bottleneck was model capacity, not just overfitting.
- ConvNeXt trained as fast as EfficientNet (~14 min each on a T4) despite its size: training was **data-loading (CPU) bound**, not GPU bound.

Raw numbers: [`reports/experiments/experiments_summary.csv`](reports/experiments/experiments_summary.csv)

### Data-leakage audit

The dataset is web-scraped, so every image was hashed with a 64-bit perceptual hash (dHash); pairs differing in 4 bits or fewer are near-duplicates (resized / re-compressed copies).

| Finding | Count |
|---|---|
| Near-duplicate pairs | 96 |
| Test images with a copy in **train** | 25 |
| Val images with a copy in train | 12 |
| Same photo with **two different labels** | 1 |
| Val/test images removed before the experiments | 39 (27 from test) |

Re-training the baseline on the cleaned splits gave **93.3%** vs **93.5%** before: the leak inflated the original score by only ~0.2 pts, so the baseline result holds up. Details: [`reports/experiments/duplicates_summary.json`](reports/experiments/duplicates_summary.json)

### Step 1 baseline (EfficientNet-B0, before the audit)

Evaluated once on a held-out test set of **941 images** (15% stratified split, never used for training or model selection).

| Metric | Score |
|---|---|
| Top-1 accuracy | **93.5%** |
| Top-3 accuracy | **97.5%** (the app shows 3 suggestions, so this is what users experience) |
| Macro-F1 | **0.935** (every dish weighted equally) |
| Best validation accuracy | 90.7% |
| Model size | 4.0M parameters, 16 MB |
| CPU inference latency | ~70–90 ms per image |

<p align="center">
  <img src="reports/baseline/figures/learning_curves.png" width="85%" alt="Learning curves">
</p>

Two-phase fine-tuning: 3 epochs training only the new classifier head, then 7 epochs unfreezing the whole network. Validation accuracy jumps from 75% to 88% the moment the backbone is unfrozen.

### Error analysis: where the model fails

<p align="center">
  <img src="reports/baseline/figures/confusion_matrix.png" width="70%" alt="Confusion matrix">
</p>

| Weakest classes | F1 | Pattern |
|---|---|---|
| dal_makhani | 0.833 | confused with **pav_bhaji**: both orange-red gravies |
| pav_bhaji | 0.874 | same gravy confusion |
| chapati | 0.885 | confused with **butter_naan**: flatbreads |
| butter_naan | 0.889 | same flatbread confusion |

Errors cluster in **visually similar food groups** (gravies, flatbreads), not at random: the model has learned meaningful visual features, and the remaining errors are ones a person might also make from a single photo.

**Honest caveats**
- Test accuracy (93.5%) is ~3 points above validation (90.7%). With ~940 images per split, part of that is split noise, so the realistic range is **~91–93%**.
- Train accuracy reaches 99% while validation plateaus at ~91%: the model **overfits**. Step 2 tested stronger augmentation and a larger backbone against this (see the comparison above).

---

## Architecture

```mermaid
flowchart LR
    A[Kaggle dataset<br/>6,269 images, 20 dishes] --> B[Versioned splits<br/>data/splits/*.csv]
    B --> C[Leakage check<br/>perceptual hashing]
    C --> D[Training on Kaggle GPU<br/>foodvision.train]
    D -- params, metrics, artifacts --> E[(MLflow)]
    D --> F[artifacts/<br/>model.pt + config]
    F --> G[FastAPI service<br/>/predict /health]
    G --> H[Docker image]
    I[GitHub Actions CI<br/>tests + Docker smoke test] --> H
```

## MLOps practices

| Practice | Where | Why it matters |
|---|---|---|
| **Versioned data splits** | `data/splits/` | Every experiment is compared on exactly the same images |
| **Data-leakage check** | `scripts/check_duplicates.py` | Duplicate photos across train/test would inflate the score |
| **Experiment tracking** | MLflow in `foodvision/train.py` | Every run's parameters, curves and artifacts are recorded and comparable |
| **Model selection on validation only** | `notebooks/02_experiments_kaggle.ipynb` | The test set stays an unbiased final check |
| **One preprocessing path** | `foodvision/transforms.py` + `model_config.json` | The API resizes and normalises exactly as in training (avoids training/serving skew) |
| **Weights out of git** | `.gitignore`, `artifacts/README.md` | Code is versioned in git; large binaries are not |
| **Automated tests** | `tests/` (36 tests) | Predictor, API errors, splits, nutrition table, duplicate detection |
| **CI + container smoke test** | `.github/workflows/ci.yml` | Every push runs the tests, builds the image and calls `/predict` inside it |

---

## Project structure

```
food-vision/
├── src/foodvision/
│   ├── data.py            # dataset discovery, validation, splits
│   ├── transforms.py      # train augmentation + the shared eval/serving transform
│   ├── model.py           # EfficientNet-B0 / ConvNeXt-Tiny factory
│   ├── train.py           # two-phase training CLI with MLflow tracking
│   ├── predictor.py       # loads artifacts, runs inference
│   ├── api.py             # FastAPI app
│   ├── nutrition.json     # approximate per-serving nutrition table
│   └── static/index.html  # upload page
├── scripts/
│   ├── check_duplicates.py
│   └── make_dummy_artifacts.py
├── notebooks/
│   ├── 01_baseline_training.ipynb    # Step 1: baseline (results above)
│   └── 02_experiments_kaggle.ipynb   # Step 2: leakage audit + tracked experiments (self-contained)
├── data/splits/           # versioned train/val/test lists
├── reports/
│   ├── baseline/          # Step 1 metrics, history, figures
│   └── experiments/       # Step 2 comparison + leakage audit results
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Run it

### API locally
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-api.txt && pip install -e .

# put model.pt, class_names.json, model_config.json in ./artifacts (see artifacts/README.md)
uvicorn foodvision.api:app --reload
```
Open http://127.0.0.1:8000 for the upload page, or http://127.0.0.1:8000/docs for the interactive API docs.

```bash
curl -F "file=@my_dosa.jpg" http://127.0.0.1:8000/predict
```

### Docker
```bash
docker build -t food-vision .
docker run -p 7860:7860 food-vision      # http://localhost:7860
```

### Train
```bash
pip install -r requirements.txt && pip install -e .
python -m foodvision.train --data-dir /path/to/dataset --splits-dir data/splits \
    --model efficientnet_b0 --output-dir outputs/run1
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Check for duplicate images
```bash
python scripts/check_duplicates.py --data-dir /path/to/dataset --splits-dir data/splits
```

### Tests
```bash
pip install -r requirements-dev.txt && pytest
```

---

## Roadmap

- [x] **Step 1:** baseline EfficientNet-B0 (93.5% top-1)
- [x] Packaged training code, FastAPI service, Docker image, tests, CI
- [x] **Step 2:** duplicate-leakage audit + MLflow comparison: ConvNeXt-Tiny wins (94.3% top-1 on clean test set)
- [x] **Step 3:** deployed to [Hugging Face Spaces](https://huggingface.co/spaces/Ankitparate/food-vision) (Gradio demo, EfficientNet-B0)

## Limitations

- Knows only **20 dishes**. Anything else gets mapped to the closest of them; the API flags predictions below 50% confidence.
- **Nutrition values are approximate table lookups** for a typical serving, not estimated from the photo. Portion size can't be judged reliably from one image.
- Web-scraped training data: photos are often well-lit and plated, so accuracy on everyday phone photos may be lower.

## Data

[Indian Food Classification](https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification) by Pushkar Jain (Kaggle): 6,269 valid images across 20 classes.

---

**Author:** Ankit Parate
