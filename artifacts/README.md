# artifacts/

Model files are **not stored in git**. Put the training outputs here before running the API or building the Docker image:

```
artifacts/
├── model.pt            # trained weights (state_dict)
├── class_names.json    # index → dish name
└── model_config.json   # architecture + preprocessing settings
```

Get them from the Kaggle notebook run: notebook page → **Output** tab → `artifacts/`.

To run the API without a trained model (e.g. CI), create random-weight placeholders:

```bash
python scripts/make_dummy_artifacts.py artifacts
```
