---
title: Kidney Graft Rejection Predictor
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Thesis demonstration of a Bayesian-optimized XGBoost model for kidney graft rejection
---

# Kidney Graft Rejection Predictor

Web application for the undergraduate thesis *Predicting Kidney Graft Rejection from Biopsy
Transcriptomic Data with a Bayesian-Optimized XGBoost Model* (School of Information Technology,
Mapúa University).

Francis Joseph G. Libres · Hanz Ian B. Silva · Kim Miguel P. Sobrepeña

**Research prototype, not for clinical use.**

## What it does

The application serves the final Optuna-tuned XGBoost classifier exactly as it was trained in the
analysis notebook. It performs no training, tuning or feature selection of its own.

* **Demo biopsies** — held-out test biopsies bundled with the application, scored in one click.
* **Score your own file** — upload a GEO series matrix (`.txt`, `.txt.gz`) or a CSV/TSV of expression
  values and score up to 200 biopsies, with a downloadable results table.
* **Marker panel** — the biopsy's expression for the 20 probesets with the highest gain in the final
  model, shown against the training averages for rejection and non-rejection. No SHAP values are used.
* **About the model** — pipeline, held-out test performance, data dictionary and proof-of-run details.

## Input requirements

Expression values must come from the Affymetrix PrimeView platform (GPL15207) and be normalised the
same way as the GSE98320 series matrix, with the same probeset identifiers (for example `11719943_at`).
Either layout is accepted: probesets as rows (GEO layout) or specimens as rows. Probesets that are
absent are treated as missing values, which XGBoost handles natively; the application reports coverage
and warns below 95%.

## Repository layout

```
app.py                 the Gradio application
requirements.txt       runtime dependencies
assets/                model and metadata produced by the export notebook
  final_tuned_xgboost.json
  probesets.txt.gz
  top20_dictionary.csv
  demo_specimens.npz
  deploy_meta.json
DEPLOYMENT_GUIDE.md    step-by-step GitHub and Hugging Face Spaces instructions
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The app listens on `PORT` (7860 by default) and binds `0.0.0.0`, so the same entry point works on
Hugging Face Spaces, on Render (`render.yaml` is included) and locally. `MAX_SAMPLES` caps how many
biopsies one upload may contain; it is 200 by default and 50 on Render's free 512 MB instance.

The application reads everything it needs from `assets/`. Regenerate that folder with
`Export_Deployment_Assets.ipynb` whenever the model is retrained.
