# ModelLens

**Your model predicts. ModelLens investigates.**

ModelLens is a working ML model investigation platform. Upload a trained
scikit-learn-compatible model and an evaluation dataset — ModelLens runs a
real statistical investigation (evaluation metrics, error analysis, slice
discovery, drift detection, confidence/calibration analysis, feature
importance, out-of-distribution detection) and reports the results as
structured, evidence-backed findings instead of a single accuracy number.

Every number in the UI is computed by the backend analysis engine from the
uploaded model and data — nothing is hardcoded.

🔗 **Live demo:** [modellens-1.onrender.com](https://modellens-1.onrender.com)
🔗 **API:** [modellens-12i4.onrender.com](https://modellens-12i4.onrender.com)

> Hosted on Render's free tier — the backend may take 30–60s to wake up on
> the first request after a period of inactivity.

---

## Why it exists

"Accuracy = 94%" tells you almost nothing about whether a model is safe to
ship. A model can be 95%+ accurate and still be nearly useless on the
cases that matter — the included imbalanced-classification sample scores
95.8% accuracy but an F1 of 0.71 and a PR-AUC of 0.51 (near coin-flip) on
the minority class. ModelLens exists to ask the harder question: *where,
and why, does this model actually fail?*

## Features

- **Evaluation** — accuracy, precision/recall/F1 (macro & weighted),
  confusion matrix, per-class metrics, ROC-AUC, PR-AUC for classification;
  MAE, MSE, RMSE, R², residual analysis for regression.
- **Dataset transparency** — full schema, dtypes, missing values, numeric
  summary stats or top categories, target class balance with imbalance
  warnings — computed from the actual uploaded file.
- **Error analysis** — per-example inspection with a plain-language
  explanation of what made each input statistically unusual, plus
  statistically-tested error concentration by feature (z-test / chi-square).
- **Slice discovery** — decision-tree-based subgroup discovery finds
  feature combinations where the model performs significantly worse than
  average, ranked by gap × sample size.
- **Confidence & calibration** — reliability diagrams, Expected
  Calibration Error, and explicit high-confidence-error detection.
- **Drift detection** — PSI, Kolmogorov–Smirnov, Wasserstein distance, and
  chi-square between a reference/training dataset and current data.
- **Out-of-distribution detection** — IsolationForest-based, always
  labeled as *potential*, never certain.
- **Feature importance** — native (impurity/coefficient) or permutation
  importance, whichever the model supports.
- **Report export** — JSON, CSV, and PDF.
- **Text classifiers work today** — any scikit-learn `Pipeline` with its
  own vectorizer (e.g. `TfidfVectorizer` + `LogisticRegression`) is
  investigated through the same upload flow as a tabular model. See the
  included `text_classification` sample case.

## What's honestly *not* supported (yet)

Raw deep-learning checkpoints (`.h5`, `.pt`/`.pth`, `.onnx`, TensorFlow
`SavedModel`) and, by extension, image models (CNNs) or transformer-based
text models. Uploading one returns a specific, clear error rather than a
crash. See [Limitations](#limitations) for what adding real support would
require.

## Tech stack

| | |
|---|---|
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| Backend | Python, FastAPI |
| ML engine | scikit-learn, pandas, numpy, scipy |
| Reports | ReportLab (PDF) |
| Deployment | Render |

## Architecture

```
frontend/        React + Vite + TypeScript + Tailwind — UI
backend/
  app/
    main.py            FastAPI app, all HTTP routes
    storage.py          local-disk file storage + in-memory job tracking
    report.py            JSON / CSV / PDF report builders
    ml_engine/
      adapters.py         model loading + compatibility validation
      evaluator.py         classification & regression metrics
      dataset_profile.py    dataset schema / stats / target balance
      error_analysis.py     error tables + plain-language explanations
      slice_discovery.py    decision-tree-based subgroup discovery
      drift.py               PSI / KS / Wasserstein / chi-square drift tests
      calibration.py          confidence analysis, ECE, reliability curves
      explanation.py           feature importance + OOD detection
      findings.py               templates computed evidence into findings
      orchestrator.py         runs the full pipeline end to end
  tests/             unit + integration tests (pytest)
sample_models/       pre-generatable trained model artifacts (.joblib)
sample_data/         pre-generatable synthetic datasets (.csv)
scripts/             generate_sample_data.py, generate_sample_models.py
```

```
Frontend → API (FastAPI) → Investigation Orchestrator
                                ├─ Evaluator (metrics)
                                ├─ Dataset Profiler (schema, stats, balance)
                                ├─ Error Analyzer (concentration + examples)
                                ├─ Slice Discovery (subgroup partitioning)
                                ├─ Drift Analyzer (PSI/KS/Wasserstein/chi²)
                                ├─ Calibration (ECE, reliability, high-conf errors)
                                ├─ Explanation (importance, OOD)
                                └─ Finding Generator → Investigation Report
```

## Supported models

Any scikit-learn-compatible estimator (or `Pipeline` ending in one) that
implements `.predict()` — `LogisticRegression`, `LinearRegression`,
`RandomForestClassifier/Regressor`, `GradientBoostingClassifier/Regressor`,
SVMs with probabilities, and XGBoost/LightGBM sklearn wrappers. Saved as
`.pkl` or `.joblib`.

## Supported dataset formats

`.csv`, `.xlsx`, `.xls` — for both the primary evaluation dataset and the
optional reference/training dataset.

## Getting started locally

Requires Python 3.11+ and Node 18+.

```bash
git clone https://github.com/<your-username>/modellens.git
cd modellens

# Generate sample models & datasets
python3 scripts/generate_sample_data.py
python3 scripts/generate_sample_models.py

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
cp .env.example .env   # points at http://localhost:8000 by default
npm run dev
```

Open the printed local URL, then either click **Try demo investigation**
on the dashboard, browse **Samples** for a specific case (healthy, weak,
imbalanced, drifted, error-prone, text, or regression), or go to **New
Investigation** to upload your own model and dataset.

## Sample cases

| Case | What it demonstrates |
|---|---|
| `healthy_classification` | A reasonably well-performing baseline |
| `weak_classification` | Low separability, high label noise |
| `imbalanced_classification` | 95/5 imbalance — accuracy is misleading (check PR-AUC) |
| `distribution_shift` | Model trained on a reference period; current data has real, deliberate drift |
| `error_prone` | A linear model applied to data with a non-linear interaction it structurally cannot capture — errors genuinely cluster |
| `text_classification` | Free-text product reviews, zero numeric features — a real TF-IDF + LogisticRegression Pipeline |
| `regression` | A regression model with real residual structure |

## Running tests

```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

23 tests covering metrics, drift, slice discovery, calibration, dataset
profiling, Excel upload, unsupported-format handling, and full integration
tests through the live API (including the text-classification pipeline).

## Limitations

- **Correlation is not causation.** Error-concentration and slice findings
  are observed associations, not proof a feature *causes* failures.
- **Drift does not automatically mean model failure** — cross-reference
  against performance-degradation and slice findings before acting on it.
- **Feature importance is not causal**, especially with correlated
  features.
- **The OOD flag is a statistical anomaly signal**, not proof a sample is
  malicious, mislabeled, or outside the model's valid domain.
- **A significant p-value doesn't guarantee real-world significance** —
  always check effect size and sample size together.
- **Local storage + in-memory job tracking** — fine for local/single-process
  use, not multi-worker production (swap for object storage + Postgres +
  a real job queue for that).
- **No SHAP integration** — local explanations use a labeled,
  importance-weighted heuristic instead, to keep the dependency footprint
  installable everywhere.
- **No image or deep-learning model support** — see [Features](#what-s-honestly-not-supported-yet).
  Adding it needs a separate model adapter, real image preprocessing, and
  different visualizations (image grids, saliency maps) — a genuine scope
  expansion, not a file-extension toggle.
- **Investigation history is stored in browser localStorage** in this
  build, not a database.

## Future work

- Real SHAP integration for supported model families
- Multiple-testing correction (Benjamini-Hochberg) across slice/feature
  significance tests
- Bootstrapped confidence intervals on all reported metrics
- Fairness/bias metrics across a user-designated sensitive attribute
- Model comparison mode (investigate two versions side-by-side)
- Celery/RQ + Postgres + object storage for real multi-user deployment
- Image model (CNN) and transformer-based text model support

.

## Contributing

Issues and PRs welcome. Run `pytest tests/ -v` in `backend/` and
`npm run build` in `frontend/` before submitting.

---

*ModelLens reports statistical associations observed in your data. It
does not establish causation, does not guarantee real-world significance,
and does not certify a model for any particular use case.*
