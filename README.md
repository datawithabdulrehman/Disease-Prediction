# Diabetes Risk Screening API

A FastAPI service that wraps the trained model from
`diabetes_prediction_pipeline.ipynb`, plus a static single-page UI for
manually testing predictions.

## Project structure

```
diabetes_api/
├── main.py                     # FastAPI app
├── diabetes_model_bundle.pkl   # <-- copy this in from your notebook (see below)
├── requirements.txt
└── static/
    └── index.html               # demo frontend
```

## 1. Get the model file in place

The notebook's final section (`8. Model Saving (Pickle)`) writes
`diabetes_model_bundle.pkl` next to the notebook. Copy that file into this
`diabetes_api/` folder so it sits next to `main.py`:

```bash
cp /path/to/diabetes_model_bundle.pkl diabetes_api/diabetes_model_bundle.pkl
```

The API refuses to start if this file is missing — it will tell you exactly
where it looked.

## 2. Install dependencies

```bash
cd diabetes_api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it

```bash
uvicorn main:app --reload
```

- **Demo UI:** http://127.0.0.1:8000
- **Interactive API docs (Swagger):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

## How it works

- `GET /schema` returns every feature the model expects, split into
  numerical fields and categorical fields (with their exact allowed values,
  taken straight from the fitted `LabelEncoder`s). The frontend calls this
  once on page load to build the form — so the UI can never drift out of
  sync with what the model was actually trained on.
- `POST /predict` takes one value per feature, encodes categoricals and
  scales numericals using the *exact* encoder/scaler objects saved from
  training, and returns:
  - `prediction` (0/1) and `prediction_label`
  - `probability_diabetes` (0–1)
  - `risk_band` (`Low` / `Moderate` / `High` / `Very High`)

## Deploying

For production, at minimum:
- Restrict CORS in `main.py` (`allow_origins=["*"]` → your real frontend origin)
- Run behind `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2` or a
  process manager (gunicorn + uvicorn workers), behind a reverse proxy (nginx/Caddy)
- Add authentication if this will be reachable outside a trusted network —
  as-is it is an open, unauthenticated screening endpoint.

## Disclaimer

This tool produces a statistical risk estimate for demonstration purposes.
It is **not** a medical diagnostic device and must not be used as a
substitute for professional clinical judgment.
