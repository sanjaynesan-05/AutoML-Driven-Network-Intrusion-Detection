# AutoML-Driven Network Intrusion Detection

Academic research prototype for UNSW-NB15 binary traffic classification. It compares a manual Random Forest baseline and tuning against FLAML and AutoGluon, evaluates the supplied held-out partition, isolates an intentional target leakage diagnostic, and provides a Streamlit demo.

## Setup

Python 3.11 is used because the AutoML dependencies support it. From `D:\AUTOML`:

```powershell
py -3.11 -m venv D:\AUTOML\.venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
```

The interpreter should be `D:\AUTOML\.venv\Scripts\python.exe`. Use the venv interpreter directly for all commands if activation is unavailable.
The requirements pin PyArrow to a NumPy 1.x-compatible release; keep this pairing intact because Streamlit uses PyArrow for chart data conversion.

## Data and workflow

Place the original files in `dataset`: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`, and `NUSW-NB15_features.csv`. Original files are never modified. Run `\.\run_all.ps1` for the complete workflow, or use the same interpreter for individual stages, for example:

```powershell
.\.venv\Scripts\python.exe -m src.exploration
.\.venv\Scripts\python.exe -m src.baseline
.\.venv\Scripts\python.exe -m src.tuning
.\.venv\Scripts\python.exe -m src.automl_flaml
.\.venv\Scripts\python.exe -m src.automl_autogluon
.\.venv\Scripts\python.exe -m src.leakage_experiment
.\.venv\Scripts\python.exe -m src.bias_analysis
.\.venv\Scripts\python.exe -m src.train_final_model
.\.venv\Scripts\python.exe -m src.explainability
.\.venv\Scripts\python.exe -m src.reporting
```

Full mode uses configurable 900-second AutoML budgets; `\.\run_all.ps1 -Dev` uses 90 seconds.

## Model and deployment

The binary target is `label` (0 normal, 1 attack). `label`, `attack_cat`, and sequential `id` are excluded. Numeric and categorical preprocessing is fitted only on training data. Launch after model generation with `\.venv\Scripts\python.exe -m streamlit run app.py`.

## Results

Machine-readable profile, metrics, model card, plots, sample input, and leakage results are under `results/`; the report and presentation materials are in `reports/`.

## Research limits

This is a research and teaching prototype trained on UNSW-NB15. Benchmark results do not validate live-network or production defense use. A prediction is an estimate, not proof of an attack. Consult `results/model_card.md` and `reports/final_report.md`.
