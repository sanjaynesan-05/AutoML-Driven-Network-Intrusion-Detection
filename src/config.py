from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
TRAIN_PATH = DATASET_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATASET_DIR / "UNSW_NB15_testing-set.csv"
FEATURES_PATH = DATASET_DIR / "NUSW-NB15_features.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "artifacts" / "models"
EXPLAINERS = ROOT / "artifacts" / "explainers"
LOGS = ROOT / "logs"
SEED = 42
TARGET = "label"
EXCLUDED = {"label": "target", "attack_cat": "target-derived multiclass label", "id": "partition-specific record identifier; not a reusable traffic feature"}
AUTOML_SECONDS = 900
DEV_AUTOML_SECONDS = 90
MAX_JOBS = 4

for _path in (RESULTS, FIGURES, MODELS, EXPLAINERS, LOGS):
    _path.mkdir(parents=True, exist_ok=True)
