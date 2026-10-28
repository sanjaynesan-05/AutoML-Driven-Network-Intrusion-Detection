import pandas as pd
from .config import TRAIN_PATH, TEST_PATH, FEATURES_PATH

def load_partitions():
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
    return (pd.read_csv(TRAIN_PATH, low_memory=False), pd.read_csv(TEST_PATH, low_memory=False))

def load_feature_descriptions():
    if not FEATURES_PATH.is_file():
        return pd.DataFrame()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(FEATURES_PATH, encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(FEATURES_PATH, low_memory=False, encoding="latin-1", on_bad_lines="warn")
