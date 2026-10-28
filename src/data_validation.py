import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .config import TARGET, RESULTS
from .data_loader import load_partitions, load_feature_descriptions

def _sample_numeric(series):
    clean=series.replace([np.inf,-np.inf],np.nan).dropna()
    return clean.sample(min(20000,len(clean)),random_state=42)

def validate_partitions(train, test):
    for name, frame in (("train", train), ("test", test)):
        if frame.columns.duplicated().any():
            raise ValueError(f"{name} has duplicated column names: {frame.columns[frame.columns.duplicated()].tolist()}")
        missing = {TARGET, "attack_cat"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required targets: {sorted(missing)}")
        values = set(pd.to_numeric(frame[TARGET], errors="coerce").dropna().unique())
        if not values <= {0, 1}:
            raise ValueError(f"{name} has unexpected binary target values: {sorted(values)}")
    a, b = set(train.columns) - {TARGET, "attack_cat"}, set(test.columns) - {TARGET, "attack_cat"}
    if a != b:
        raise ValueError(f"Train/test feature schema mismatch; train-only={sorted(a-b)}, test-only={sorted(b-a)}")

def profile():
    train, test = load_partitions()
    validate_partitions(train, test)
    desc = load_feature_descriptions()
    report = {}
    for name, frame in (("train", train), ("test", test)):
        numeric = frame.select_dtypes(include=np.number)
        report[name] = {
            "shape": list(frame.shape), "columns": list(frame.columns),
            "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
            "missing_by_column": frame.isna().sum().astype(int).to_dict(),
            "infinite_by_column": {c: int(np.isinf(numeric[c]).sum()) for c in numeric},
            "duplicate_rows": int(frame.duplicated().sum()),
            "label_counts": frame[TARGET].value_counts(dropna=False).to_dict(),
            "attack_category_counts": frame.attack_cat.value_counts(dropna=False).to_dict(),
        }
    report["train_test_feature_columns_match"] = (set(train.columns)-{TARGET,"attack_cat"}) == (set(test.columns)-{TARGET,"attack_cat"})
    common = (set(train.columns) & set(test.columns)) - {TARGET, "attack_cat"}
    numeric = [c for c in common if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])]
    report["numeric_distribution_shift"] = {c: {"train_mean": float(train[c].replace([np.inf,-np.inf],np.nan).mean()), "test_mean": float(test[c].replace([np.inf,-np.inf],np.nan).mean()), "ks_statistic": float(ks_2samp(_sample_numeric(train[c]), _sample_numeric(test[c])).statistic)} for c in numeric}
    feature_cols=sorted(common)
    train_hash = pd.util.hash_pandas_object(train[feature_cols], index=False)
    test_hash = pd.util.hash_pandas_object(test[feature_cols], index=False)
    report["exact_duplicate_feature_rows_across_partitions"] = int(train_hash.isin(set(test_hash)).sum())
    report["feature_reference"] = {"shape": list(desc.shape), "columns": list(desc.columns), "sample": desc.head(50).fillna("").to_dict(orient="records")}
    report["excluded_features"] = {"label": "target", "attack_cat": "target-derived multiclass label", "id": "partition-specific record identifier excluded from the reusable binary model"}
    RESULTS.joinpath("dataset_profile.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.concat([train[TARGET].value_counts().rename("train"), test[TARGET].value_counts().rename("test")], axis=1).fillna(0).to_csv(RESULTS / "class_distribution.csv")
    return report

def make_demo_sample(rows=200):
    _, test = load_partitions()
    sample = test.sample(min(rows, len(test)), random_state=42).drop(columns=["label", "attack_cat"], errors="ignore")
    sample.to_csv(RESULTS / "sample_upload.csv", index=False)
    attack = test.loc[test.label == 1].sample(min(rows, int((test.label == 1).sum())), random_state=11).drop(columns=["label", "attack_cat"], errors="ignore")
    normal = test.loc[test.label == 0].sample(min(rows, int((test.label == 0).sum())), random_state=19).drop(columns=["label", "attack_cat"], errors="ignore")
    attack.to_csv(RESULTS / "sample_attack_heavy.csv", index=False)
    normal.to_csv(RESULTS / "sample_normal_heavy.csv", index=False)
    return sample
