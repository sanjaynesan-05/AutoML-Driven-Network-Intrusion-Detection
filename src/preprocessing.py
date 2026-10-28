import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from .config import EXCLUDED

def _infinity_to_missing(values):
    return np.where(np.isfinite(values), values, np.nan)

def permitted_features(frame):
    return frame.drop(columns=list(EXCLUDED), errors="ignore")

def build_preprocessor(frame):
    X = permitted_features(frame)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num = Pipeline([("infinity_to_missing", FunctionTransformer(_infinity_to_missing, feature_names_out="one-to-one")), ("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], remainder="drop", verbose_feature_names_out=False)

def clean_infinities(X):
    X = X.copy()
    for c in X.select_dtypes(include=np.number):
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
    return X
