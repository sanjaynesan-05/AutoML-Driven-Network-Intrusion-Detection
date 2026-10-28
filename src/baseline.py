import json, time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from .config import TRAIN_PATH, TEST_PATH, MODELS, RESULTS, SEED, MAX_JOBS
from .preprocessing import build_preprocessor, permitted_features, clean_infinities
from .evaluation import evaluate
from .data_loader import load_partitions

def run():
    train, test = load_partitions(); y, yt = train.label.astype(int), test.label.astype(int)
    X, Xt = clean_infinities(permitted_features(train)), clean_infinities(permitted_features(test))
    sample=train.sample(min(20000,len(train)),random_state=SEED)
    Xs=clean_infinities(permitted_features(sample)); ys=sample.label.astype(int)
    Xfit,Xval,yfit,yval=train_test_split(Xs,ys,test_size=.2,random_state=SEED,stratify=ys)
    validation_model=Pipeline([("preprocess",build_preprocessor(sample)),("classifier",RandomForestClassifier(n_estimators=100,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=MAX_JOBS,random_state=SEED))])
    vt=time.perf_counter(); validation_model.fit(Xfit,yfit); validation_fit=time.perf_counter()-vt
    validation_f1=float(f1_score(yval,validation_model.predict(Xval),zero_division=0))
    model = Pipeline([("preprocess", build_preprocessor(train)), ("classifier", RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=MAX_JOBS, random_state=SEED))])
    t = time.perf_counter(); model.fit(X, y); fit = time.perf_counter()-t
    metrics = evaluate(model, Xt, yt); metrics.update(model="RandomForest baseline", validation_f1=validation_f1, validation_fit_seconds=validation_fit, training_seconds=fit, train_f1=float(f1_score(y, model.predict(X))))
    joblib.dump(model, MODELS / "baseline.joblib", compress=3)
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics | {"confusion_matrix": str(metrics["confusion_matrix"]), "per_class": "see baseline_metrics.json"}]).to_csv(RESULTS / "baseline_results.csv", index=False)
    return model, metrics

if __name__ == "__main__": run()
