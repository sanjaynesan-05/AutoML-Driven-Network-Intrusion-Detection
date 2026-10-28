import json
import joblib
import pandas as pd
from .config import RESULTS,MODELS
from .data_loader import load_partitions
from .preprocessing import permitted_features

def select_and_export():
    candidates=[]
    base=RESULTS/"baseline_metrics.json"
    if base.exists():
        m=json.loads(base.read_text(encoding="utf-8"))
        if m.get("validation_f1") is not None: candidates.append((float(m["validation_f1"]),"baseline",MODELS/"baseline.joblib",m))
    tune=RESULTS/"tuning_results.csv"
    if tune.exists():
        for _,row in pd.read_csv(tune).iterrows():
            name=str(row["model"]); path=MODELS/("grid_search.joblib" if name=="grid" else "random_search.joblib" if name=="random" else "optuna.joblib")
            if path.exists(): candidates.append((float(row.get("validation_f1",0)),name,path,row.to_dict()))
    ag_metrics=RESULTS/"autogluon_metrics.json"; ag_dir=MODELS/"autogluon"
    if ag_metrics.exists() and (ag_dir/"predictor.pkl").exists():
        leaderboard=RESULTS/"autogluon_leaderboard.csv"
        validation=float(pd.read_csv(leaderboard).iloc[0].get("score_val",0)) if leaderboard.exists() else 0.0
        candidates.append((validation,"AutoGluon Tabular",ag_dir,json.loads(ag_metrics.read_text(encoding="utf-8"))))
    flaml_metrics=RESULTS/"flaml_metrics.json"
    if flaml_metrics.exists() and (MODELS/"flaml.joblib").exists():
        fm=json.loads(flaml_metrics.read_text(encoding="utf-8")); candidates.append((float(fm.get("validation_f1",0)),"FLAML",MODELS/"flaml.joblib",fm))
    if not candidates: raise FileNotFoundError("No valid trained candidate artifacts found")
    # Validation score determines selection; tie favors baseline simplicity.
    score,name,path,metrics=max(candidates,key=lambda x:x[0])
    if name == "AutoGluon Tabular":
        artifact=str(path)
    else:
        model=joblib.load(path); joblib.dump(model,MODELS/"final_model.joblib",compress=3); artifact="final_model.joblib"
    train,_=load_partitions(); feature_names=permitted_features(train).columns.tolist()
    metadata={"model_name":name,"selection_rule":"highest training-only validation F1 among available valid candidates; held-out test metrics are not used for selection","validation_f1":score,"target":"label (0=Normal, 1=Attack)","input_features":feature_names,"excluded_features":{"label":"target","attack_cat":"target-derived label","id":"partition-specific identifier"},"artifact":artifact,"model_type":"autogluon" if name=="AutoGluon Tabular" else "joblib"}
    (RESULTS/"model_metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    return metadata
