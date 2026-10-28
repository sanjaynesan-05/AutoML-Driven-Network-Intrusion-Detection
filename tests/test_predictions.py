import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import build_preprocessor,permitted_features
from pathlib import Path

def test_binary_predictions_and_target_exclusion():
    x=pd.DataFrame({"id":range(8),"label":[0,1]*4,"attack_cat":["N","A"]*4,"v":range(8),"c":["x","y"]*4}); X=permitted_features(x)
    model=Pipeline([("preprocess",build_preprocessor(x)),("classifier",RandomForestClassifier(n_estimators=5,random_state=1))]).fit(X,x.label)
    assert set(model.predict(X)) <= {0,1}
    assert "label" not in model.named_steps["preprocess"].feature_names_in_

def test_saved_pipeline_round_trip(tmp_path):
    x=pd.DataFrame({"id":range(8),"label":[0,1]*4,"attack_cat":["N","A"]*4,"v":range(8),"c":["x","y"]*4}); X=permitted_features(x)
    model=Pipeline([("preprocess",build_preprocessor(x)),("classifier",RandomForestClassifier(n_estimators=5,random_state=1))]).fit(X,x.label)
    path=tmp_path/"model.joblib"; joblib.dump(model,path); loaded=joblib.load(path)
    assert loaded.predict(X[:2]).tolist()==model.predict(X[:2]).tolist()
