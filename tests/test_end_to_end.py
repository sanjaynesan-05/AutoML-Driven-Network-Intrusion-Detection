import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor,permitted_features,clean_infinities

def test_small_training_smoke():
    x=pd.DataFrame({"label":[0,1]*20,"attack_cat":["Normal","DoS"]*20,"id":range(40),"n":list(range(40)),"cat":["tcp","udp"]*20})
    X=clean_infinities(permitted_features(x)); model=Pipeline([("preprocess",build_preprocessor(x)),("classifier",RandomForestClassifier(n_estimators=5,random_state=42))])
    model.fit(X,x.label); assert len(model.predict(X[:4]))==4
