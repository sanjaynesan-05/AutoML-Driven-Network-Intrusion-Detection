import pandas as pd
from src.preprocessing import build_preprocessor,permitted_features,clean_infinities

def test_preprocessor_unknown_category_and_infinity():
    x=pd.DataFrame({"id":[1,2],"label":[0,1],"attack_cat":["N","A"],"num":[1.,float("inf")],"cat":["a","b"]})
    pp=build_preprocessor(x); train=clean_infinities(permitted_features(x)); pp.fit(train); transformed=pp.transform(pd.DataFrame({"num":[float("inf")],"cat":["new"]}))
    assert transformed.shape[0]==1
