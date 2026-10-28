import pandas as pd
from src.preprocessing import permitted_features

def test_target_fields_excluded():
    x=pd.DataFrame({"label":[0],"attack_cat":["Normal"],"id":[1],"v":[2]})
    assert list(permitted_features(x).columns)==["v"]
def test_intentional_leak_name_is_explicit(): assert "__intentional_leak_label".startswith("__intentional_leak_")
