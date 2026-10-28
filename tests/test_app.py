import pandas as pd
import pytest
from app import prepare_upload

def test_app_requires_schema():
    with pytest.raises(ValueError): prepare_upload(pd.DataFrame({"x":[1]}),["x","y"])
def test_app_drops_target_fields():
    out=prepare_upload(pd.DataFrame({"x":[1],"label":[0],"attack_cat":["N"]}),["x"])
    assert list(out.columns)==["x"]
