import pandas as pd
import pytest
from src.data_validation import validate_partitions

def sample(): return pd.DataFrame({"id":[1,2],"x":[1.,2.],"attack_cat":["Normal","DoS"],"label":[0,1]})
def test_valid_schema(): validate_partitions(sample(),sample())
def test_missing_target():
    x=sample().drop(columns="label")
    with pytest.raises(ValueError): validate_partitions(x,sample())
def test_invalid_label():
    x=sample(); x.loc[0,"label"]=2
    with pytest.raises(ValueError): validate_partitions(x,sample())
def test_schema_mismatch():
    x=sample(); y=sample().rename(columns={"x":"z"})
    with pytest.raises(ValueError): validate_partitions(x,y)
