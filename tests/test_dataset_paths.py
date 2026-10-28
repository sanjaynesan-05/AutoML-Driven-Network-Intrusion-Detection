from src.config import TRAIN_PATH,TEST_PATH,FEATURES_PATH

def test_supplied_dataset_paths_exist():
    assert TRAIN_PATH.is_file(); assert TEST_PATH.is_file(); assert FEATURES_PATH.is_file()
