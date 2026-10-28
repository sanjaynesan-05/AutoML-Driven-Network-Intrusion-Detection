import json
import pandas as pd
from .data_loader import load_partitions
from .config import RESULTS

def run():
    train,test=load_partitions(); report={}
    for split,df in (("train",train),("test",test)):
        report[split]={"binary_counts":{str(k):int(v) for k,v in df.label.value_counts().items()},"attack_category_counts":{str(k):int(v) for k,v in df.attack_cat.value_counts().items()},"rare_categories_below_1pct":df.attack_cat.value_counts(normalize=True)[lambda x:x<.01].index.tolist()}
    report["interpretation"]="No demographic attributes are used. This is a class representation, collection, and category-level performance analysis; it is not demographic fairness analysis. Rare attack classes warrant per-class recall review."
    (RESULTS/"bias_analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

if __name__=="__main__":run()
