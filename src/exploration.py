import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .data_validation import profile, make_demo_sample
from .data_loader import load_partitions
from .config import FIGURES, RESULTS

def run():
    report=profile(); make_demo_sample(); train,test=load_partitions()
    for name,frame in (("train",train),("test",test)):
        frame.attack_cat.value_counts().plot(kind="bar",figsize=(10,4)); plt.title(f"{name}: attack categories"); plt.tight_layout(); plt.savefig(FIGURES/f"{name}_attack_categories.png"); plt.close()
        frame.label.value_counts().sort_index().plot(kind="bar"); plt.title(f"{name}: binary class distribution"); plt.tight_layout(); plt.savefig(FIGURES/f"{name}_label_distribution.png"); plt.close()
    numeric=train.select_dtypes("number").drop(columns=["label","id"],errors="ignore")
    cols=numeric.var().sort_values(ascending=False).head(18).index
    plt.figure(figsize=(12,9)); sns.heatmap(train[list(cols)].corr(),cmap="coolwarm",center=0); plt.title("Train numeric feature correlation (top variance)"); plt.tight_layout(); plt.savefig(FIGURES/"correlation_heatmap.png"); plt.close()
    numeric[list(cols[:8])].hist(figsize=(12,8),bins=30); plt.tight_layout(); plt.savefig(FIGURES/"numeric_distributions.png"); plt.close()
    (RESULTS/"eda_summary.json").write_text(json.dumps({"train_shape":report["train"]["shape"],"test_shape":report["test"]["shape"],"train_label":report["train"]["label_counts"],"test_label":report["test"]["label_counts"],"train_duplicates":report["train"]["duplicate_rows"],"test_duplicates":report["test"]["duplicate_rows"]},indent=2),encoding="utf-8")

if __name__=="__main__":run()
