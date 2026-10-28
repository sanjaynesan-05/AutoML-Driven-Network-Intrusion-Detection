import json
import joblib
from .data_loader import load_partitions
from .preprocessing import permitted_features,clean_infinities
from .config import RESULTS,MODELS
from .evaluation import evaluate
from .model_registry import select_and_export

def run():
    meta=select_and_export(); train,test=load_partitions(); X=clean_infinities(permitted_features(test))
    if meta["model_type"]=="autogluon":
        from autogluon.tabular import TabularPredictor
        model=TabularPredictor.load(meta["artifact"])
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,classification_report
        import time
        t=time.perf_counter(); pred=model.predict(X); prob=model.predict_proba(X)[1].to_numpy(); latency=time.perf_counter()-t
        metrics={"accuracy":accuracy_score(test.label,pred),"precision":precision_score(test.label,pred),"recall":recall_score(test.label,pred),"f1":f1_score(test.label,pred),"roc_auc":roc_auc_score(test.label,prob),"pr_auc":average_precision_score(test.label,prob),"prediction_seconds":latency,"confusion_matrix":confusion_matrix(test.label,pred).tolist(),"per_class":classification_report(test.label,pred,output_dict=True,zero_division=0)}
    else:
        model=joblib.load(MODELS/"final_model.joblib"); metrics=evaluate(model,X,test.label.astype(int))
    metrics.update(meta)
    (RESULTS/"final_model_metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    # Reload verification
    if meta["model_type"]=="autogluon": model.predict(clean_infinities(permitted_features(test.head(3))))
    else: joblib.load(MODELS/"final_model.joblib").predict(clean_infinities(permitted_features(test.head(3))))

if __name__=="__main__":run()
