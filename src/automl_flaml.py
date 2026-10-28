import json,time,traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from .data_loader import load_partitions
from .preprocessing import permitted_features,clean_infinities
from .config import RESULTS,MODELS,SEED,AUTOML_SECONDS
from .evaluation import evaluate

def run(seconds=None):
    import os
    seconds = int(os.environ.get("AUTOML_BUDGET", seconds or AUTOML_SECONDS))
    train,test=load_partitions(); X=clean_infinities(permitted_features(train)); y=train.label.astype(int); Xv=clean_infinities(permitted_features(test)); yv=test.label.astype(int)
    Xa,Xvalid,ya,yvalid=train_test_split(X,y,test_size=.15,random_state=SEED,stratify=y)
    try:
        from flaml import AutoML
        automl=AutoML(); t=time.perf_counter(); automl.fit(X_train=Xa,y_train=ya,X_val=Xvalid,y_val=yvalid,task="classification",metric="f1",time_budget=seconds,estimator_list=["lgbm","xgboost","rf","extra_tree"],eval_method="holdout",n_jobs=2,seed=SEED,verbose=1); elapsed=time.perf_counter()-t
        metrics=evaluate(automl,Xv,yv); metrics.update(model="FLAML",validation_f1=float(1-automl.best_loss if automl.best_loss is not None else float("nan")),search_seconds=elapsed,best_estimator=automl.best_estimator,best_config=automl.best_config)
        import joblib; joblib.dump(automl,MODELS/"flaml.joblib",compress=3)
        history=getattr(automl,"model_history",{}) or {}
        pd.DataFrame([{"trial_model":str(k),"details":str(v)} for k,v in history.items()]).to_csv(RESULTS/"flaml_trials.csv",index=False)
        if not history:
            pd.DataFrame([{"record_type":"selected trial summary; detailed internal trial history is unavailable in this FLAML release","estimator":automl.best_estimator,"validation_f1":metrics["validation_f1"],"config":str(automl.best_config)}]).to_csv(RESULTS/"flaml_trials.csv",index=False)
        pd.DataFrame([metrics]).to_csv(RESULTS/"automl_results.csv",index=False)
        (RESULTS/"flaml_metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
        (RESULTS/"flaml_failure.txt").unlink(missing_ok=True)
    except Exception as e:
        (RESULTS/"flaml_failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise

if __name__=="__main__":run()
