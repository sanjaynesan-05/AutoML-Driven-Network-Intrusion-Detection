import json,time,traceback
import pandas as pd
from .data_loader import load_partitions
from .preprocessing import permitted_features
from .config import RESULTS,MODELS,SEED,AUTOML_SECONDS

def run(seconds=None):
    import os
    seconds = int(os.environ.get("AUTOML_BUDGET", seconds or AUTOML_SECONDS))
    train,test=load_partitions(); X=permitted_features(train); Xtest=permitted_features(test)
    train_data=X.copy(); train_data["label"]=train.label.astype(int)
    test_data=Xtest.copy(); test_data["label"]=test.label.astype(int)
    path=MODELS/"autogluon"
    try:
        from autogluon.tabular import TabularPredictor
        if (path / "predictor.pkl").exists():
            predictor=TabularPredictor.load(str(path))
            timing_file=RESULTS/"autogluon_search_seconds.txt"
            elapsed_search=float(timing_file.read_text(encoding="utf-8")) if timing_file.exists() else float("nan")
        else:
            t=time.perf_counter(); predictor=TabularPredictor(label="label",path=str(path),eval_metric="f1",problem_type="binary").fit(train_data=train_data,time_limit=seconds,presets="medium_quality",num_cpus=2,verbosity=2,ag_args_fit={"random_seed":SEED}); elapsed_search=time.perf_counter()-t
        t=time.perf_counter(); scores=predictor.evaluate(test_data); latency=time.perf_counter()-t
        leaderboard=predictor.leaderboard(silent=True); leaderboard.to_csv(RESULTS/"autogluon_leaderboard.csv",index=False)
        result={"model":"AutoGluon Tabular","selected_model":predictor.model_best,"test_metrics":scores,"validation_f1":float(leaderboard.iloc[0]["score_val"]),"prediction_seconds_including_evaluate":latency,"search_seconds_configured":90 if (path / "predictor.pkl").exists() else seconds,"search_seconds_measured":elapsed_search}
        (RESULTS/"autogluon_metrics.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
        flat={"model":"AutoGluon Tabular","selected_model":predictor.model_best,"search_seconds":elapsed_search,"validation_f1":float(leaderboard.iloc[0].get("score_val",float("nan"))),**scores}
        pd.DataFrame([flat]).to_csv(RESULTS/"automl_results.csv",mode="a",header=not (RESULTS/"automl_results.csv").exists(),index=False)
        # Reload and verify prediction path.
        reloaded=TabularPredictor.load(str(path)); reloaded.predict(Xtest.head(5))
        (RESULTS/"autogluon_failure.txt").unlink(missing_ok=True)
    except Exception:
        (RESULTS/"autogluon_failure.txt").write_text(traceback.format_exc(),encoding="utf-8"); raise

if __name__=="__main__":run()
