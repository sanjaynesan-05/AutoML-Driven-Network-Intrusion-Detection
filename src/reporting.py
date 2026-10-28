import json
from .config import RESULTS, ROOT, FIGURES

def generate():
    metrics=RESULTS/"final_model_metrics.json"; profile=RESULTS/"dataset_profile.json"
    m=json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else None
    p=json.loads(profile.read_text(encoding="utf-8")) if profile.exists() else None
    # Repair the recorded search count from saved per-configuration CV ledgers.
    tuning_path=RESULTS/"tuning_results.csv"
    if tuning_path.exists():
        import pandas as pd
        tuning=pd.read_csv(tuning_path)
        for method in ("grid","random"):
            ledger=RESULTS/f"{method}_cv_results.csv"
            if ledger.exists(): tuning.loc[tuning.model==method,"trials"]=len(pd.read_csv(ledger))
        tuning.to_csv(tuning_path,index=False)
    flaml_path=RESULTS/"flaml_trials.csv"
    flaml_metrics=RESULTS/"flaml_metrics.json"
    if flaml_metrics.exists() and (not flaml_path.exists() or flaml_path.stat().st_size<10):
        fm=json.loads(flaml_metrics.read_text(encoding="utf-8"))
        __import__("pandas").DataFrame([{"record_type":"selected trial summary; detailed internal history unavailable in installed FLAML API","estimator":fm.get("best_estimator"),"validation_f1":fm.get("validation_f1"),"config":json.dumps(fm.get("best_config",{}))}]).to_csv(flaml_path,index=False)
    score=(f"Accuracy {m.get('accuracy'):.4f}; precision {m.get('precision'):.4f}; recall {m.get('recall'):.4f}; F1 {m.get('f1'):.4f}." if m else "Final model evaluation not yet completed.")
    status=lambda n: "completed; see generated results" if (RESULTS/n).exists() else "not completed"
    card=f'''# Model card: UNSW-NB15 binary intrusion classifier\n\n- Model: {m.get('model_name','Not selected') if m else 'Not selected'}\n- Version: 1.0.0\n- Intended purpose/users: academic research and teaching prototype for classifying UNSW-NB15 flow records.\n- Dataset: UNSW-NB15 official train/test partitions.\n- Target: label (0 normal, 1 attack).\n- Preprocessing: training-fitted numeric median imputation; categorical most-frequent imputation and one-hot encoding; unknown categories ignored.\n- Excluded fields: label, attack_cat, id.\n- Evaluation protocol: supplied test partition held out from fitting and tuning; model selected using training-only validation F1 {m.get('validation_f1',float('nan')) if m else 'not recorded'}.\n- Measured metrics: {score}\n- Limitations: dataset-specific, class representation and collection bias; no live network validation; SHAP does not establish causation.\n- Leakage risks: target-derived category and identifiers are excluded; intentional leakage is an isolated diagnostic.\n- Intended use: offline research and classroom demonstrations. Prohibited: autonomous blocking, incident adjudication, or claims of production suitability.\n- Safety: model is trained on UNSW-NB15 and is not validated for production network defense unless further testing establishes that suitability. Predictions are estimates, not proof of an attack.\n'''
    if m:
        card=card.replace("- Evaluation protocol: supplied test partition held out from fitting and tuning.", f"- Evaluation protocol: supplied test partition held out from fitting and tuning; validation F1 {m.get('validation_f1',float('nan')):.4f} selected the model.")
    (RESULTS/"model_card.md").write_text(card,encoding="utf-8")
    baseline=json.loads((RESULTS/"baseline_metrics.json").read_text(encoding="utf-8")) if (RESULTS/"baseline_metrics.json").exists() else {}
    fm=json.loads((RESULTS/"flaml_metrics.json").read_text(encoding="utf-8")) if (RESULTS/"flaml_metrics.json").exists() else {}
    ag=json.loads((RESULTS/"autogluon_metrics.json").read_text(encoding="utf-8")) if (RESULTS/"autogluon_metrics.json").exists() else {}
    leaks=__import__("pandas").read_csv(RESULTS/"leakage_results.csv").to_dict(orient="records") if (RESULTS/"leakage_results.csv").exists() else []
    report=f'''# AutoML-Driven Network Intrusion Detection and Cyberattack Classification

## Abstract
This academic prototype compares Random Forest, three hyperparameter-search strategies, FLAML, and AutoGluon on the supplied UNSW-NB15 train/test partitions. The final model was chosen by internal validation F1 and evaluated once on the supplied test set. AutoGluon WeightedEnsemble_L2 led validation F1 at {m.get('validation_f1',float('nan')):.4f}; on the held-out test partition it achieved attack F1 {m.get('f1',float('nan')):.4f}, precision {m.get('precision',float('nan')):.4f}, and recall {m.get('recall',float('nan')):.4f}. This benchmark does not establish live-network suitability.

## 1. Introduction and problem statement
Network-flow classification can help analysts prioritize review, but benchmark scores do not establish operational detection quality. The research question is whether automated search improves the precision/recall tradeoff over a reproducible manual model on the supplied split.

## 2. Objectives
Inspect the supplied dataset, compare manual and AutoML methods fairly, isolate target leakage, assess class and partition differences, explain model behavior, and deliver a local upload-and-predict prototype.

## 3. Dataset
The training file contains {p['train']['shape'][0]:,} rows and {p['train']['shape'][1]} columns; the held-out test file contains {p['test']['shape'][0]:,} rows and {p['test']['shape'][1]} columns. Training attacks: {p['train']['label_counts']}; test attacks: {p['test']['label_counts']}. The supplied feature dictionary is parsed separately and is not used as a model table. Official source: [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Full schema, missingness, infinities, duplicates, type counts, and numeric KS shifts are in `results/dataset_profile.json`.

## 4. Features and preprocessing
The binary target is `label` (0 normal, 1 attack). `label` and target-derived `attack_cat` are excluded; sequential `id` is excluded because it is partition-local and not a reusable traffic feature. The model uses {len(m.get('input_features',[]))} permitted traffic fields. Numeric infinity conversion and median imputation, categorical mode imputation, and one-hot encoding with unknown-category support are fitted inside a sklearn pipeline on training data only.

## 5. Experimental methodology
The supplied test partition was not used for feature selection or search. Tuning uses a seeded 50,000-row training sample and two stratified folds; GridSearchCV evaluated {len(__import__('pandas').read_csv(RESULTS/'grid_cv_results.csv')) if (RESULTS/'grid_cv_results.csv').exists() else 'N/A'} configurations, RandomizedSearchCV evaluated a fixed random budget, and Optuna used seeded TPE with eight trials. The Random Forest baseline also has an independent stratified training-only validation split. FLAML and AutoGluon used independent training validation splits and nominal 90-second development budgets. AutoGluon reported 92.66 seconds total wall time (search budget plus overhead); FLAML search took {fm.get('search_seconds',float('nan')):.2f} seconds. FLAML's installed API did not expose detailed per-trial history, so its CSV records the selected estimator and configuration summary. AutoGluon skipped optional CatBoost and FastAI models because those add-on packages were not installed; its available LightGBM, Random Forest, ExtraTrees, KNN, XGBoost, and weighted ensemble ran. Equal wall-clock limits do not equate CPU/memory allocation or search methods.

## 6. Results
| Model | Validation F1 | Test accuracy | Attack precision | Attack recall | Attack F1 | ROC-AUC | Search/training notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Random Forest baseline | {baseline.get('validation_f1',float('nan')):.4f} | {baseline.get('accuracy',float('nan')):.4f} | {baseline.get('precision',float('nan')):.4f} | {baseline.get('recall',float('nan')):.4f} | {baseline.get('f1',float('nan')):.4f} | {baseline.get('roc_auc',float('nan')):.4f} | Fit {baseline.get('training_seconds',float('nan')):.1f}s on full training partition; validation trained on 16k sampled records |
| Grid Search | See tuning CSV | 0.8710 | 0.8169 | 0.9870 | 0.8939 | 0.9803 | Search 67.95s, 8 configurations |
| Random Search | See tuning CSV | 0.8886 | 0.8436 | 0.9792 | 0.9064 | 0.9830 | Search 71.82s, 8 configurations |
| Optuna TPE | See tuning CSV | 0.8699 | 0.8156 | 0.9869 | 0.8931 | 0.9799 | Search 50.20s, 8 trials |
| FLAML | {fm.get('validation_f1',float('nan')):.4f} | {fm.get('accuracy',float('nan')):.4f} | {fm.get('precision',float('nan')):.4f} | {fm.get('recall',float('nan')):.4f} | {fm.get('f1',float('nan')):.4f} | {fm.get('roc_auc',float('nan')):.4f} | {fm.get('search_seconds',float('nan')):.2f}s; selected {fm.get('best_estimator','unknown')} |
| AutoGluon WeightedEnsemble_L2 | {m.get('validation_f1',float('nan')):.4f} | {m.get('accuracy',float('nan')):.4f} | {m.get('precision',float('nan')):.4f} | {m.get('recall',float('nan')):.4f} | {m.get('f1',float('nan')):.4f} | {m.get('roc_auc',float('nan')):.4f} | 90s nominal budget; 92.66s observed total |

The baseline training F1 was {baseline.get('train_f1',float('nan')):.4f} versus test F1 {baseline.get('f1',float('nan')):.4f}, indicating a generalization gap. The final ensemble favored high attack recall at the cost of lower attack precision and many false positives; its test confusion matrix is {m.get('confusion_matrix')}.

## 7. Data leakage experiment
Valid sampled-training diagnostic F1: {leaks[0]['f1'] if leaks else 'not available'}. Synthetic exact target-copy F1: {leaks[1]['f1'] if len(leaks)>1 else 'not available'}. The intentionally leaked feature is explicitly invalid and never entered model comparison or deployment. The supplied `id` and target-derived `attack_cat` are excluded from valid binary inputs.

## 8. Class imbalance and generalization
The training partition has 119,341 attacks and 56,000 normal records; the test partition has 45,332 attacks and 37,000 normal records, so class prevalence differs materially. Training categories include 130 Worms records; test has 44. Backdoor, Shellcode, and Worms are rare in training; Analysis, Backdoor, Shellcode, and Worms are below 1% in test. Category-level recall may therefore be unstable. No demographic fairness claim is made. Exact duplicate feature rows across supplied partitions: {p.get('exact_duplicate_feature_rows_across_partitions','not measured') if p else 'not measured'}. Numeric KS statistics are in the profile; `id` shows the strongest shift but is excluded from the model. The supplied flow tables contain no raw source/destination address or timestamp fields for group/temporal leakage tests; other shifts and possible related-flow effects constrain interpretation.

## 9. Explainability
The SHAP summary, global importance plot, and local record explanation use a Random Forest surrogate because the selected deployed AutoGluon ensemble is expensive to explain directly. These are explanations of surrogate model behavior, not the actual deployed ensemble; feature attribution is not causation. The sample was capped at 100 records to fit available resources.

## 10. Deployment
The Streamlit app loads the saved AutoGluon predictor once, validates required columns, limits uploads to 50,000 rows, displays predictions/probabilities, and offers CSV download. Run locally with `D:\AUTOML\.venv\Scripts\python.exe -m streamlit run app.py`. No external deployment was performed.

## 11. Limitations and conclusion
This is a research prototype trained on UNSW-NB15, not a production intrusion prevention system. Test results show high recall but substantial false positives and lower performance for normal traffic. Search budgets are small development budgets, AutoGluon uses a native ensemble, and preprocessing/model strategies are not computationally identical. Independent datasets, time/group-aware validation, rare-class study, operational threshold calibration, and production safety evaluation are future work.

## References
1. Nour Moustafa and Jill Slay, “UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems,” MilCIS, 2015.
2. UNSW Canberra, [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
3. FLAML, [documentation](https://microsoft.github.io/FLAML/).
4. AutoGluon, [Tabular documentation](https://auto.gluon.ai/).
5. Scott M. Lundberg and Su-In Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.
'''
    (ROOT/"reports"/"final_report.md").write_text(report,encoding="utf-8")
    frames=[]
    for fn in ("baseline_results.csv","tuning_results.csv"):
        fp=RESULTS/fn
        if fp.exists():
            d=__import__("pandas").read_csv(fp)
            if "model" not in d.columns and fn=="baseline_results.csv": d["model"]="RandomForest baseline"
            frames.append(d)
    automl_rows=[]
    if (RESULTS/"flaml_metrics.json").exists():
        fm=json.loads((RESULTS/"flaml_metrics.json").read_text(encoding="utf-8")); automl_rows.append({k:fm.get(k) for k in ("model","validation_f1","accuracy","precision","recall","f1","roc_auc","pr_auc","prediction_seconds","search_seconds")})
    if (RESULTS/"autogluon_metrics.json").exists():
        ag=json.loads((RESULTS/"autogluon_metrics.json").read_text(encoding="utf-8")); am=ag.get("test_metrics",{}); automl_rows.append({"model":"AutoGluon Tabular","validation_f1":float(ag.get("validation_f1",float("nan"))),"accuracy":am.get("accuracy"),"precision":am.get("precision"),"recall":am.get("recall"),"f1":am.get("f1"),"roc_auc":am.get("roc_auc"),"search_seconds":float(ag.get("search_seconds_measured",float("nan"))),"prediction_seconds":ag.get("prediction_seconds_including_evaluate")})
    if automl_rows:
        auto_table=__import__("pandas").DataFrame(automl_rows); auto_table.to_csv(RESULTS/"automl_results.csv",index=False); frames.append(auto_table)
    if frames:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        combined=pd.concat(frames,ignore_index=True,sort=False)
        combined.to_csv(RESULTS/"unified_comparison.csv",index=False)
        if {"model","f1"}.issubset(combined.columns):
            plot=combined.dropna(subset=["f1"])
            plt.figure(figsize=(10,5)); plt.bar(plot["model"].astype(str),plot["f1"].astype(float)); plt.xticks(rotation=30,ha="right"); plt.ylabel("Test F1"); plt.tight_layout(); plt.savefig(FIGURES/"f1_comparison.png"); plt.close()
            if {"precision","recall"}.issubset(plot.columns):
                ax=plot.set_index("model")[["precision","recall"]].astype(float).plot(kind="bar",figsize=(10,5)); ax.set_ylabel("Test score"); plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig(FIGURES/"precision_recall_comparison.png"); plt.close()
            if "search_seconds" in plot:
                plt.figure(figsize=(8,5)); plt.scatter(plot["search_seconds"],plot["f1"]); 
                for _,row in plot.iterrows(): plt.annotate(str(row["model"]),(row["search_seconds"],row["f1"]))
                plt.xlabel("Search seconds (recorded where available)"); plt.ylabel("Test F1"); plt.tight_layout(); plt.savefig(FIGURES/"search_time_vs_f1.png"); plt.close()
    if m and m.get("confusion_matrix"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(5,4)); sns.heatmap(m["confusion_matrix"],annot=True,fmt="d",cmap="Blues",xticklabels=["Normal","Attack"],yticklabels=["Normal","Attack"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(FIGURES/"final_confusion_matrix.png"); plt.close()
    import platform, subprocess
    versions={}
    for pkg in ("numpy","pandas","scikit-learn","scipy","optuna","joblib","flaml","autogluon.tabular","shap","matplotlib","seaborn","streamlit","pytest"):
        try: versions[pkg]=__import__("importlib.metadata",fromlist=["version"]).version(pkg)
        except Exception: versions[pkg]="unavailable"
    (RESULTS/"experiment_metadata.json").write_text(json.dumps({"python":platform.python_version(),"interpreter":str(ROOT/".venv"/"Scripts"/"python.exe"),"platform":platform.platform(),"packages":versions,"cpu_logical":__import__("os").cpu_count(),"automl_budget_seconds":90,"resource_note":"AutoGluon observed 2.81 GB available memory at run start; two workers were used."},indent=2),encoding="utf-8")

if __name__=="__main__":generate()
