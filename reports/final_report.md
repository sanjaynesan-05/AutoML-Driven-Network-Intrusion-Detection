# AutoML-Driven Network Intrusion Detection and Cyberattack Classification

## Abstract
This academic prototype compares Random Forest, three hyperparameter-search strategies, FLAML, and AutoGluon on the supplied UNSW-NB15 train/test partitions. The final model was chosen by internal validation F1 and evaluated once on the supplied test set. AutoGluon WeightedEnsemble_L2 led validation F1 at 0.9780; on the held-out test partition it achieved attack F1 0.8853, precision 0.7994, and recall 0.9919. This benchmark does not establish live-network suitability.

## 1. Introduction and problem statement
Network-flow classification can help analysts prioritize review, but benchmark scores do not establish operational detection quality. The research question is whether automated search improves the precision/recall tradeoff over a reproducible manual model on the supplied split.

## 2. Objectives
Inspect the supplied dataset, compare manual and AutoML methods fairly, isolate target leakage, assess class and partition differences, explain model behavior, and deliver a local upload-and-predict prototype.

## 3. Dataset
The training file contains 175,341 rows and 45 columns; the held-out test file contains 82,332 rows and 45 columns. Training attacks: {'1': 119341, '0': 56000}; test attacks: {'1': 45332, '0': 37000}. The supplied feature dictionary is parsed separately and is not used as a model table. Official source: [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Full schema, missingness, infinities, duplicates, type counts, and numeric KS shifts are in `results/dataset_profile.json`.

## 4. Features and preprocessing
The binary target is `label` (0 normal, 1 attack). `label` and target-derived `attack_cat` are excluded; sequential `id` is excluded because it is partition-local and not a reusable traffic feature. The model uses 42 permitted traffic fields. Numeric infinity conversion and median imputation, categorical mode imputation, and one-hot encoding with unknown-category support are fitted inside a sklearn pipeline on training data only.

## 5. Experimental methodology
The supplied test partition was not used for feature selection or search. Tuning uses a seeded 50,000-row training sample and two stratified folds; GridSearchCV evaluated 8 configurations, RandomizedSearchCV evaluated a fixed random budget, and Optuna used seeded TPE with eight trials. The Random Forest baseline also has an independent stratified training-only validation split. FLAML and AutoGluon used independent training validation splits and nominal 90-second development budgets. AutoGluon reported 92.66 seconds total wall time (search budget plus overhead); FLAML search took 89.87 seconds. FLAML's installed API did not expose detailed per-trial history, so its CSV records the selected estimator and configuration summary. AutoGluon skipped optional CatBoost and FastAI models because those add-on packages were not installed; its available LightGBM, Random Forest, ExtraTrees, KNN, XGBoost, and weighted ensemble ran. Equal wall-clock limits do not equate CPU/memory allocation or search methods.

## 6. Results
| Model | Validation F1 | Test accuracy | Attack precision | Attack recall | Attack F1 | ROC-AUC | Search/training notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Random Forest baseline | 0.9638 | 0.8866 | 0.8395 | 0.9817 | 0.9051 | 0.9839 | Fit 89.3s on full training partition; validation trained on 16k sampled records |
| Grid Search | See tuning CSV | 0.8710 | 0.8169 | 0.9870 | 0.8939 | 0.9803 | Search 67.95s, 8 configurations |
| Random Search | See tuning CSV | 0.8886 | 0.8436 | 0.9792 | 0.9064 | 0.9830 | Search 71.82s, 8 configurations |
| Optuna TPE | See tuning CSV | 0.8699 | 0.8156 | 0.9869 | 0.8931 | 0.9799 | Search 50.20s, 8 trials |
| FLAML | 0.9728 | 0.8663 | 0.8161 | 0.9775 | 0.8895 | 0.9803 | 89.87s; selected lgbm |
| AutoGluon WeightedEnsemble_L2 | 0.9780 | 0.8585 | 0.7994 | 0.9919 | 0.8853 | 0.9833 | 90s nominal budget; 92.66s observed total |

The baseline training F1 was 0.9939 versus test F1 0.9051, indicating a generalization gap. The final ensemble favored high attack recall at the cost of lower attack precision and many false positives; its test confusion matrix is [[25719, 11281], [368, 44964]].

## 7. Data leakage experiment
Valid sampled-training diagnostic F1: 0.9608992218272648. Synthetic exact target-copy F1: 1.0. The intentionally leaked feature is explicitly invalid and never entered model comparison or deployment. The supplied `id` and target-derived `attack_cat` are excluded from valid binary inputs.

## 8. Class imbalance and generalization
The training partition has 119,341 attacks and 56,000 normal records; the test partition has 45,332 attacks and 37,000 normal records, so class prevalence differs materially. Training categories include 130 Worms records; test has 44. Backdoor, Shellcode, and Worms are rare in training; Analysis, Backdoor, Shellcode, and Worms are below 1% in test. Category-level recall may therefore be unstable. No demographic fairness claim is made. Exact duplicate feature rows across supplied partitions: 0. Numeric KS statistics are in the profile; `id` shows the strongest shift but is excluded from the model. The supplied flow tables contain no raw source/destination address or timestamp fields for group/temporal leakage tests; other shifts and possible related-flow effects constrain interpretation.

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
