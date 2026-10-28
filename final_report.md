# AutoML-Driven Network Intrusion Detection and Cyberattack Classification

## Abstract
This project evaluates binary classification of normal and attack traffic using the supplied UNSW-NB15 partitions, comparing manual, tuned and AutoML methods. Results below are populated only from local generated artifacts.

## Introduction and problem statement
Network-flow classification can support analysis, but benchmark performance does not establish operational detection quality. The research question is whether automated model search improves the precision/recall tradeoff over a reproducible manual baseline on the supplied split.

## Objectives and dataset
Training profile: [175341, 45]. Test profile: [82332, 45]. Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset. Training and testing partitions are preserved.

## Feature engineering and preprocessing
Target-derived fields `label` and `attack_cat`, plus partition-local sequential `id`, are excluded from the binary input schema. Imputation and one-hot encoding are fitted through the training pipeline only.

## Methodology and metrics
Random Forest is the manual baseline. GridSearchCV, RandomizedSearchCV and Optuna TPE use training-only folds. FLAML and AutoGluon use internal training validation and configured time budgets. Accuracy, attack precision/recall/F1, ROC-AUC, PR-AUC, runtime and latency are recorded when available.

## Experiments and actual results
Baseline: completed; see generated results. Tuning: completed; see generated results. FLAML: completed; see generated results. AutoGluon: completed; see generated results. Final selection/test: Accuracy 0.8585; precision 0.7994; recall 0.9919; F1 0.8853.

## Leakage, overfitting, generalization, and imbalance
The isolated experiment copies the target into a synthetic feature and contrasts that score with a valid run. Supplied partition duplicate counts and class/category distributions are in `dataset_profile.json`; rare-category results require per-class recall review. This split may exhibit distribution shift; performance does not imply generalization to other networks or live traffic.

## Explainability
SHAP plots and one local explanation are generated for a supported tree model. Contributions describe fitted model behavior and are not causal evidence.

## Deployment and limitations
Run locally with Streamlit. Uploads must contain the required model input columns and are capped at 50,000 rows. Predictions are research estimates, not proof of attacks.

## Conclusion and future work
Use the generated comparison table to assess the validation-based winner and its held-out performance. Future study should include independent datasets, temporal/group split audits, and live operational validation.

## References
- Moustafa, N. and Slay, J. “UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems,” MilCIS, 2015.
- UNSW Canberra, [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- FLAML documentation: https://microsoft.github.io/FLAML/
- AutoGluon documentation: https://auto.gluon.ai/
- Lundberg, S. M. and Lee, S.-I. “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.
