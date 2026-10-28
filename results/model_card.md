# Model card: UNSW-NB15 binary intrusion classifier

- Model: AutoGluon Tabular
- Version: 1.0.0
- Intended purpose/users: academic research and teaching prototype for classifying UNSW-NB15 flow records.
- Dataset: UNSW-NB15 official train/test partitions.
- Target: label (0 normal, 1 attack).
- Preprocessing: training-fitted numeric median imputation; categorical most-frequent imputation and one-hot encoding; unknown categories ignored.
- Excluded fields: label, attack_cat, id.
- Evaluation protocol: supplied test partition held out from fitting and tuning; model selected using training-only validation F1 0.9779582366589328.
- Measured metrics: Accuracy 0.8585; precision 0.7994; recall 0.9919; F1 0.8853.
- Limitations: dataset-specific, class representation and collection bias; no live network validation; SHAP does not establish causation.
- Leakage risks: target-derived category and identifiers are excluded; intentional leakage is an isolated diagnostic.
- Intended use: offline research and classroom demonstrations. Prohibited: autonomous blocking, incident adjudication, or claims of production suitability.
- Safety: model is trained on UNSW-NB15 and is not validated for production network defense unless further testing establishes that suitability. Predictions are estimates, not proof of an attack.
