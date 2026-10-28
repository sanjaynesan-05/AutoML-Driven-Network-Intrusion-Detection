# Five-minute presentation outline

1. Problem and motivation (30 sec): benchmark network-flow classification as an analyst aid.
2. Dataset (35 sec): report actual profile and supplied train/test separation.
3. Architecture (35 sec): pipeline, validation and artifact flow; show architecture.mmd.
4. Manual and AutoML experiments (55 sec): Random Forest, Grid/Random/Optuna, FLAML and AutoGluon; show `results/unified_comparison.csv`.
5. Actual results (40 sec): AutoGluon validation F1 0.9780 and test attack F1 0.8853; recall 0.9919 and precision 0.7994. Explain the false-positive tradeoff.
6. Leakage and explainability (40 sec): synthetic target-copy experiment and SHAP limitations.
7. Live demo (45 sec): upload sample CSV, inspect predictions, download results.
8. Conclusion (20 sec): benchmark scope and need for external validation.

## Likely viva questions

- Why exclude `attack_cat`? It is a second target label and would reveal the outcome.
- Why exclude `id`? It is sequential within supplied partitions and is unlikely to represent reusable flow behavior.
- How do you avoid preprocessing leakage? The transformer is part of a pipeline fit on training folds only.
- Why use both F1 and PR-AUC? They surface attack precision/recall and ranking quality under class imbalance.
- Is the intentional leakage result valid performance? No; a copied label is only a diagnostic of leakage.
- Does SHAP prove causal factors? No; it attributes fitted model outputs, not causes.
- Can the model defend production traffic? Not based on this benchmark alone; independent data and operational validation are required.
- Why is test F1 lower than validation F1? The split distribution changes and the supplied test is a separate partition; training-only model selection cannot guarantee the held-out distribution.
