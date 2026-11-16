# Generated reports and model evidence

Run the pipeline from the project root to regenerate these artifacts:

```bash
make data
make eda
make train
```

## Files

- [`EDA.md`](EDA.md) summarizes the source row count, cleaning result, target distribution, missingness, experience-level medians, and permutation importance.
- [`salary_histogram.png`](salary_histogram.png) shows the cleaned annual USD target distribution.
- [`feature_importance.png`](feature_importance.png) plots held-out permutation importance from the Ridge(alpha=1) reference model.
- [`feature_importance.csv`](feature_importance.csv) contains each input feature's average increase in held-out MAE and shuffle variability, in USD.
- [`../models/metrics.json`](../models/metrics.json) contains the training/validation/test split, candidate comparison, selected model, and holdout scores.

Permutation importance was computed on at most 2,000 test rows with two shuffles per feature. A positive value means that shuffling the feature increased MAE for this fitted model. It does not establish causality; correlated inputs can share predictive signal. The results are tied to the dataset snapshot recorded in [`../data/source.json`](../data/source.json).

The checked model run selected Ridge(alpha=1), with test MAE $47,217.81 and RMSE $67,735.40. Re-running after an upstream dataset update may produce different values.
