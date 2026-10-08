# Results

Every number below is produced by `python -m src.train`.

## Protocol

- One fixed 80/20 split (`random_state=42`): 1168 train / 292 test houses. Test set untouched until the final step.
- Cross-validation: 5-fold on the train split; values are mean ± std over folds. Models with tuned hyper-parameters use *nested* CV (the grid search runs inside each fold).
- **Selection rule (fixed before looking at test results):** the final model is the Stage 3 configuration with the lowest mean CV RMSLE over all validation rows.
- **RMSLE and negative predictions:** RMSLE = RMSE of log1p(price). log1p is undefined for values ≤ -1, so negative predictions are clipped at 0 before the log (they count as a $0 prediction). MAE, RMSE and R² use the raw predictions.
- *fit time* = mean seconds per outer fold, including the inner hyper-parameter search.

## Cross-validation results

| model | outliers in training | CV MAE ($) | CV RMSE ($) | CV RMSLE | CV R² | fit time (s) |
|---|---|---|---|---|---|---|
| A: train median | kept | 54,586 ± 2,677 | 78,860 ± 4,112 | 0.390 ± 0.018 | -0.046 ± 0.014 | 0.0 |
| B1: LinearRegression (5 feats) | kept | 24,633 ± 1,375 | 38,673 ± 5,706 | 0.460 ± 0.360 | 0.746 ± 0.060 | 0.0 |
| B2: LinearRegression (5 feats, log1p target) | kept | 22,638 ± 2,200 | 52,626 ± 34,398 | 0.178 ± 0.029 | 0.386 ± 0.758 | 0.0 |
| Ridge | kept | 17,321 ± 1,405 | 40,405 ± 24,427 | 0.140 ± 0.029 | 0.652 ± 0.399 | 2.3 |
| Ridge | removed | 16,409 ± 3,622 | 51,665 ± 46,199 | 0.140 ± 0.042 | 0.265 ± 1.088 | 1.0 |
| Lasso | kept | 17,753 ± 1,555 | 46,555 ± 29,540 | 0.146 ± 0.028 | 0.525 ± 0.550 | 5.4 |
| Lasso | removed | 16,463 ± 3,792 | 55,684 ± 53,644 | 0.140 ± 0.044 | 0.089 ± 1.418 | 4.1 |
| RandomForest | kept | 16,699 ± 1,104 | 28,976 ± 4,503 | 0.137 ± 0.015 | 0.857 ± 0.038 | 16.9 |
| RandomForest | removed | 16,789 ± 1,208 | 29,055 ± 5,343 | 0.138 ± 0.018 | 0.855 ± 0.048 | 18.9 |
| HistGradientBoosting | kept | 15,899 ± 736 | 27,625 ± 4,851 | 0.129 ± 0.016 | 0.870 ± 0.036 | 11.5 |
| HistGradientBoosting | removed | 15,699 ± 1,129 | 27,993 ± 7,403 | 0.129 ± 0.020 | 0.863 ± 0.063 | 11.5 |

Baseline diagnostics (why some stds are large):

| model | worst-fold R² | negative price predictions |
|---|---|---|
| A: train median | -0.07 | 0 |
| B1: LinearRegression (5 feats) | 0.66 | 3 |
| B2: LinearRegression (5 feats, log1p target) | -1.11 | 0 |

The 2 outlier houses (GrLivArea > 4000 sqft and price < $300k) fall in the validation part of fold(s) [2, 3].

## Outlier experiment

The 2 outliers are removed from the **training part of each fold only** (also inside the inner tuning folds). Validation rows are never filtered, so both arms are scored on exactly the same houses. The right-hand columns score the same predictions *excluding* the 2 outliers from validation, i.e. on typical houses.

| model | RMSLE kept | RMSLE removed | MAE kept ($) | MAE removed ($) | RMSLE kept, typical houses | RMSLE removed, typical houses |
|---|---|---|---|---|---|---|
| Ridge | 0.140 | 0.140 | 17,321 | 16,409 | 0.121 | 0.113 |
| Lasso | 0.146 | 0.140 | 17,753 | 16,463 | 0.122 | 0.111 |
| RandomForest | 0.137 | 0.138 | 16,699 | 16,789 | 0.132 | 0.131 |
| HistGradientBoosting | 0.129 | 0.129 | 15,899 | 15,699 | 0.124 | 0.120 |

## Hyper-parameters chosen in each outer fold

| model | outliers | best params per fold |
|---|---|---|
| Ridge | kept | alpha=100; alpha=30; alpha=30; alpha=100; alpha=30 |
| Ridge | removed | alpha=30; alpha=30; alpha=10; alpha=3; alpha=10 |
| Lasso | kept | alpha=0.003; alpha=0.0003; alpha=0.001; alpha=0.01; alpha=0.003 |
| Lasso | removed | alpha=0.001; alpha=0.001; alpha=0.001; alpha=0.0003; alpha=0.001 |
| RandomForest | kept | max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1 |
| RandomForest | removed | max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1; max_features=0.3, min_samples_leaf=1 |
| HistGradientBoosting | kept | learning_rate=0.03, max_depth=3; learning_rate=0.03, max_depth=3; learning_rate=0.03, max_depth=3; learning_rate=0.1, max_depth=3; learning_rate=0.1, max_depth=3 |
| HistGradientBoosting | removed | learning_rate=0.1, max_depth=3; learning_rate=0.1, max_depth=3; learning_rate=0.03, max_depth=3; learning_rate=0.03, max_depth=3; learning_rate=0.1, max_depth=3 |

## Final model

Chosen by the rule above: **HistGradientBoosting**, outliers **kept** (CV RMSLE 0.129). Refitted on the full train split; tuned parameters: `{'learning_rate': 0.1, 'max_depth': 3}`.

### Final test result

Evaluated once on the held-out test set (292 houses). The baseline rows were scored in the same final step for reference only; no decision depends on them.

| model | test MAE ($) | test RMSE ($) | test RMSLE | test R² |
|---|---|---|---|---|
| HistGradientBoosting (final) | 16,391 | 30,980 | 0.136 | 0.875 |
| A: train median | 59,568 | 88,667 | 0.432 | -0.025 |
| B1: LinearRegression (5 feats) | 25,415 | 39,763 | 0.264 | 0.794 |
| B2: LinearRegression (5 feats, log1p target) | 20,742 | 31,905 | 0.174 | 0.867 |
