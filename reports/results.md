# Results

## Stage 2 — Baselines

5-fold CV on the training split (1168 houses, random_state=42). Values are mean ± std over folds. The test set has not been used.
Features for B: OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt.

| model | CV MAE ($) | CV RMSE ($) | CV RMSLE | CV R² |
|---|---|---|---|---|
| A: train median | 54,586 ± 2,677 | 78,860 ± 4,112 | 0.390 ± 0.018 | -0.046 ± 0.014 |
| B1: LinearRegression (5 feats) | 24,633 ± 1,375 | 38,673 ± 5,706 | 0.460 ± 0.360 | 0.746 ± 0.060 |
| B2: LinearRegression (5 feats, log1p target) | 22,638 ± 2,200 | 52,626 ± 34,398 | 0.178 ± 0.029 | 0.386 ± 0.758 |

Diagnostics (why some stds are large):

| model | worst-fold R² | negative price predictions |
|---|---|---|
| A: train median | -0.07 | 0 |
| B1: LinearRegression (5 feats) | 0.66 | 3 |
| B2: LinearRegression (5 feats, log1p target) | -1.11 | 0 |

The 2 outlier houses (GrLivArea > 4000 sqft and price < $300k) fall in the validation part of fold(s) [2, 3]. Large stds of the linear baselines come from a few extreme predictions (negative prices in B1, expm1 blow-ups in B2); RMSLE clips negative predictions to 0.
