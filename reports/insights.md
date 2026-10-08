# Insights

Every number below is computed by `python -m src.train` and `notebooks/04_error_analysis.ipynb`.

## 1. The model is accurate on typical houses and far ahead of the baselines
On the test set the final model has MAE $16,391 and RMSLE 0.136, versus
$59,568 / 0.432 for the median baseline and $20,742 / 0.174 for the
5-feature linear baseline (log1p target). The median house is predicted within 5.9%
(mean absolute error 9.3%). Out-of-fold RMSLE on train (0.128) agrees with the
test RMSLE, so the test result is not a lucky draw.

## 2. Error grows with price in dollars, but not in percent, and the model pulls predictions toward the middle
Spearman correlation between price and absolute error is +0.27 in dollars but -0.04 in
percent. The most expensive fifth of houses has MAE $32,900 versus $11,682 for the
cheapest fifth, yet their median percent errors are similar (7.5% vs 8.0%).
The bias is systematic: the cheapest fifth is on average over-predicted by $5,724 and the most
expensive fifth under-predicted by $12,669.

## 3. "Less training data, more error" is only partly supported for neighborhoods
Over the 22 neighborhoods with at least 10 training houses (out-of-fold predictions):
median % error vs training count has Spearman rho = -0.65 (p = 0.001),
but mean |log error| has rho = -0.19 (p = 0.39), which is not significant.
At house level, houses in neighborhoods with fewer than 30 training houses have a median error of
7.7% (n = 169) versus 5.9% (n = 999) elsewhere: a real but modest gap.
The largest dollar errors are in expensive, heterogeneous neighborhoods (NoRidge (n_train=33, MAE $46,257), StoneBr (n_train=20, MAE $35,041), NridgHt (n_train=61, MAE $25,151)),
and price variability inside a neighborhood correlates with error (rho = +0.45, p = 0.03),
so scarcity is confounded with price level. The test set cannot settle this: most neighborhoods have fewer than 8 test houses.

## 4. The worst predictions are cheap houses that sold under unusual conditions
Of the 10 worst test predictions, 9 were over-predicted; their median price is $73,000
(rest of the test set: $154,950); 60% had a non-"Normal" SaleCondition (abnormal, partial,
family, allocation) versus 18% of the other houses; mean OverallQual is 5.3 vs 6.0.
These 10 houses (3% of the test set) account for 49% of the total squared log error. The 30 worst
out-of-fold predictions on train show the same pattern (80% over-predicted, 53% non-Normal sales
vs 17%). SaleCondition is a model input, so this is not a missing column. A plausible but untested explanation:
there are too few such sales for the model to learn how large their discount is, so it predicts closer to a normal market price.

## 5. Size and quality dominate; Neighborhood adds little once they are known
Permutation importance (increase in test RMSLE when a column is shuffled): OverallQual 0.066, 2ndFlrSF 0.050, 1stFlrSF 0.049, TotalBsmtSF 0.046, YearBuilt 0.011.
The top 3 columns carry 53% of the total positive importance. Neighborhood ranks
14 with 0.0045. Caveat: permutation importance of correlated columns
(1stFlrSF, 2ndFlrSF, TotalBsmtSF all feed TotalSF) is shared between them, so it understates each one's true role.
