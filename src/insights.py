"""Writes reports/insights.md. Every number comes from reports/analysis_facts.json
(produced by notebooks/04_error_analysis.ipynb) or from the CSVs written by src.train.
"""
import json
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def write_insights() -> str:
    f = json.loads((REPORTS / "analysis_facts.json").read_text())
    t = pd.read_csv(REPORTS / "test_results.csv", index_col="model")
    final = t.iloc[0]
    med = t.loc["A: train median"]
    lin = t.loc["B2: LinearRegression (5 feats, log1p target)"]
    text = f"""# Insights

Every number below is computed by `python -m src.train` and `notebooks/04_error_analysis.ipynb`.

## 1. The model is accurate on typical houses and far ahead of the baselines
On the test set the final model has MAE ${final['MAE']:,.0f} and RMSLE {final['RMSLE']:.3f}, versus
${med['MAE']:,.0f} / {med['RMSLE']:.3f} for the median baseline and ${lin['MAE']:,.0f} / {lin['RMSLE']:.3f} for the
5-feature linear baseline (log1p target). The median house is predicted within {f['test_median_pct_err']:.1f}%
(mean absolute error {f['test_mean_pct_err']:.1f}%). Out-of-fold RMSLE on train ({f['oof_rmsle']:.3f}) agrees with the
test RMSLE, so the test result is not a lucky draw.

## 2. Error grows with price in dollars, but not in percent, and the model pulls predictions toward the middle
Spearman correlation between price and absolute error is {f['rho_abs']:+.2f} in dollars but {f['rho_pct']:+.2f} in
percent. The most expensive fifth of houses has MAE ${f['q_mae_high']:,.0f} versus ${f['q_mae_low']:,.0f} for the
cheapest fifth, yet their median percent errors are similar ({f['q_pct_high']:.1f}% vs {f['q_pct_low']:.1f}%).
The bias is systematic: the cheapest fifth is on average over-predicted by ${-f['q_bias_low']:,.0f} and the most
expensive fifth under-predicted by ${f['q_bias_high']:,.0f}.

## 3. "Less training data, more error" is only partly supported for neighborhoods
Over the {f['nb_n']} neighborhoods with at least 10 training houses (out-of-fold predictions):
median % error vs training count has Spearman rho = {f['nb_rho_oof_med_pct']:+.2f} (p = {f['nb_p_oof_med_pct']:.3f}),
but mean |log error| has rho = {f['nb_rho_oof_log_err']:+.2f} (p = {f['nb_p_oof_log_err']:.2f}), which is not significant.
At house level, houses in neighborhoods with fewer than 30 training houses have a median error of
{f['hs_med']:.1f}% (n = {f['hs_n']}) versus {f['hl_med']:.1f}% (n = {f['hl_n']}) elsewhere: a real but modest gap.
The largest dollar errors are in expensive, heterogeneous neighborhoods ({', '.join(f"{n} (n_train={k}, MAE ${v:,.0f})" for n, v, k in zip(f['top_mae_nb'], f['top_mae_vals'], f['top_mae_ntrain']))}),
and price variability inside a neighborhood correlates with error (rho = {f['nb_rho_cv']:+.2f}, p = {f['nb_p_cv']:.2f}),
so scarcity is confounded with price level. The test set cannot settle this: most neighborhoods have fewer than 8 test houses.

## 4. The worst predictions are cheap houses that sold under unusual conditions
Of the 10 worst test predictions, {f['w_over']} were over-predicted; their median price is ${f['w_price']:,.0f}
(rest of the test set: ${f['r_price']:,.0f}); {f['w_nonnormal']:.0%} had a non-"Normal" SaleCondition (abnormal, partial,
family, allocation) versus {f['r_nonnormal']:.0%} of the other houses; mean OverallQual is {f['w_qual']:.1f} vs {f['r_qual']:.1f}.
These 10 houses (3% of the test set) account for {f['w_sqshare']:.0f}% of the total squared log error. The 30 worst
out-of-fold predictions on train show the same pattern ({f['o_over']:.0%} over-predicted, {f['o_nonnormal']:.0%} non-Normal sales
vs {f['o_rest_nonnormal']:.0%}). SaleCondition is a model input, so this is not a missing column. A plausible but untested explanation:
there are too few such sales for the model to learn how large their discount is, so it predicts closer to a normal market price.

## 5. Size and quality dominate; Neighborhood adds little once they are known
Permutation importance (increase in test RMSLE when a column is shuffled): {', '.join(f"{n} {v:.3f}" for n, v in zip(f['pi_top'], f['pi_top_vals']))}.
The top 3 columns carry {f['pi_top3_share']:.0%} of the total positive importance. Neighborhood ranks
{f['pi_neigh_rank']} with {f['pi_neigh_val']:.4f}. Caveat: permutation importance of correlated columns
(1stFlrSF, 2ndFlrSF, TotalBsmtSF all feed TotalSF) is shared between them, so it understates each one's true role.
"""
    (REPORTS / "insights.md").write_text(text)
    return text


if __name__ == "__main__":
    print(write_insights())
