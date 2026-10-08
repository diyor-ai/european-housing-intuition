"""Reproducible entry point: `python -m src.train [--sample]`.

Protocol (fixed BEFORE any test-set result was looked at):
  1. One 80/20 split (random_state=42). The test set is untouched until step 4.
  2. Every configuration is scored by nested 5-fold CV on the TRAIN split
     (hyper-parameters tuned by an inner 5-fold CV inside each outer fold).
  3. SELECTION RULE: the final model is the Stage 3 configuration (model x outlier
     handling) with the lowest mean CV RMSLE over all validation rows.
  4. The chosen configuration is refitted on the full train split and evaluated ONCE
     on the test set. Baselines are evaluated on it in the same final step, for
     reference only (they are not used for any decision).

`--sample` runs the same code path on 300 houses with tiny grids and writes nothing.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from src.data import load_data, outlier_mask, split_data
from src.evaluate import N_SPLITS, all_metrics, cross_validate_model, make_kfold
from src.models import (
    BASELINE_FEATURES, MODEL_NAMES, baseline_linear, baseline_median, make_model,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

BASELINES = {
    "A: train median": baseline_median,
    "B1: LinearRegression (5 feats)": lambda: baseline_linear(log_target=False),
    "B2: LinearRegression (5 feats, log1p target)": lambda: baseline_linear(log_target=True),
}


def run_cv(X_train, y_train, fast: bool, n_splits: int) -> pd.DataFrame:
    rows = []
    for name, make in BASELINES.items():
        res = cross_validate_model(make(), X_train, y_train, n_splits)
        rows.append({"model": name, "stage": "baseline", "drop_outliers": False, **res})
    for name in MODEL_NAMES:
        for drop in (False, True):
            print(f"  CV: {name} (outliers {'removed from train folds' if drop else 'kept'})", flush=True)
            res = cross_validate_model(make_model(name, drop, fast, n_splits), X_train, y_train, n_splits)
            rows.append({"model": name, "stage": "full", "drop_outliers": drop, **res})
    return pd.DataFrame(rows)


def select_final(cv: pd.DataFrame) -> pd.Series:
    """Selection rule: lowest mean CV RMSLE among the Stage 3 configurations."""
    full = cv[cv["stage"] == "full"]
    return full.loc[full["RMSLE"].idxmin()]


def outlier_folds(X_train, y_train, n_splits: int):
    mask = outlier_mask(X_train, y_train).to_numpy()
    folds = [i for i, (_, va) in enumerate(make_kfold(n_splits).split(X_train), 1) if mask[va].any()]
    return folds, int(mask.sum())


def final_evaluation(choice, cv, X_train, X_test, y_train, y_test, fast, n_splits, save):
    """The single test-set evaluation."""
    model = make_model(choice["model"], bool(choice["drop_outliers"]), fast, n_splits)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    table = {f"{choice['model']} (final)": all_metrics(y_test, pred)}
    for name, make in BASELINES.items():  # reference only
        b = make().fit(X_train, y_train)
        table[name] = all_metrics(y_test, b.predict(X_test))
    result = {"table": table, "best_params": model.best_params_, "n_test": len(y_test)}
    if save:
        # out-of-fold predictions on train with the chosen hyper-parameters (for error analysis)
        oof = cross_val_predict(clone(model.best_estimator_), X_train, y_train, cv=make_kfold(n_splits))
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model.best_estimator_, MODELS_DIR / "final_model.joblib")
        pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred,
                      "Neighborhood": X_test["Neighborhood"].to_numpy()},
                     index=X_test.index).to_csv(REPORTS / "predictions_test.csv", index_label="row")
        pd.DataFrame({"actual": y_train.to_numpy(), "predicted": oof,
                      "Neighborhood": X_train["Neighborhood"].to_numpy()},
                     index=X_train.index).to_csv(REPORTS / "predictions_oof_train.csv", index_label="row")
    return result


def _pm(r, k, fmt):
    return f"{fmt.format(r[k])} ± {fmt.format(r[k + '_std'])}"


def to_markdown(cv, choice, final, n_train, n_splits, folds, n_out) -> str:
    L = ["# Results", "",
         "Every number below is produced by `python -m src.train`.", "",
         "## Protocol", "",
         "- One fixed 80/20 split (`random_state=42`): "
         f"{n_train} train / {final['n_test']} test houses. Test set untouched until the final step.",
         f"- Cross-validation: {n_splits}-fold on the train split; values are mean ± std over folds. "
         "Models with tuned hyper-parameters use *nested* CV (the grid search runs inside each fold).",
         "- **Selection rule (fixed before looking at test results):** the final model is the "
         "Stage 3 configuration with the lowest mean CV RMSLE over all validation rows.",
         "- **RMSLE and negative predictions:** RMSLE = RMSE of log1p(price). log1p is undefined "
         "for values ≤ -1, so negative predictions are clipped at 0 before the log (they count as "
         "a $0 prediction). MAE, RMSE and R² use the raw predictions.",
         "- *fit time* = mean seconds per outer fold, including the inner hyper-parameter search.", "",
         "## Cross-validation results", "",
         "| model | outliers in training | CV MAE ($) | CV RMSE ($) | CV RMSLE | CV R² | fit time (s) |",
         "|---|---|---|---|---|---|---|"]
    for _, r in cv.iterrows():
        o = "removed" if r["drop_outliers"] else "kept"
        if r["stage"] == "baseline":
            o = "kept"
        L.append(f"| {r['model']} | {o} | {_pm(r, 'MAE', '{:,.0f}')} | {_pm(r, 'RMSE', '{:,.0f}')} "
                 f"| {_pm(r, 'RMSLE', '{:.3f}')} | {_pm(r, 'R2', '{:.3f}')} | {r['fit_time']:.1f} |")
    L += ["", "Baseline diagnostics (why some stds are large):", "",
          "| model | worst-fold R² | negative price predictions |", "|---|---|---|"]
    for _, r in cv[cv["stage"] == "baseline"].iterrows():
        L.append(f"| {r['model']} | {r['R2_worst_fold']:.2f} | {int(r['n_negative_preds'])} |")
    L += ["", f"The {n_out} outlier houses (GrLivArea > 4000 sqft and price < $300k) fall in the "
          f"validation part of fold(s) {folds}.", "",
          "## Outlier experiment", "",
          "The 2 outliers are removed from the **training part of each fold only** (also inside the "
          "inner tuning folds). Validation rows are never filtered, so both arms are scored on "
          "exactly the same houses. The right-hand columns score the same predictions "
          "*excluding* the 2 outliers from validation, i.e. on typical houses.", "",
          "| model | RMSLE kept | RMSLE removed | MAE kept ($) | MAE removed ($) "
          "| RMSLE kept, typical houses | RMSLE removed, typical houses |",
          "|---|---|---|---|---|---|---|"]
    full = cv[cv["stage"] == "full"]
    for name in MODEL_NAMES:
        k = full[(full["model"] == name) & ~full["drop_outliers"]].iloc[0]
        d = full[(full["model"] == name) & full["drop_outliers"]].iloc[0]
        L.append(f"| {name} | {k['RMSLE']:.3f} | {d['RMSLE']:.3f} | {k['MAE']:,.0f} | {d['MAE']:,.0f} "
                 f"| {k['RMSLE_no_outliers']:.3f} | {d['RMSLE_no_outliers']:.3f} |")
    L += ["", "## Hyper-parameters chosen in each outer fold", "",
          "| model | outliers | best params per fold |", "|---|---|---|"]
    for _, r in full.iterrows():
        ps = "; ".join(", ".join(f"{k.split('__')[-1]}={v}" for k, v in p.items()) for p in r["best_params"])
        L.append(f"| {r['model']} | {'removed' if r['drop_outliers'] else 'kept'} | {ps} |")
    L += ["", "## Final model", "",
          f"Chosen by the rule above: **{choice['model']}**, outliers "
          f"**{'removed from training' if choice['drop_outliers'] else 'kept'}** "
          f"(CV RMSLE {choice['RMSLE']:.3f}). Refitted on the full train split; tuned "
          f"parameters: `{ {k.split('__')[-1]: v for k, v in final['best_params'].items()} }`.", "",
          "### Final test result", "",
          f"Evaluated once on the held-out test set ({final['n_test']} houses). The baseline rows "
          "were scored in the same final step for reference only; no decision depends on them.", "",
          "| model | test MAE ($) | test RMSE ($) | test RMSLE | test R² |", "|---|---|---|---|---|"]
    for name, m in final["table"].items():
        L.append(f"| {name} | {m['MAE']:,.0f} | {m['RMSE']:,.0f} | {m['RMSLE']:.3f} | {m['R2']:.3f} |")
    return "\n".join(L) + "\n"


def run(sample: bool = False) -> dict:
    df = load_data()
    if sample:
        df = df.sample(300, random_state=42)
    n_splits = 3 if sample else N_SPLITS
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train {X_train.shape}, test {X_test.shape} (test untouched until the final step)")
    cv = run_cv(X_train, y_train, fast=sample, n_splits=n_splits)
    choice = select_final(cv)
    print(f"Selected: {choice['model']} (outliers removed: {choice['drop_outliers']}), "
          f"CV RMSLE {choice['RMSLE']:.3f}")
    final = final_evaluation(choice, cv, X_train, X_test, y_train, y_test, sample, n_splits, save=not sample)
    folds, n_out = outlier_folds(X_train, y_train, n_splits)
    md = to_markdown(cv, choice, final, len(X_train), n_splits, folds, n_out)
    if not sample:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "results.md").write_text(md)
        cv.drop(columns=["best_params"]).to_csv(REPORTS / "cv_results.csv", index=False)
    print(md)
    return {"cv": cv, "choice": choice, "final": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="fast end-to-end run on 300 houses, writes nothing")
    run(sample=ap.parse_args().sample)


if __name__ == "__main__":
    main()
