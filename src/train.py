"""Reproducible entry point: `python -m src.train`.

Stage 2: baselines evaluated with 5-fold CV on the TRAIN split only.
The test set is not touched here.
"""
from pathlib import Path

import pandas as pd

from src.data import load_data, split_data
from src.evaluate import N_SPLITS, cross_validate_model, make_kfold
from src.models import BASELINE_FEATURES, baseline_linear, baseline_median

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def run_baselines(X_train, y_train) -> pd.DataFrame:
    models = {
        "A: train median": baseline_median(),
        "B1: LinearRegression (5 feats)": baseline_linear(log_target=False),
        "B2: LinearRegression (5 feats, log1p target)": baseline_linear(log_target=True),
    }
    rows = []
    for name, model in models.items():
        res = cross_validate_model(model, X_train, y_train)
        rows.append({"model": name, **res})
    return pd.DataFrame(rows)


def outlier_folds(X_train, y_train) -> tuple[list[int], int]:
    """Which CV folds hold the 2 known outliers (huge but cheap) in their validation part."""
    mask = ((X_train["GrLivArea"] > 4000) & (y_train < 300_000)).to_numpy()
    folds = [i for i, (_, va) in enumerate(make_kfold().split(X_train), 1) if mask[va].any()]
    return folds, int(mask.sum())


def to_markdown(df: pd.DataFrame, n_train: int, folds: list[int], n_out: int) -> str:
    lines = [
        "# Results",
        "",
        "## Stage 2 — Baselines",
        "",
        f"{N_SPLITS}-fold CV on the training split ({n_train} houses, random_state=42). "
        "Values are mean ± std over folds. The test set has not been used.",
        f"Features for B: {', '.join(BASELINE_FEATURES)}.",
        "",
        "| model | CV MAE ($) | CV RMSE ($) | CV RMSLE | CV R² |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['MAE']:,.0f} ± {r['MAE_std']:,.0f} "
            f"| {r['RMSE']:,.0f} ± {r['RMSE_std']:,.0f} "
            f"| {r['RMSLE']:.3f} ± {r['RMSLE_std']:.3f} "
            f"| {r['R2']:.3f} ± {r['R2_std']:.3f} |"
        )
    lines += [
        "",
        "Diagnostics (why some stds are large):",
        "",
        "| model | worst-fold R² | negative price predictions |",
        "|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | {r['R2_worst_fold']:.2f} | {int(r['n_negative_preds'])} |")
    lines += [
        "",
        f"The {n_out} outlier houses (GrLivArea > 4000 sqft and price < $300k) fall in the validation "
        f"part of fold(s) {folds}. Large stds of the linear baselines come from a few extreme "
        "predictions (negative prices in B1, expm1 blow-ups in B2); RMSLE clips negative "
        "predictions to 0.",
    ]
    return "\n".join(lines) + "\n"


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train {X_train.shape}, test {X_test.shape} (test untouched)")
    res = run_baselines(X_train, y_train)
    folds, n_out = outlier_folds(X_train, y_train)
    md = to_markdown(res, len(X_train), folds, n_out)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
