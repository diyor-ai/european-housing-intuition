from src import train


def test_sample_run_end_to_end_and_writes_nothing():
    results_file = train.REPORTS / "results.md"
    before = results_file.stat().st_mtime if results_file.exists() else None
    out = train.run(sample=True)
    assert set(out["cv"]["model"]) >= {"Ridge", "Lasso", "RandomForest", "HistGradientBoosting"}
    assert out["choice"]["stage"] == "full"
    final = out["final"]["table"]
    assert all(m["RMSLE"] > 0 for m in final.values())
    # the tuned model must beat the median baseline even on 300 houses
    assert final[next(iter(final))]["MAE"] < final["A: train median"]["MAE"]
    after = results_file.stat().st_mtime if results_file.exists() else None
    assert before == after
