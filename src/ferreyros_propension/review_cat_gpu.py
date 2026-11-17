"""Measured GPU CatBoost extension under the frozen nested temporal protocol."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from .data import CATEGORICAL
from .review import (
    classification_metrics,
    development_data,
    estimator,
    feature_sets,
    fingerprint,
    optimal_threshold,
    predictions,
    suggest_parameters,
    weights,
)


def _identity(data, config):
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract = {"review": fingerprint(data, config), "gpu_source": source}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _fit_gpu(events, features, params, cut, config, directory, prefix):
    """Choose GPU tree count on an earlier stop window, then refit all train."""
    target = events.incidencia_compra
    stop_days = config["early_stopping"]["days"]
    inner = events.dia_visita.le(cut - stop_days)
    stop = events.dia_visita.between(cut - stop_days + 1, cut)
    train = events.dia_visita.le(cut)
    parameters = {**params, "cat_features": [c for c in features if c in CATEGORICAL]}

    def model(iterations):
        return estimator(
            "catboost", parameters, iterations, config, config["seed"]
        ).set_params(task_type="GPU", devices="0", gpu_ram_part=0.7)

    started = perf_counter()
    selected_model = model(config["early_stopping"]["max_trees"])
    selected_model.fit(
        features.loc[inner],
        target[inner],
        sample_weight=weights(events, inner, cut - stop_days, params["half_life"]),
        eval_set=[(features.loc[stop], target[stop])],
        early_stopping_rounds=config["early_stopping"]["rounds"],
    )
    selected = max(1, selected_model.get_best_iteration() + 1)
    executed = len(selected_model.get_evals_result()["learn"]["Logloss"])
    selection_record = {
        "name": f"{prefix}_selection",
        "phase": "selection",
        "training_end": cut - stop_days,
        "stop_end": cut,
        "seconds": perf_counter() - started,
        "selected_rounds": selected,
        "executed_rounds": executed,
        "train_rows": int(inner.sum()),
        "device": "GPU:0",
    }
    (directory / f"{prefix}_selection_fit.json").write_text(
        json.dumps(selection_record, indent=2)
    )
    del selected_model
    started = perf_counter()
    fitted = model(selected)
    fitted.fit(
        features.loc[train],
        target[train],
        sample_weight=weights(events, train, cut, params["half_life"]),
    )
    refit_record = {
        "name": f"{prefix}_refit",
        "phase": "refit",
        "training_end": cut,
        "seconds": perf_counter() - started,
        "selected_rounds": selected,
        "executed_rounds": fitted.tree_count_,
        "train_rows": int(train.sum()),
        "device": "GPU:0",
    }
    (directory / f"{prefix}_refit_fit.json").write_text(
        json.dumps(refit_record, indent=2)
    )
    return fitted, selected, [selection_record, refit_record]


def _run_fold(events, features, groups, cut, end, config, identity, timeout):
    started = perf_counter()
    directory = Path(config["output"]) / f"catboost_gpu_{cut}"
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "fingerprint.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text())["fingerprint"] != identity
    ):
        raise ValueError("GPU CatBoost source/data/config fingerprint changed")
    identity_path.write_text(json.dumps({"fingerprint": identity}, indent=2))
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"catboost_gpu_{cut}_{identity[:12]}",
        storage=f"sqlite:///{(directory / 'study.db').resolve().as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=10),
    )

    def objective(trial):
        group, parameters = suggest_parameters(trial, "catboost", config)
        frames, iterations = [], []
        for fold, (inner_cut, inner_end) in enumerate(config["inner_folds"][cut], 1):
            model, selected, _ = _fit_gpu(
                events,
                features[groups[group]],
                parameters,
                inner_cut,
                config,
                directory,
                f"trial_{trial.number}_fold_{fold}",
            )
            frames.append(
                predictions(
                    events,
                    model,
                    features[groups[group]],
                    inner_cut + 1,
                    inner_end,
                    fold,
                )
            )
            iterations.append(selected)
            del model
        threshold, score, f1 = optimal_threshold(frames)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr(
            "logloss", float(np.mean([log_loss(frame.y, frame.p) for frame in frames]))
        )
        trial.set_user_attr("iterations", iterations)
        pd.concat(frames).to_csv(
            directory / f"trial_{trial.number}_inner.csv", index=False
        )
        print(
            f"CATBOOST_GPU cut={cut} trial={trial.number} score={score:.4f}", flush=True
        )
        return score

    target_trials = config["trials"].get("catboost_gpu", 20)
    remaining = max(0, target_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout)
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError(f"No GPU CatBoost trial completed for fold {cut}")
    best = max(
        completed,
        key=lambda trial: (
            trial.value,
            trial.user_attrs["f1"],
            -trial.user_attrs["logloss"],
        ),
    )
    params = best.user_attrs["parameters"]
    group = best.params["feature_group"]
    threshold = best.user_attrs["threshold"]
    pd.read_csv(directory / f"trial_{best.number}_inner.csv").to_csv(
        directory / "inner_predictions.csv", index=False
    )
    model, selected, _ = _fit_gpu(
        events, features[groups[group]], params, cut, config, directory, "outer"
    )
    rows = []
    for split, start, stop in [("train", 1, cut), ("validation", cut + 1, end)]:
        frame = predictions(events, model, features[groups[group]], start, stop, 0)
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
        rows.append(
            {
                "model": "catboost_gpu",
                "cut": cut,
                "end": end,
                "split": split,
                "threshold": threshold,
                "n": len(frame),
                **classification_metrics(frame.y, frame.p, threshold),
            }
        )
    joblib.dump(
        {
            "model": model,
            "feature_builder_version": "review_v1",
            "columns": groups[group],
            "threshold": threshold,
            "training_end": cut,
        },
        directory / "model.joblib",
    )
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    fit_records = [
        json.loads(path.read_text()) for path in directory.glob("*_fit.json")
    ]
    pd.DataFrame(fit_records).to_csv(directory / "fit_history.csv", index=False)
    result = {
        "model": "catboost_gpu",
        "cut": cut,
        "end": end,
        "fingerprint": identity,
        "parameters": params,
        "feature_group": group,
        "columns": groups[group],
        "threshold": threshold,
        "inner_score": best.value,
        "iterations": selected,
        "trials": len(study.trials),
        "complete_trials": len(completed),
        "actual_fits": len(fit_records),
        "device": "GPU:0",
        "gpu_ram_part": 0.7,
        "executed_rounds": sum(record["executed_rounds"] for record in fit_records),
        "seconds": perf_counter() - started,
        "metrics": rows,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_gpu(config):
    """Run GPU-only studies serially, using a shared 900-second soft budget.

    For a pilot set ``trials.catboost_gpu=1``, restrict ``outer_folds`` and set a
    fresh temporary output directory. No test-period data is accessed here.
    """
    started = perf_counter()
    data = development_data(config)
    events, features, groups = feature_sets(data)
    identity = _identity(data, config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    budget = config["search"].get("catboost_gpu_timeout_seconds", 900)
    results = []
    for index, (cut, end) in enumerate(config["outer_folds"]):
        remaining = max(1, budget - (perf_counter() - started))
        timeout = remaining / (len(config["outer_folds"]) - index)
        result = _run_fold(
            events, features, groups, cut, end, config, identity, timeout
        )
        results.append(result)
        pd.DataFrame([row for item in results for row in item["metrics"]]).to_csv(
            output / "catboost_gpu_metrics.csv", index=False
        )
        print(
            f"DONE catboost_gpu {cut}: {result['complete_trials']} trials, "
            f"P={result['metrics'][-1]['precision']:.4f}, R={result['metrics'][-1]['recall']:.4f}",
            flush=True,
        )
    (output / "catboost_gpu_runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "soft_budget_seconds": budget,
                "fingerprint": identity,
                "gpu_ram_part": 0.7,
                "folds": [
                    {
                        key: result[key]
                        for key in [
                            "cut",
                            "trials",
                            "complete_trials",
                            "actual_fits",
                            "seconds",
                        ]
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )
    return results
