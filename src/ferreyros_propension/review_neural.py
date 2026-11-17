"""Reproducible nested selection of neural tabular models, development only."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import log_loss

from .neural_training import fit_neural
from .review import (
    classification_metrics,
    development_data,
    feature_sets,
    fingerprint,
    optimal_threshold,
    weights,
)


class NeuralPredictor:
    """CPU prediction interface matching the classic review artifacts."""

    def __init__(self, artifact):
        self.model = artifact["model"].cpu().eval()
        self.preprocessing = artifact["preprocessing"]
        self.columns = artifact["columns"]
        self.architecture = artifact["architecture"]

    @torch.no_grad()
    def predict_proba(self, frame):
        transformed = self.preprocessing.transform(frame[self.columns]).astype(
            np.float32
        )
        values = torch.as_tensor(transformed)
        parts = []
        for batch in values.split(2048):
            logits = (
                self.model(batch, None)
                if self.architecture == "tabm"
                else self.model(batch)
            )
            logits = logits.squeeze(-1) if self.architecture == "tabm" else logits
            parts.append(logits.sigmoid().mean(dim=1).numpy())
        probability = np.concatenate(parts) if parts else np.empty(0)
        return np.column_stack([1 - probability, probability])


def _identity(data, config):
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ["neural_training.py", "review_neural.py"]
    }
    contract = {"review": fingerprint(data, config), "neural_sources": sources}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _suggest(trial):
    group = trial.suggest_categorical(
        "feature_group", ["all", "enhanced", "history_offers"]
    )
    parameters = {
        "architecture": trial.suggest_categorical("architecture", ["tabm", "mlp"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.003, log=True),
        "width": trial.suggest_categorical("width", [64, 128, 256]),
        "depth": trial.suggest_int("depth", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "quantile_transform": trial.suggest_categorical(
            "quantile_transform", [False, True]
        ),
        "half_life": trial.suggest_float("half_life", 90, 720, log=True),
        "k": 16,
    }
    return group, parameters


def _training_config(parameters, config):
    stopping = config["early_stopping"]
    return {
        **parameters,
        "max_epochs": stopping["max_epochs"],
        "min_epochs": stopping["min_epochs"],
        "patience": stopping["patience"],
        "min_delta": 0.0001,
        "threads": config["threads_per_model"],
        "precision": "bf16" if torch.cuda.is_available() else "32",
    }


def _save_fit(result, directory, prefix, records, config):
    pd.DataFrame(result["curves"]).to_csv(
        directory / f"{prefix}_curves.csv", index=False
    )
    record = {
        "name": prefix,
        "architecture": config["architecture"],
        "best_epochs": result["best_epochs"],
        "executed_epochs": result["executed_epochs"],
        "seconds": result["seconds"],
        "stop_reason": result["stop_reason"],
        "parameters": result["parameters"],
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        "device": result["device"],
    }
    records.append(record)
    (directory / f"{prefix}_fit.json").write_text(json.dumps(record, indent=2))
    return record


def _frame(events, mask, probability, fold):
    frame = events.loc[mask, ["id", "dia_visita"]].copy()
    frame["y"] = events.loc[mask, "incidencia_compra"]
    frame["p"] = probability
    frame["inner_fold"] = fold
    return frame


def _fit_window(
    events,
    features,
    parameters,
    cut,
    end,
    config,
    directory,
    prefix,
    records,
    artifact=False,
):
    stop_days = config["early_stopping"]["days"]
    inner = events.dia_visita.le(cut - stop_days)
    stop = events.dia_visita.between(cut - stop_days + 1, cut)
    train = events.dia_visita.le(cut)
    valid = events.dia_visita.between(cut + 1, end)
    y = events.incidencia_compra
    settings = _training_config(parameters, config)
    settings["sample_weight"] = weights(
        events, inner, cut - stop_days, parameters["half_life"]
    )
    selection = fit_neural(
        features.loc[inner],
        y[inner],
        features.loc[stop],
        y[stop],
        {},
        settings,
        config["seed"],
    )
    _save_fit(selection, directory, f"{prefix}_selection", records, settings)
    settings["fixed_epochs"] = selection["best_epochs"]
    settings["sample_weight"] = weights(events, train, cut, parameters["half_life"])
    settings["return_artifact"] = artifact
    frames = {"validation": features.loc[valid]}
    if artifact:
        frames["train"] = features.loc[train]
    fitted = fit_neural(
        features.loc[train], y[train], None, None, frames, settings, config["seed"]
    )
    _save_fit(fitted, directory, f"{prefix}_refit", records, settings)
    return fitted


def _run_fold(events, features, groups, cut, end, config, identity, timeout):
    started = perf_counter()
    directory = Path(config["output"]) / f"neural_{cut}"
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "fingerprint.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if previous["fingerprint"] != identity:
            raise ValueError(
                "Neural results have a different data/config/source fingerprint"
            )
    else:
        identity_path.write_text(json.dumps({"fingerprint": identity}, indent=2))
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["fingerprint"] != identity:
            raise ValueError("Neural result fingerprint does not match")
        return result
    records = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"neural_{cut}_{identity[:12]}",
        storage=f"sqlite:///{(directory / 'study.db').resolve().as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=10),
    )

    def objective(trial):
        group, parameters = _suggest(trial)
        frames = []
        fit_records = []
        for fold, (inner_cut, inner_end) in enumerate(config["inner_folds"][cut], 1):
            fitted = _fit_window(
                events,
                features[groups[group]],
                parameters,
                inner_cut,
                inner_end,
                config,
                directory,
                f"trial_{trial.number}_fold_{fold}",
                fit_records,
            )
            mask = events.dia_visita.between(inner_cut + 1, inner_end)
            frames.append(
                _frame(events, mask, fitted["probabilities"]["validation"], fold)
            )
        threshold, score, f1 = optimal_threshold(frames)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr(
            "logloss", float(np.mean([log_loss(f.y, f.p) for f in frames]))
        )
        trial.set_user_attr("fits", fit_records)
        pd.concat(frames).to_csv(
            directory / f"trial_{trial.number}_inner.csv", index=False
        )
        print(
            f"NEURAL cut={cut} trial={trial.number} arch={parameters['architecture']} score={score:.4f}",
            flush=True,
        )
        return score

    remaining = max(0, config["trials"]["neural"] - len(study.trials))
    if remaining:
        study.optimize(
            objective, n_trials=remaining, timeout=timeout, catch=(FloatingPointError,)
        )
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError(f"No completed neural trial for outer fold {cut}")
    best = max(
        complete,
        key=lambda trial: (
            trial.value,
            trial.user_attrs["f1"],
            -trial.user_attrs["logloss"],
        ),
    )
    parameters = best.user_attrs["parameters"]
    group = best.params["feature_group"]
    threshold = best.user_attrs["threshold"]
    inner = pd.read_csv(directory / f"trial_{best.number}_inner.csv")
    inner.to_csv(directory / "inner_predictions.csv", index=False)
    fitted = _fit_window(
        events,
        features[groups[group]],
        parameters,
        cut,
        end,
        config,
        directory,
        "outer",
        records,
        artifact=True,
    )
    rows = []
    for split, mask in [
        ("train", events.dia_visita.le(cut)),
        ("validation", events.dia_visita.between(cut + 1, end)),
    ]:
        frame = _frame(events, mask, fitted["probabilities"][split], 0)
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
        rows.append(
            {
                "model": "neural",
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
            "model": NeuralPredictor(fitted["artifact"]),
            "feature_builder_version": "review_v1",
            "columns": groups[group],
            "threshold": threshold,
            "training_end": cut,
        },
        directory / "model.joblib",
    )
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    all_fit_files = list(directory.glob("*_fit.json"))
    fit_history = [json.loads(path.read_text()) for path in all_fit_files]
    pd.DataFrame(fit_history).to_csv(directory / "fit_history.csv", index=False)
    result = {
        "model": "neural",
        "cut": cut,
        "end": end,
        "fingerprint": identity,
        "parameters": parameters,
        "feature_group": group,
        "columns": groups[group],
        "threshold": threshold,
        "inner_score": best.value,
        "iterations": fitted["best_epochs"],
        "trials": len(study.trials),
        "complete_trials": len(complete),
        "actual_fits": len(fit_history),
        "executed_epochs": sum(record["executed_epochs"] for record in fit_history),
        "outer_selected_epochs": fitted["best_epochs"],
        "outer_refit_executed_epochs": fitted["executed_epochs"],
        "peak_gpu_memory_bytes": max(
            record["peak_gpu_memory_bytes"] for record in fit_history
        ),
        "seconds": perf_counter() - started,
        "metrics": rows,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_neural(config):
    """Run three nested neural studies sequentially on one GPU, with a soft budget."""
    started = perf_counter()
    data = development_data(config)
    events, features, groups = feature_sets(data)
    identity = _identity(data, config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    budget = config["search"]["neural_timeout_seconds"]
    results = []
    for index, (cut, end) in enumerate(config["outer_folds"]):
        remaining = max(1, budget - (perf_counter() - started))
        timeout = remaining / (len(config["outer_folds"]) - index)
        result = _run_fold(
            events, features, groups, cut, end, config, identity, timeout
        )
        results.append(result)
        pd.DataFrame([row for item in results for row in item["metrics"]]).to_csv(
            output / "neural_metrics.csv", index=False
        )
        print(
            f"DONE neural {cut}: {result['complete_trials']} complete trials, "
            f"P={result['metrics'][-1]['precision']:.4f}, R={result['metrics'][-1]['recall']:.4f}",
            flush=True,
        )
    (output / "neural_runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "soft_budget_seconds": budget,
                "fingerprint": identity,
                "results": [
                    {
                        key: result[key]
                        for key in [
                            "cut",
                            "trials",
                            "complete_trials",
                            "actual_fits",
                            "seconds",
                            "executed_epochs",
                        ]
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )
    return results
