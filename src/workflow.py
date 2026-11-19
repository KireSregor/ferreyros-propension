"""Pasos del notebook: entrenamiento, scoring y exportación reproducibles."""

import hashlib
import importlib.metadata as metadata
import json
import platform
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

from .data import load_data
from .feature_research import reviewed_features
from .training import (
    evaluate_bundle,
    f1_threshold,
    inner_predictions,
    ranking_metrics,
    score_visits,
    train_window,
    tune_fold,
)

BASELINES = {"Histórico": "tasa_previa", "Tasa mensual": "month_recent_smoothed"}


def rank_deciles(frame, cut, split, seed):
    """Rank each period separately and break score ties without using outcomes."""
    target = frame.y.to_numpy()
    tie_order = np.random.default_rng(seed).permutation(len(target))
    boundaries = np.arange(11) * len(target) // 10
    ranked = target[np.lexsort((tie_order, -frame.p.to_numpy()))]
    return [
        {
            "model": "LightGBM",
            "cut": cut,
            "split": split,
            "decile": decile,
            "visits": int(end - start),
            "purchases": int(ranked[start:end].sum()),
        }
        for decile, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True), 1
        )
    ]


def evaluate_period(events, features, bundle, start, end, split, seed):
    """Use the same visits and realized outcomes for all three scoring rules."""
    cut = bundle["training_end"]
    row, frame = evaluate_bundle(events, features, bundle, start, end)
    results = [{"model": "LightGBM", "cut": cut, "split": split, **row}]
    for model, column in BASELINES.items():
        results.append(
            {
                "model": model,
                "cut": cut,
                "split": split,
                "n": len(frame),
                **ranking_metrics(frame.y, features.loc[frame.index, column]),
            }
        )
    deciles = [] if split == "Train" else rank_deciles(frame, cut, split, seed)
    return results, frame.p.to_numpy(), deciles


def run_fold(events, features, candidate_features, cut, end, config):
    """Ajusta solo en folds internos; evalúa train y validación exterior sin reajuste."""
    results, stored_predictions = [], {}
    study = tune_fold(events, features, cut, config)
    parameters = study.best_params
    internal = inner_predictions(events, features, parameters, cut, config)
    threshold = f1_threshold(internal)
    bundle = train_window(events, features, parameters, cut, config)
    bundle.update(
        threshold=threshold,
        parameters=parameters,
        feature_builder_version="lightgbm_v1",
    )
    for split, start, finish in [("Train", 1, cut), ("Validación", cut + 1, end)]:
        rows, probabilities, deciles = evaluate_period(
            events,
            candidate_features,
            bundle,
            start,
            finish,
            split,
            config.seed,
        )
        results.extend(rows)
        stored_predictions[f"lightgbm_{cut}_{split}"] = probabilities
    return {
        "study": study,
        "bundle": bundle,
        "metrics": results,
        "predictions": stored_predictions,
        "deciles": deciles,
    }


def verify_scoring(bundle, raw, validation_predictions, events, day=620):
    """Comprueba que scoring sin desenlace actual reproduce la evaluación por lotes."""
    scored = score_visits(
        bundle, raw.loc[raw.dia_visita.lt(day)], raw.loc[raw.dia_visita.eq(day)]
    )
    days = events.loc[events.dia_visita.between(513, 620), "dia_visita"].to_numpy()
    np.testing.assert_allclose(
        scored.probability, validation_predictions[days == day], rtol=0, atol=1e-10
    )
    return scored


def evaluate_test(run, data_path, config):
    """Evalúa 621–730 con modelo y umbral ya congelados; no entrena ni selecciona."""
    complete = load_data(data_path, end_day=730)
    events, features = reviewed_features(complete)
    results, _, deciles = evaluate_period(
        events,
        features,
        run["bundle"],
        621,
        730,
        "Test",
        config.seed,
    )
    return results, deciles


def save_run(output, runs, results, deciles, config, data_path, started):
    """Guarda únicamente modelo, métricas y trazabilidad de esta ejecución."""
    output.mkdir(parents=True, exist_ok=True)
    bundle = runs[512]["bundle"]
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(deciles).to_csv(output / "deciles.csv", index=False)
    joblib.dump(bundle, output / "model.joblib")
    trace = {
        "model_schema": {
            key: bundle[key]
            for key in [
                "columns",
                "training_end",
                "threshold",
                "parameters",
                "iterations",
                "feature_builder_version",
            ]
        },
        "environment": {
            name: metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "optuna"]
        },
        "python": platform.python_version(),
        "config": asdict(config),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "studies": {
            str(cut): {
                "parameters": run["study"].best_params,
                "values": [t.value for t in run["study"].trials],
                "trial_parameters": [t.params for t in run["study"].trials],
                "reason": run["study"].user_attrs["stop_reason"],
                "seconds": run["study"].user_attrs["seconds"],
            }
            for cut, run in runs.items()
        },
        "thresholds": {
            str(cut): run["bundle"]["threshold"] for cut, run in runs.items()
        },
        "seconds": time.perf_counter() - started,
        "evaluation_context": {
            "test_days": [621, 730],
            "test_previously_reviewed": True,
            "test_used_for_refitting_this_run": False,
        },
    }
    (output / "execution.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
    return trace
