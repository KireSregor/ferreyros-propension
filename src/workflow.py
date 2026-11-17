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
    historical_frames,
    inner_predictions,
    metrics,
    score_visits,
    train_window,
    tune_fold,
)


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
    baseline_threshold = f1_threshold(
        historical_frames(events, candidate_features, cut)
    )
    for split, a, b in [("Train", 1, cut), ("Validación", cut + 1, end)]:
        row, frame = evaluate_bundle(events, features, bundle, a, b)
        results.append({"model": "LightGBM", "cut": cut, "split": split, **row})
        stored_predictions[f"lightgbm_{cut}_{split}"] = frame.p.to_numpy()
        mask = events.dia_visita.between(a, b)
        results.append(
            {
                "model": "Histórico",
                "cut": cut,
                "split": split,
                "n": int(mask.sum()),
                **metrics(
                    events.loc[mask, "incidencia_compra"],
                    candidate_features.loc[mask, "tasa_previa"],
                    baseline_threshold,
                ),
            }
        )
    return {
        "study": study,
        "bundle": bundle,
        "metrics": results,
        "predictions": stored_predictions,
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


def evaluate_test(bundle, data_path):
    """Evalúa 621–730 con modelo y umbral ya congelados; no entrena ni selecciona."""
    complete = load_data(data_path, end_day=730)
    events, features = reviewed_features(complete)
    row, _ = evaluate_bundle(events, features, bundle, 621, 730)
    return {"model": "LightGBM", "cut": bundle["training_end"], "split": "Test", **row}


def save_run(output, runs, results, config, data_path, started):
    """Guarda únicamente modelo, métricas y trazabilidad de esta ejecución."""
    output.mkdir(parents=True, exist_ok=True)
    bundle = runs[512]["bundle"]
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
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
    }
    (output / "execution.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
    return trace
