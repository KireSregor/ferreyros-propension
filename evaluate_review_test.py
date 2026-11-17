"""Evaluate the six explicitly authorized, frozen last-fold artifacts on test."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd

from ferreyros_propension.data import load_data
from ferreyros_propension.review import classification_metrics, feature_sets


MODELS = [
    "xgboost", "chronological_weighted", "reference",
    "catboost", "lightgbm", "chronological_stack",
]


def main():
    source = Path("outputs/review_v1")
    output = Path("outputs/review_test_512")
    output.mkdir(exist_ok=True)
    frozen = []
    bundles = {}
    for name in MODELS:
        artifact = source / f"{name}_512/model.joblib"
        bundle = joblib.load(artifact)
        if bundle["training_end"] != 512 or bundle["feature_builder_version"] != "review_v1":
            raise ValueError(f"Unexpected artifact: {name}")
        bundles[name] = bundle
        frozen.append({
            "model": name, "artifact": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "threshold": bundle["threshold"], "training_end": 512,
        })
    protocol = {
        "authorization": "User requested test evaluation of the first six procedures",
        "train": [1, 512], "validation": [513, 620], "test": [621, 730],
        "refit": False, "threshold_adjustment": False,
        "history": "Prior observed visits update online; store signals exclude current day",
        "limitation": "Test was examined historically; not an untouched independent holdout",
        "artifacts": frozen,
    }
    freeze_path = output / "frozen_protocol.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != protocol:
        raise ValueError("Frozen test protocol differs; do not overwrite")
    freeze_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    started = perf_counter()
    events, features, _ = feature_sets(load_data("Dataset/compras_data.csv", end_day=730))
    build_seconds = perf_counter() - started
    mask = events.dia_visita.between(621, 730)
    rows = []
    for name, bundle in bundles.items():
        tick = perf_counter()
        probability = bundle["model"].predict_proba(features.loc[mask, bundle["columns"]])[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError(f"Invalid probabilities: {name}")
        frame = events.loc[mask, ["id", "dia_visita", "incidencia_compra"]].copy()
        frame["probability"] = probability
        frame["prediction"] = (probability >= bundle["threshold"]).astype(int)
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
        row = {
            "model": name, "train_end": 512, "test_start": 621, "test_end": 730,
            "n": int(mask.sum()), "threshold": bundle["threshold"],
            **classification_metrics(frame.incidencia_compra, probability, bundle["threshold"]),
            "scoring_and_export_seconds": perf_counter() - tick,
        }
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    table.to_csv(output / "metrics.csv", index=False)
    (output / "timing.json").write_text(json.dumps({
        "feature_build_seconds": build_seconds, "total_seconds": perf_counter() - started,
    }, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
