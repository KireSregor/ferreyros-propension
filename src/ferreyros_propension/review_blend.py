"""Blend complementary neural/tree predictions without exterior-label selection."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .review import development_data, feature_sets, optimal_threshold, predictions
from .review_ensembles import AveragedPredictor, ColumnPredictor, aligned, write_result


def blend_neural_trees(config):
    started = perf_counter()
    output = Path(config["output"])
    events, features, _ = feature_sets(development_data(config))
    for cut, end in config["outer_folds"]:
        fold_started = perf_counter()
        classical = []
        for family in ["lightgbm", "xgboost", "catboost"]:
            directory = output / f"{family}_{cut}"
            result = json.loads((directory / "result.json").read_text())
            classical.append((result["inner_score"], directory, result))
        neural = []
        for family in ["neural", "neural_wide"]:
            directory = output / f"{family}_{cut}"
            result = json.loads((directory / "result.json").read_text())
            neural.append((result["inner_score"], directory, result))
        choices = [
            max(classical, key=lambda row: row[0]),
            max(neural, key=lambda row: row[0]),
        ]
        directories = [row[1] for row in choices]
        identity = hashlib.sha256(
            Path(__file__).read_bytes()
            + b"".join(
                (directory / "result.json").read_bytes() for directory in directories
            )
        ).hexdigest()
        destination = output / f"neural_tree_blend_{cut}"
        marker = destination / "result.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["fingerprint"] != identity:
                raise ValueError(
                    "Neural/tree blend inputs changed; use a new result version"
                )
            if (destination / "model.joblib").exists():
                continue
        frames = [
            pd.read_csv(directory / "inner_predictions.csv")
            for directory in directories
        ]
        matrix = aligned(frames)
        correlation = float(np.corrcoef(matrix.T)[0, 1])
        candidates = []
        for neural_weight in [0.25, 0.5, 0.75]:
            frame = frames[0].copy()
            frame["p"] = matrix @ np.array([1 - neural_weight, neural_weight])
            folds = [part for _, part in frame.groupby("inner_fold")]
            threshold, score, f1 = optimal_threshold(folds)
            candidates.append(
                (score, f1, -abs(neural_weight - 0.5), neural_weight, threshold, frame)
            )
        score, _, _, neural_weight, threshold, inner = max(
            candidates, key=lambda row: row[:3]
        )
        models = []
        for directory in directories:
            bundle = joblib.load(directory / "model.joblib")
            if bundle["feature_builder_version"] != "review_v1":
                raise ValueError("Incompatible blend feature schema")
            models.append(ColumnPredictor(bundle["model"], bundle["columns"]))
        model = AveragedPredictor(models, [1 - neural_weight, neural_weight])
        train = predictions(events, model, features, 1, cut, 0)
        valid = predictions(events, model, features, cut + 1, end, 0)
        write_result(
            destination,
            "neural_tree_blend",
            cut,
            end,
            inner,
            train,
            valid,
            threshold,
            {
                "fingerprint": identity,
                "inner_score": score,
                "neural_weight": neural_weight,
                "base_models": [row[2]["model"] for row in choices],
                "inner_prediction_correlation": correlation,
                "seconds": perf_counter() - fold_started,
                "selection": "Bases, mixing weight and threshold selected internally; internal scores optimistic",
            },
        )
        joblib.dump(
            {
                "model": model,
                "feature_builder_version": "review_v1",
                "columns": list(features),
                "threshold": threshold,
                "training_end": cut,
            },
            destination / "model.joblib",
        )
        print(f"NEURAL/TREE BLEND cut={cut} correlation={correlation:.3f}", flush=True)
    (output / "blend_runtime.json").write_text(
        json.dumps({"wall_seconds": perf_counter() - started})
    )
