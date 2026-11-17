"""Targeted neural capacity check with selection confined to inner folds."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .review import (
    classification_metrics,
    development_data,
    feature_sets,
    fingerprint,
    optimal_threshold,
)
from .review_neural import NeuralPredictor, _fit_window, _frame

VARIANTS = {
    "tabm_256_k32": {
        "architecture": "tabm",
        "k": 32,
        "width": 256,
        "depth": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 0.0003,
        "quantile_transform": True,
        "half_life": 180,
        "batch_size": 512,
    },
    "tabm_512_k32": {
        "architecture": "tabm",
        "k": 32,
        "width": 512,
        "depth": 3,
        "dropout": 0.35,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
        "quantile_transform": True,
        "half_life": 180,
        "batch_size": 512,
    },
    "mlp_512": {
        "architecture": "mlp",
        "width": 512,
        "depth": 3,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
        "quantile_transform": True,
        "half_life": 180,
        "batch_size": 512,
    },
}


def _identity(data, config):
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ["review_neural_wide.py", "review_neural.py", "neural_training.py"]
    }
    value = {"review": fingerprint(data, config), "sources": sources}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _run_fold(events, features, cut, end, config, identity, timeout):
    started = perf_counter()
    directory = Path(config["output"]) / f"neural_wide_{cut}"
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "fingerprint.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text())["fingerprint"] != identity
    ):
        raise ValueError("Wide neural source/data/config fingerprint changed")
    identity_path.write_text(json.dumps({"fingerprint": identity}, indent=2))
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    candidates = []
    budget_stop = False
    for name, parameters in VARIANTS.items():
        if candidates and perf_counter() - started >= timeout:
            budget_stop = True
            break
        variant_started = perf_counter()
        metadata_path = directory / f"{name}_inner.json"
        if metadata_path.exists():
            candidates.append(json.loads(metadata_path.read_text()))
            continue
        frames, records = [], []
        for fold, (inner_cut, inner_end) in enumerate(config["inner_folds"][cut], 1):
            result = _fit_window(
                events,
                features,
                parameters,
                inner_cut,
                inner_end,
                config,
                directory,
                f"{name}_fold_{fold}",
                records,
            )
            mask = events.dia_visita.between(inner_cut + 1, inner_end)
            frames.append(
                _frame(events, mask, result["probabilities"]["validation"], fold)
            )
        threshold, score, f1 = optimal_threshold(frames)
        pd.concat(frames).to_csv(directory / f"{name}_inner.csv", index=False)
        candidate = {
            "variant": name,
            "parameters": parameters,
            "threshold": threshold,
            "inner_score": score,
            "f1": f1,
            "logloss": float(np.mean([log_loss(frame.y, frame.p) for frame in frames])),
            "seconds": perf_counter() - variant_started,
            "fits": records,
        }
        metadata_path.write_text(json.dumps(candidate, indent=2))
        candidates.append(candidate)
        print(
            f"WIDE cut={cut} variant={name}: score={score:.4f}, {candidate['seconds']:.1f}s",
            flush=True,
        )
    best = max(
        candidates, key=lambda item: (item["inner_score"], item["f1"], -item["logloss"])
    )
    pd.read_csv(directory / f"{best['variant']}_inner.csv").to_csv(
        directory / "inner_predictions.csv", index=False
    )
    outer_records = []
    fitted = _fit_window(
        events,
        features,
        best["parameters"],
        cut,
        end,
        config,
        directory,
        "outer",
        outer_records,
        artifact=True,
    )
    threshold = best["threshold"]
    rows = []
    for split, mask in [
        ("train", events.dia_visita.le(cut)),
        ("validation", events.dia_visita.between(cut + 1, end)),
    ]:
        frame = _frame(events, mask, fitted["probabilities"][split], 0)
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
        rows.append(
            {
                "model": "neural_wide",
                "cut": cut,
                "end": end,
                "split": split,
                "threshold": threshold,
                "n": len(frame),
                **classification_metrics(frame.y, frame.p, threshold),
            }
        )
    artifact = {
        "model": NeuralPredictor(fitted["artifact"]),
        "feature_builder_version": "review_v1",
        "columns": list(features),
        "threshold": threshold,
        "training_end": cut,
    }
    joblib.dump(artifact, directory / "model.joblib")
    fit_records = [
        record for candidate in candidates for record in candidate["fits"]
    ] + outer_records
    pd.DataFrame(fit_records).to_csv(directory / "fit_history.csv", index=False)
    result = {
        "model": "neural_wide",
        "cut": cut,
        "end": end,
        "fingerprint": identity,
        "parameters": best["parameters"],
        "selected_variant": best["variant"],
        "feature_group": "enhanced",
        "columns": list(features),
        "threshold": threshold,
        "inner_score": best["inner_score"],
        "iterations": fitted["best_epochs"],
        "trials": len(candidates),
        "complete_trials": len(candidates),
        "actual_fits": len(fit_records),
        "executed_epochs": sum(record["executed_epochs"] for record in fit_records),
        "outer_selected_epochs": fitted["best_epochs"],
        "outer_refit_executed_epochs": fitted["executed_epochs"],
        "peak_gpu_memory_bytes": max(
            record["peak_gpu_memory_bytes"] for record in fit_records
        ),
        "variants_planned": list(VARIANTS),
        "variants_completed": [item["variant"] for item in candidates],
        "stopped_for_budget": budget_stop,
        "soft_timeout_seconds": timeout,
        "seconds": perf_counter() - started,
        "metrics": rows,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_wide(config):
    """Check three predeclared capacities, sequentially on the available GPU.

    This is a targeted hypothesis test, not an architecture-search optimum.
    The 600-second budget is soft: an active variant and its selected outer
    refit complete, and each outer fold receives at least one complete variant.
    """
    started = perf_counter()
    data = development_data(config)
    events, features, groups = feature_sets(data)
    features = features[groups["enhanced"]]
    identity = _identity(data, config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    budget = 600
    results = []
    for index, (cut, end) in enumerate(config["outer_folds"]):
        remaining = max(1, budget - (perf_counter() - started))
        timeout = remaining / (len(config["outer_folds"]) - index)
        result = _run_fold(events, features, cut, end, config, identity, timeout)
        results.append(result)
        pd.DataFrame([row for item in results for row in item["metrics"]]).to_csv(
            output / "neural_wide_metrics.csv", index=False
        )
        print(
            f"DONE neural_wide {cut}: {result['selected_variant']}, "
            f"P={result['metrics'][-1]['precision']:.4f}, R={result['metrics'][-1]['recall']:.4f}",
            flush=True,
        )
    (output / "neural_wide_runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "soft_budget_seconds": budget,
                "fingerprint": identity,
                "folds": [
                    {
                        key: item[key]
                        for key in [
                            "cut",
                            "variants_completed",
                            "stopped_for_budget",
                            "actual_fits",
                            "seconds",
                        ]
                    }
                    for item in results
                ],
            },
            indent=2,
        )
    )
    return results
