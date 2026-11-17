"""Controlled calibration experiment with a frozen, earlier-trained base model."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .review import (
    development_data,
    feature_sets,
    fingerprint,
    fit_classic,
    optimal_threshold,
    predictions,
)
from .review_ensembles import write_result


def _logit(probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


class CalibratedReference:
    """Apply calibration without changing the base model after calibration."""

    def __init__(self, base, columns, calibrator, with_store):
        self.base, self.columns = base, columns
        self.calibrator, self.with_store = calibrator, with_store

    def calibration_inputs(self, frame):
        probability = self.base.predict_proba(frame[self.columns])[:, 1]
        inputs = [_logit(probability)]
        if self.with_store:
            inputs.append(_logit(frame.tienda_tasa_30.fillna(0.25)))
        return np.column_stack(inputs)

    def predict_proba(self, frame):
        return self.calibrator.predict_proba(self.calibration_inputs(frame))


def _calibrated_model(base, columns, features, events, cut, with_store):
    mask = events.dia_visita.between(cut - 39, cut)
    target = events.loc[mask, "incidencia_compra"]
    if target.nunique() != 2:
        raise ValueError("Temporal calibration requires both purchase classes")
    model = CalibratedReference(base, columns, None, with_store)
    inputs = model.calibration_inputs(features.loc[mask])
    model.calibrator = LogisticRegression(C=1, max_iter=1000, random_state=42).fit(
        inputs, target
    )
    return model


def run_calibration(config):
    """Select Platt versus store-augmented calibration internally, within 300 s.

    Each base is trained through cut-40; only its calibrator sees the final
    40 training days. The calibrated base is never refitted after that step.
    This is exploratory diagnosis of threshold transfer, not a new clean test.
    """
    started = perf_counter()
    output = Path(config["output"])
    data = development_data(config)
    events, features, groups = feature_sets(data)
    columns = groups["reference"]
    identity = hashlib.sha256(
        fingerprint(data, config).encode() + Path(__file__).read_bytes()
    ).hexdigest()
    completed = []
    for cut, end in config["outer_folds"]:
        fold_started = perf_counter()
        directory = output / f"calibrated_reference_{cut}"
        marker = directory / "result.json"
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != identity:
                raise ValueError("Calibration source/config/data identity changed")
            if (directory / "model.joblib").exists():
                completed.append(previous)
                continue
        variants = {False: [], True: []}
        base_fits, calibrator_fits = 0, 0
        for fold, (inner_cut, inner_end) in enumerate(config["inner_folds"][cut], 1):
            if perf_counter() - started >= 300:
                raise TimeoutError(
                    "Calibration experiment reached its 300-second budget"
                )
            base, _ = fit_classic(
                events,
                features[columns],
                "reference",
                {"half_life": 180},
                inner_cut - 40,
                config,
            )
            base_fits += 1
            for with_store in variants:
                model = _calibrated_model(
                    base, columns, features, events, inner_cut, with_store
                )
                calibrator_fits += 1
                variants[with_store].append(
                    predictions(events, model, features, inner_cut + 1, inner_end, fold)
                )
        choices = []
        for with_store, frames in variants.items():
            threshold, balance, f1 = optimal_threshold(frames)
            loss = float(np.mean([log_loss(frame.y, frame.p) for frame in frames]))
            choices.append((balance, f1, -loss, with_store, threshold))
        balance, _, _, with_store, threshold = max(choices, key=lambda row: row[:3])
        if perf_counter() - started >= 300:
            raise TimeoutError("Calibration experiment reached its 300-second budget")
        base, _ = fit_classic(
            events, features[columns], "reference", {"half_life": 180}, cut - 40, config
        )
        base_fits += 1
        model = _calibrated_model(base, columns, features, events, cut, with_store)
        calibrator_fits += 1
        train = predictions(events, model, features, 1, cut, 0)
        valid = predictions(events, model, features, cut + 1, end, 0)
        result = write_result(
            directory,
            "calibrated_reference",
            cut,
            end,
            pd.concat(variants[with_store]),
            train,
            valid,
            threshold,
            {
                "fingerprint": identity,
                "inner_score": balance,
                "variant": "platt_store" if with_store else "platt",
                "base_training_end": cut - 40,
                "calibration_days": [cut - 39, cut],
                "base_fits": base_fits,
                "calibrator_fits": calibrator_fits,
                "total_fits": base_fits + calibrator_fits,
                "trials": 2,
                "iterations": 400,
                "seconds": perf_counter() - fold_started,
                "selection": "Two predefined calibration variants and shared threshold selected on internal folds; exploratory threshold-transfer diagnosis; base never refit after calibration",
                "variant_scores": [
                    {
                        "variant": "platt_store" if choice[3] else "platt",
                        "balance": choice[0],
                        "f1": choice[1],
                        "logloss": -choice[2],
                    }
                    for choice in choices
                ],
            },
        )
        joblib.dump(
            {
                "model": model,
                "columns": list(features),
                "threshold": threshold,
                "feature_builder_version": "review_v1",
                "training_end": cut,
                "base_training_end": cut - 40,
            },
            directory / "model.joblib",
        )
        completed.append(result)
        print(f"CALIBRATED REFERENCE cut={cut} variant={result['variant']}", flush=True)
    (output / "calibration_runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "budget_seconds": 300,
                "fits_in_completed_folds": sum(r["total_fits"] for r in completed),
            }
        ),
        encoding="utf-8",
    )
    return completed
