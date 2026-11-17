"""Seed stability and ensembles whose parameters precede the outer period."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .review import (
    classification_metrics,
    development_data,
    feature_sets,
    fit_classic,
    optimal_threshold,
    predictions,
)


class AveragedPredictor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = np.asarray(
            weights if weights is not None else np.ones(len(models)) / len(models)
        )

    def predict_proba(self, frame):
        probability = np.average(
            [model.predict_proba(frame)[:, 1] for model in self.models],
            axis=0,
            weights=self.weights,
        )
        return np.column_stack([1 - probability, probability])


class StackedPredictor:
    def __init__(self, models, combiner):
        self.models = models
        self.combiner = combiner

    def predict_proba(self, frame):
        probabilities = np.column_stack(
            [model.predict_proba(frame)[:, 1] for model in self.models]
        )
        logits = np.log(
            np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1)
        )
        return self.combiner.predict_proba(logits)


class ColumnPredictor:
    def __init__(self, model, columns):
        self.model, self.columns = model, columns

    def predict_proba(self, frame):
        return self.model.predict_proba(frame[self.columns])


class StoreAdjustedPredictor:
    def __init__(self, model, columns, beta, anchor):
        self.model, self.columns = model, columns
        self.beta, self.anchor = beta, anchor

    def predict_proba(self, frame):
        original = self.model.predict_proba(frame[self.columns])[:, 1]
        probability = adjust_store_rate(
            original, frame.tienda_tasa_30, self.beta, self.anchor
        )
        return np.column_stack([1 - probability, probability])


def adjust_store_rate(probability, recent, beta, anchor):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    recent = np.clip(np.asarray(recent.fillna(0.25)), 1e-6, 1 - 1e-6)
    anchor = np.clip(anchor, 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability))
    logits += beta * (np.log(recent / (1 - recent)) - np.log(anchor / (1 - anchor)))
    return 1 / (1 + np.exp(-logits))


def aligned(frames):
    keys = ["id", "dia_visita", "y"]
    if any(frame.duplicated(keys[:2]).any() for frame in frames):
        raise ValueError("Duplicate prediction keys")
    if any(
        not frame[keys]
        .reset_index(drop=True)
        .equals(frames[0][keys].reset_index(drop=True))
        for frame in frames[1:]
    ):
        raise ValueError("Ensemble predictions are not aligned")
    return np.column_stack([frame.p.to_numpy() for frame in frames])


def write_result(directory, name, cut, end, inner, train, valid, threshold, details):
    directory.mkdir(parents=True, exist_ok=True)
    inner.to_csv(directory / "inner_predictions.csv", index=False)
    rows = []
    for split, frame in [("train", train), ("validation", valid)]:
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
        rows.append(
            {
                "model": name,
                "cut": cut,
                "end": end,
                "split": split,
                "n": len(frame),
                "threshold": threshold,
                **classification_metrics(frame.y, frame.p, threshold),
            }
        )
    result = {
        "model": name,
        "cut": cut,
        "end": end,
        "threshold": threshold,
        "metrics": rows,
        **details,
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def extra_identity(config):
    manifest = Path(config["output"]) / "manifest.json"
    content = manifest.read_bytes() + Path(__file__).read_bytes()
    return hashlib.sha256(content).hexdigest()


def cached(directory, identity):
    if not (directory / "result.json").exists():
        return False
    result = json.loads((directory / "result.json").read_text())
    if result.get("fingerprint") != identity:
        raise ValueError("Ensemble results have a different source/config identity")
    return (directory / "model.joblib").exists()


def seed_stability(config):
    """Repeat two families selected solely by their internal objective per outer fold."""
    out = Path(config["output"])
    identity = extra_identity(config)
    events, features, groups = feature_sets(development_data(config))
    for cut, end in config["outer_folds"]:
        candidates = [
            json.loads((out / f"{family}_{cut}" / "result.json").read_text())
            for family in ["logistic", "lightgbm", "xgboost", "catboost"]
        ]
        selected = sorted(candidates, key=lambda r: r["inner_score"], reverse=True)[:2]
        for rank, result in enumerate(selected, 1):
            started = perf_counter()
            family, columns = result["model"], result["columns"]
            directory = out / f"seedmean_{family}_{cut}"
            if cached(directory, identity):
                continue
            all_inner, all_train, all_valid, models, individual = [], [], [], [], []
            for seed in config["seeds"]:
                if seed == config["seed"]:
                    original = out / f"{family}_{cut}"
                    inner = pd.read_csv(original / "inner_predictions.csv")
                    train = pd.read_csv(original / "train_predictions.csv")
                    valid = pd.read_csv(original / "validation_predictions.csv")
                    model = joblib.load(original / "model.joblib")["model"]
                else:
                    frames = []
                    for fold, (inner_cut, stop) in enumerate(
                        config["inner_folds"][cut], 1
                    ):
                        model, _ = fit_classic(
                            events,
                            features[columns],
                            family,
                            result["parameters"],
                            inner_cut,
                            config,
                            seed,
                        )
                        frames.append(
                            predictions(
                                events,
                                model,
                                features[columns],
                                inner_cut + 1,
                                stop,
                                fold,
                            )
                        )
                    inner = pd.concat(frames)
                    model, _ = fit_classic(
                        events,
                        features[columns],
                        family,
                        result["parameters"],
                        cut,
                        config,
                        seed,
                    )
                    train = predictions(events, model, features[columns], 1, cut, 0)
                    valid = predictions(
                        events, model, features[columns], cut + 1, end, 0
                    )
                threshold, _, _ = optimal_threshold(
                    [frame for _, frame in inner.groupby("inner_fold")]
                )
                individual.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        **classification_metrics(valid.y, valid.p, threshold),
                    }
                )
                all_inner.append(inner)
                all_train.append(train)
                all_valid.append(valid)
                models.append(ColumnPredictor(model, columns))
            inner, train, valid = [
                frames[0].copy() for frames in [all_inner, all_train, all_valid]
            ]
            for target, frames in [
                (inner, all_inner),
                (train, all_train),
                (valid, all_valid),
            ]:
                target["p"] = aligned(frames).mean(axis=1)
            threshold, score, _ = optimal_threshold(
                [f for _, f in inner.groupby("inner_fold")]
            )
            predictor = AveragedPredictor(models)
            write_result(
                directory,
                f"seedmean_rank{rank}",
                cut,
                end,
                inner,
                train,
                valid,
                threshold,
                {
                    "fingerprint": identity,
                    "underlying_family": family,
                    "inner_score": score,
                    "seeds": individual,
                    "seconds": perf_counter() - started,
                    "selection": "Top two classical families by inner score, never outer labels",
                },
            )
            joblib.dump(
                {
                    "model": predictor,
                    "columns": list(features),
                    "threshold": threshold,
                    "feature_builder_version": "review_v1",
                    "training_end": cut,
                },
                directory / "model.joblib",
            )
            print(f"SEEDS {family} cut={cut} completed", flush=True)


def first_fold_parameters(directory):
    result = json.loads((directory / "result.json").read_text())
    study = optuna.load_study(
        study_name=f"{result['model']}_{result['cut']}_{result['fingerprint'][:12]}",
        storage=f"sqlite:///{(directory / 'study.db').resolve().as_posix()}",
    )
    choices = []
    for trial in study.trials:
        # TPE proposals after startup have already used later inner-fold scores.
        if trial.number >= 10 or trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        frame = pd.read_csv(directory / f"trial_{trial.number}_inner.csv")
        first = frame.loc[frame.inner_fold.eq(1)]
        _, score, f1 = optimal_threshold([first])
        choices.append((score, f1, -log_loss(first.y, first.p), trial))
    best = max(choices, key=lambda row: row[:3])[-1]
    return best.user_attrs["parameters"], best.params["feature_group"]


def chronological_ensembles(config):
    """Freeze base tuning on inner fold 1, fit combiners on 2, calibrate on 3."""
    started = perf_counter()
    out = Path(config["output"])
    identity = extra_identity(config)
    events, features, groups = feature_sets(development_data(config))
    for cut, end in config["outer_folds"]:
        if all(
            cached(out / f"{name}_{cut}", identity)
            for name in [
                "chronological_equal",
                "chronological_weighted",
                "chronological_stack",
            ]
        ):
            continue
        fold_started = perf_counter()
        families = ["logistic", "lightgbm", "xgboost", "catboost"]
        inner_frames = {2: [], 3: []}
        full_models = []
        details = {}
        for family in families:
            parameters, group = first_fold_parameters(out / f"{family}_{cut}")
            columns = groups[group]
            details[family] = {"parameters": parameters, "group": group}
            for fold in [2, 3]:
                inner_cut, stop = config["inner_folds"][cut][fold - 1]
                model, _ = fit_classic(
                    events, features[columns], family, parameters, inner_cut, config
                )
                inner_frames[fold].append(
                    predictions(
                        events, model, features[columns], inner_cut + 1, stop, fold
                    )
                )
            model, _ = fit_classic(
                events, features[columns], family, parameters, cut, config
            )
            full_models.append(ColumnPredictor(model, columns))
        a, b = aligned(inner_frames[2]), aligned(inner_frames[3])
        y_a = inner_frames[2][0].y.to_numpy()
        opt = minimize(
            lambda w: log_loss(y_a, a @ w),
            np.full(4, 0.25),
            method="SLSQP",
            bounds=[(0, 1)] * 4,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        )
        if not opt.success:
            raise RuntimeError(f"Weight optimization failed: {opt.message}")
        logits_a = np.log(np.clip(a, 1e-6, 1 - 1e-6) / np.clip(1 - a, 1e-6, 1))
        stack = LogisticRegression(C=0.01, max_iter=1000).fit(logits_a, y_a)
        predictors = {
            "chronological_equal": AveragedPredictor(full_models),
            "chronological_weighted": AveragedPredictor(full_models, opt.x),
            "chronological_stack": StackedPredictor(full_models, stack),
        }
        logits_b = np.log(np.clip(b, 1e-6, 1 - 1e-6) / np.clip(1 - b, 1e-6, 1))
        calibration = {
            "chronological_equal": b.mean(axis=1),
            "chronological_weighted": b @ opt.x,
            "chronological_stack": stack.predict_proba(logits_b)[:, 1],
        }
        for name, predictor in predictors.items():
            frame = inner_frames[3][0].copy()
            frame["p"] = calibration[name]
            threshold, score, _ = optimal_threshold([frame])
            train = predictions(events, predictor, features, 1, cut, 0)
            valid = predictions(events, predictor, features, cut + 1, end, 0)
            directory = out / f"{name}_{cut}"
            write_result(
                directory,
                name,
                cut,
                end,
                frame,
                train,
                valid,
                threshold,
                {
                    "fingerprint": identity,
                    "inner_score": score,
                    "base_parameters": details,
                    "weights": opt.x.tolist() if name.endswith("weighted") else None,
                    "seconds": perf_counter() - fold_started,
                    "shared_training_group": f"chronological_{cut}",
                    "selection": "Base tuning inner1; combiner inner2; threshold inner3; fixed C=.01",
                },
            )
            joblib.dump(
                {
                    "model": predictor,
                    "columns": list(features),
                    "threshold": threshold,
                    "feature_builder_version": "review_v1",
                    "training_end": cut,
                },
                directory / "model.joblib",
            )
        print(f"CHRONOLOGICAL ENSEMBLES cut={cut} completed", flush=True)
    (out / "ensembles_runtime.json").write_text(
        json.dumps({"wall_seconds": perf_counter() - started})
    )


def causal_rate_adjustment(config):
    """Test a store-rate odds correction, selecting beta only on internal folds."""
    out = Path(config["output"])
    identity = extra_identity(config)
    events, features, _ = feature_sets(development_data(config))
    lookup = events[["id", "dia_visita"]].copy()
    lookup["recent_rate"] = features.tienda_tasa_30
    for cut, end in config["outer_folds"]:
        candidates = [
            json.loads((out / f"{family}_{cut}" / "result.json").read_text())
            for family in ["logistic", "lightgbm", "xgboost", "catboost"]
        ]
        for rank, result in enumerate(
            sorted(candidates, key=lambda r: r["inner_score"], reverse=True)[:2], 1
        ):
            started = perf_counter()
            family = result["model"]
            directory = out / f"store_adjusted_{family}_{cut}"
            if cached(directory, identity):
                continue
            source = out / f"{family}_{cut}"
            inner = pd.read_csv(source / "inner_predictions.csv").merge(
                lookup, on=["id", "dia_visita"], validate="one_to_one"
            )
            half_life = result["parameters"]["half_life"]

            def anchor(end_day):
                mask = events.dia_visita.le(end_day)
                w = np.exp2((events.loc[mask, "dia_visita"] - end_day) / half_life)
                return float(
                    np.average(events.loc[mask, "incidencia_compra"], weights=w)
                )

            scores = []
            for beta in [0, 0.25, 0.5, 0.75, 1, 1.5, 2]:
                frames = []
                for fold, (inner_cut, _) in enumerate(config["inner_folds"][cut], 1):
                    frame = inner.loc[inner.inner_fold.eq(fold)].copy()
                    frame["p"] = adjust_store_rate(
                        frame.p, frame.recent_rate, beta, anchor(inner_cut)
                    )
                    frames.append(frame)
                threshold, score, f1 = optimal_threshold(frames)
                scores.append((score, f1, -abs(beta), beta, threshold, frames))
            _, _, _, beta, threshold, frames = max(scores, key=lambda r: r[:3])
            model = joblib.load(source / "model.joblib")["model"]
            predictor = StoreAdjustedPredictor(
                model, result["columns"], beta, anchor(cut)
            )
            train = predictions(events, predictor, features, 1, cut, 0)
            valid = predictions(events, predictor, features, cut + 1, end, 0)
            write_result(
                directory,
                f"store_adjusted_rank{rank}",
                cut,
                end,
                pd.concat(frames),
                train,
                valid,
                threshold,
                {
                    "fingerprint": identity,
                    "underlying_family": family,
                    "beta": beta,
                    "anchor": anchor(cut),
                    "inner_score": max(scores, key=lambda r: r[:3])[0],
                    "seconds": perf_counter() - started,
                    "selection": "Beta and threshold from inner folds; recent rate excludes current day",
                },
            )
            joblib.dump(
                {
                    "model": predictor,
                    "columns": list(features),
                    "threshold": threshold,
                    "feature_builder_version": "review_v1",
                    "training_end": cut,
                },
                directory / "model.joblib",
            )
