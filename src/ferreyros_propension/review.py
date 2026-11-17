"""Versioned temporal review: train-local preprocessing and nested selection."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, PRICES, PROFILE, PROMOS, load_data
from .feature_research import monthly_trend_features, reviewed_features


def read_config(path="review_config.yml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def development_data(config):
    return load_data("Dataset/compras_data.csv", end_day=config["development_end"])


def feature_sets(data):
    events, all_features = reviewed_features(data)
    _, original = monthly_trend_features(data)
    original = original.drop(
        columns=[
            c
            for c in original
            if c.startswith("fase_") or c in {"cliente_conocido", "dia_relativo"}
        ]
    )
    offer_columns = [c for c in original if "precio" in c or "promo" in c]
    history = [
        c for c in original if c not in PROFILE + PRICES + PROMOS + offer_columns
    ]
    reference = [c for c in original if c not in PROFILE + PRICES + PROMOS]
    groups = {
        "history": history,
        "history_offers": [c for c in original if c not in PROFILE],
        "all": list(original),
        "enhanced": list(all_features),
        "reference": reference,
    }
    return events, all_features, groups


def classification_metrics(y, probability, threshold):
    y, probability = np.asarray(y), np.asarray(probability)
    labels = probability >= threshold
    return {
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "logloss": float(log_loss(y, probability)),
        "purchase_rate": float(y.mean()),
    }


def optimal_threshold(frames):
    """One threshold maximizing equal-fold mean min(P,R), then mean F1."""
    candidates = np.unique(np.concatenate([frame.p.to_numpy() for frame in frames]))
    score = np.zeros(len(candidates))
    f1 = np.zeros(len(candidates))
    for frame in frames:
        order = np.argsort(frame.p.to_numpy(), kind="stable")
        values = frame.p.to_numpy()[order]
        target = frame.y.to_numpy()[order]
        cumulative = np.r_[0, target.cumsum()]
        left = np.searchsorted(values, candidates, side="left")
        tp = cumulative[-1] - cumulative[left]
        predicted = len(values) - left
        precision = np.divide(
            tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0
        )
        recall = tp / cumulative[-1]
        score += np.minimum(precision, recall) / len(frames)
        f1 += np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=precision + recall > 0,
        ) / len(frames)
    best = np.lexsort((candidates, f1, score))[-1]
    return float(candidates[best]), float(score[best]), float(f1[best])


def preprocessing(frame, family):
    categorical = [c for c in frame if c in CATEGORICAL]
    numeric = [c for c in frame if c not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                    StandardScaler() if family == "logistic" else "passthrough",
                ),
                numeric,
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def estimator(family, params, iterations, config, seed, stopping=False):
    threads = config["threads_per_model"]
    rounds = config["early_stopping"]["rounds"]
    if family == "logistic":
        return LogisticRegression(C=params["C"], max_iter=1500, random_state=seed)
    if family == "lightgbm":
        return LGBMClassifier(
            n_estimators=iterations,
            learning_rate=params["learning_rate"],
            max_depth=params["depth"],
            num_leaves=min(2 ** params["depth"], 32),
            min_child_samples=params["min_samples"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            colsample_bytree=params["colsample"],
            n_jobs=threads,
            random_state=seed,
            verbosity=-1,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=iterations,
            learning_rate=params["learning_rate"],
            max_depth=params["depth"],
            min_child_weight=params["min_samples"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            colsample_bytree=params["colsample"],
            subsample=params["subsample"],
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=threads,
            random_state=seed,
            early_stopping_rounds=rounds if stopping else None,
        )
    if family == "reference":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=3,
            max_depth=4,
            min_child_samples=300,
            reg_lambda=100,
            reg_alpha=5,
            verbosity=-1,
            n_jobs=threads,
            random_state=seed,
        )
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["reg_lambda"],
        random_strength=params["random_strength"],
        thread_count=threads,
        cat_features=params["cat_features"],
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def weights(events, mask, cut, half_life):
    result = np.exp2((events.loc[mask, "dia_visita"].to_numpy() - cut) / half_life)
    return result / result.mean()


def fit_classic(events, features, family, params, cut, config, seed=42):
    """Choose iterations internally, refit full train, and return a fitted pipeline."""
    train = events.dia_visita.le(cut)
    target = events.incidencia_compra
    half_life = params.get("half_life", 180)
    selected = None

    def transformed(mask, other=None):
        frame = features.loc[mask].copy()
        if family == "catboost":
            return (
                frame,
                features.loc[other].copy() if other is not None else None,
                None,
            )
        transformer = preprocessing(frame, family)
        a = transformer.fit_transform(frame)
        b = transformer.transform(features.loc[other]) if other is not None else None
        return a, b, transformer

    if family not in {"logistic", "reference"}:
        stop_days = config["early_stopping"]["days"]
        inner = events.dia_visita.le(cut - stop_days)
        stop = events.dia_visita.between(cut - stop_days + 1, cut)
        a, b, _ = transformed(inner, stop)
        parameters = {
            **params,
            "cat_features": [c for c in features if c in CATEGORICAL],
        }
        model = estimator(
            family,
            parameters,
            config["early_stopping"]["max_trees"],
            config,
            seed,
            True,
        )
        arguments = {
            "sample_weight": weights(events, inner, cut - stop_days, half_life)
        }
        if family == "lightgbm":
            arguments.update(
                eval_X=b,
                eval_y=target[stop],
                callbacks=[
                    early_stopping(config["early_stopping"]["rounds"], verbose=False),
                    log_evaluation(0),
                ],
            )
        else:
            arguments["eval_set"] = [(b, target[stop])]
            if family == "catboost":
                arguments["early_stopping_rounds"] = config["early_stopping"]["rounds"]
            else:
                arguments["verbose"] = False
        model.fit(a, target[inner], **arguments)
        if family == "lightgbm":
            selected = max(1, model.best_iteration_)
        elif family == "catboost":
            selected = max(1, model.best_iteration_ + 1)
        else:
            selected = model.best_iteration + 1
    a, _, transformer = transformed(train)
    parameters = {**params, "cat_features": [c for c in features if c in CATEGORICAL]}
    model = estimator(family, parameters, selected or 400, config, seed)
    model.fit(a, target[train], sample_weight=weights(events, train, cut, half_life))
    pipeline = model if transformer is None else make_pipeline(transformer, model)
    return pipeline, selected


def suggest_parameters(trial, family, config):
    group = trial.suggest_categorical(
        "feature_group", config["search"]["feature_groups"]
    )
    params = {"half_life": trial.suggest_float("half_life", 90, 720, log=True)}
    if family == "logistic":
        params["C"] = trial.suggest_float("C", 0.0001, 10, log=True)
        return group, params
    params.update(
        learning_rate=trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
        depth=trial.suggest_int("depth", 2, 6),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 500, log=True),
    )
    if family == "catboost":
        params["random_strength"] = trial.suggest_float(
            "random_strength", 0.01, 10, log=True
        )
    else:
        params.update(
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 40, log=True),
            colsample=trial.suggest_float("colsample", 0.6, 1.0),
        )
        params["min_samples"] = (
            trial.suggest_int("min_samples", 100, 900)
            if family == "lightgbm"
            else trial.suggest_float("min_samples", 2, 150, log=True)
        )
        if family == "xgboost":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return group, params


def predictions(events, model, features, start, end, fold):
    mask = events.dia_visita.between(start, end)
    result = events.loc[mask, ["id", "dia_visita"]].copy()
    result["y"] = events.loc[mask, "incidencia_compra"]
    result["p"] = model.predict_proba(features.loc[mask])[:, 1]
    result["inner_fold"] = fold
    return result


def inner_predictions(events, features, family, params, cut, config):
    frames, iterations = [], []
    for fold, (inner_cut, end) in enumerate(config["inner_folds"][cut], 1):
        model, selected = fit_classic(
            events, features, family, params, inner_cut, config
        )
        frames.append(predictions(events, model, features, inner_cut + 1, end, fold))
        iterations.append(selected)
    return frames, iterations


def fingerprint(data, config):
    digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ["review.py", "data.py", "feature_research.py"]
    }
    contract = json.dumps(
        {"data": digest, "config": config, "sources": sources}, sort_keys=True
    )
    return hashlib.sha256(contract.encode()).hexdigest()


def run_classic_task(family, cut, end, config, timeout):
    started = perf_counter()
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    data = development_data(config)
    events, features, groups = feature_sets(data)
    identity = fingerprint(data, config)
    name = f"{family}_{cut}"
    output = out / name
    output.mkdir(exist_ok=True)
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        if result["fingerprint"] != identity:
            raise ValueError(
                "Existing results have another development/config fingerprint"
            )
        return result
    study = None
    if family not in {"reference", "historical", "constant"}:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{name}_{identity[:12]}",
            storage=f"sqlite:///{(output / 'study.db').resolve().as_posix()}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=config["seed"], n_startup_trials=10
            ),
        )

        def objective(trial):
            group, params = suggest_parameters(trial, family, config)
            frames, iterations = inner_predictions(
                events, features[groups[group]], family, params, cut, config
            )
            threshold, score, f1 = optimal_threshold(frames)
            trial.set_user_attr("threshold", threshold)
            trial.set_user_attr("f1", f1)
            trial.set_user_attr(
                "logloss", float(np.mean([log_loss(f.y, f.p) for f in frames]))
            )
            trial.set_user_attr("iterations", iterations)
            trial.set_user_attr("parameters", params)
            pd.concat(frames).to_csv(
                output / f"trial_{trial.number}_inner.csv", index=False
            )
            return score

        def plateau(study, trial):
            complete = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            settings = config["search"]
            if len(complete) < settings["plateau_min_trials"]:
                return
            patience = settings["plateau_patience"]
            if (
                len(complete) > patience
                and max(t.value for t in complete[-patience:])
                < max(t.value for t in complete[:-patience]) + settings["plateau_delta"]
            ):
                study.stop()

        remaining = max(0, config["trials"][family] - len(study.trials))
        study.optimize(
            objective, n_trials=remaining, timeout=timeout, callbacks=[plateau]
        )
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(f"No completed trials for {name}")
        best = max(
            completed,
            key=lambda t: (t.value, t.user_attrs["f1"], -t.user_attrs["logloss"]),
        )
        params = best.user_attrs["parameters"]
        group = best.params["feature_group"]
        threshold = best.user_attrs["threshold"]
        frames = pd.read_csv(output / f"trial_{best.number}_inner.csv")
        study.trials_dataframe().to_csv(output / "trials.csv", index=False)
        inner_score = best.value
    else:
        params, group = {"half_life": 180}, "reference"
        if family == "reference":
            inner_frames, _ = inner_predictions(
                events, features[groups[group]], family, params, cut, config
            )
        else:
            inner_frames = []
            for fold, (start, stop) in enumerate(config["inner_folds"][cut], 1):
                mask = events.dia_visita.between(start + 1, stop)
                frame = events.loc[mask, ["id", "dia_visita"]].copy()
                frame["y"] = events.loc[mask, "incidencia_compra"]
                frame["p"] = (
                    events.loc[mask, "tasa_previa"]
                    if family == "historical"
                    else float(
                        events.loc[
                            events.dia_visita.le(start), "incidencia_compra"
                        ].mean()
                    )
                )
                frame["inner_fold"] = fold
                inner_frames.append(frame)
        threshold, inner_score, _ = optimal_threshold(inner_frames)
        frames = pd.concat(inner_frames)
    frames.to_csv(output / "inner_predictions.csv", index=False)
    selected = None
    if family not in {"historical", "constant"}:
        model, selected = fit_classic(
            events, features[groups[group]], family, params, cut, config
        )
    rows = []
    for split, start, stop in [("train", 1, cut), ("validation", cut + 1, end)]:
        if family in {"historical", "constant"}:
            mask = events.dia_visita.between(start, stop)
            frame = events.loc[mask, ["id", "dia_visita"]].copy()
            frame["y"] = events.loc[mask, "incidencia_compra"]
            frame["p"] = (
                events.loc[mask, "tasa_previa"]
                if family == "historical"
                else float(
                    events.loc[events.dia_visita.le(cut), "incidencia_compra"].mean()
                )
            )
        else:
            frame = predictions(events, model, features[groups[group]], start, stop, 0)
        frame.to_csv(output / f"{split}_predictions.csv", index=False)
        rows.append(
            {
                "model": family,
                "cut": cut,
                "end": end,
                "split": split,
                "threshold": threshold,
                "n": len(frame),
                **classification_metrics(frame.y, frame.p, threshold),
            }
        )
    if family not in {"historical", "constant"}:
        joblib.dump(
            {
                "model": model,
                "feature_builder_version": "review_v1",
                "columns": groups[group],
                "threshold": threshold,
                "training_end": cut,
            },
            output / "model.joblib",
        )
    result = {
        "model": family,
        "cut": cut,
        "end": end,
        "fingerprint": identity,
        "parameters": params,
        "feature_group": group,
        "columns": groups[group],
        "threshold": threshold,
        "inner_score": inner_score,
        "iterations": selected,
        "trials": len(study.trials) if study is not None else 0,
        "seconds": perf_counter() - started,
        "metrics": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score_review_visits(bundle, history, visits):
    """Score one future store-day using the exact versioned feature constructor."""
    if bundle.get("feature_builder_version") != "review_v1":
        raise ValueError("Unsupported feature builder version")
    if visits.empty or visits.dia_visita.nunique() != 1:
        raise ValueError("Scoring requires exactly one nonempty future day")
    if not history.empty and history.dia_visita.max() >= visits.dia_visita.min():
        raise ValueError("History must precede scoring day")
    pending = visits.copy()
    pending[["incidencia_compra", "id_marca", "cantidad"]] = np.nan
    joined = pd.concat([history, pending], ignore_index=True)
    if joined.duplicated(["id", "dia_visita"]).any():
        raise ValueError("Duplicate visit key")
    events, features = reviewed_features(joined)
    columns = bundle["columns"]
    if any(c not in features for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Incompatible feature schema")
    mask = events.dia_visita.eq(visits.dia_visita.iloc[0])
    result = events.loc[mask, ["id", "dia_visita"]].copy()
    result["probability"] = bundle["model"].predict_proba(features.loc[mask, columns])[
        :, 1
    ]
    result["prediction"] = result.probability.ge(bundle["threshold"]).astype(int)
    return result
