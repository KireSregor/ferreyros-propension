"""Deterministic nested LightGBM tuning and visit-level scoring."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL

OUTER_FOLDS = [(296, 404), (404, 512), (512, 620)]
INNER_FOLDS = {
    296: [(120, 180), (180, 238), (238, 296)],
    404: [(220, 296), (296, 350), (350, 404)],
    512: [(296, 404), (404, 458), (458, 512)],
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    threads: int = 4
    fold_workers: int = 3
    minimum_trials: int = 50
    patience: int = 30
    minimum_improvement: float = 0.0001
    stop_days: int = 40
    stopping_rounds: int = 60
    max_trees: int = 2000


class StagnationStopper:
    """Stop after patience completed trials without meaningful improvement."""

    def __init__(self, config):
        self.config = config
        self.anchor = np.inf
        self.last_improvement = 0
        self.completed = 0

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        self.completed += 1
        if study.best_value < self.anchor - self.config.minimum_improvement:
            self.anchor = study.best_value
            self.last_improvement = self.completed
        if (
            self.completed >= self.config.minimum_trials
            and self.completed - self.last_improvement >= self.config.patience
        ):
            study.set_user_attr("stop_reason", "stagnation")
            study.stop()


def metrics(y, p, threshold):
    labels = np.asarray(p) >= threshold
    return {
        "precision": precision_score(y, labels, zero_division=0),
        "recall": recall_score(y, labels, zero_division=0),
        "f1": f1_score(y, labels, zero_division=0),
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "logloss": log_loss(y, p),
    }


def f1_threshold(frames):
    """Maximize equal-fold mean F1; ties prefer the lower threshold."""
    candidates = np.unique(np.concatenate([frame.p.to_numpy() for frame in frames]))
    scores = np.zeros(len(candidates))
    for frame in frames:
        order = np.argsort(frame.p.to_numpy(), kind="stable")
        values = frame.p.to_numpy()[order]
        target = frame.y.to_numpy()[order]
        cumulative = np.r_[0, target.cumsum()]
        positions = np.searchsorted(values, candidates, side="left")
        tp = cumulative[-1] - cumulative[positions]
        denominator = len(target) - positions + cumulative[-1]
        scores += np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator > 0,
        ) / len(frames)
    return float(candidates[np.argmax(scores)])


def preprocessing(frame):
    categorical = [column for column in frame if column in CATEGORICAL]
    numeric = [column for column in frame if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )


def estimator(parameters, trees, config):
    parameters = {key: value for key, value in parameters.items() if key != "half_life"}
    return LGBMClassifier(
        **parameters,
        n_estimators=trees,
        objective="binary",
        verbosity=-1,
        random_state=config.seed,
        n_jobs=config.threads,
        deterministic=True,
        force_col_wise=True,
    )


def train_window(events, features, parameters, cut, config):
    """Select tree count on the last 40 training days, then refit full training."""
    target = events.incidencia_compra
    inner = events.dia_visita.le(cut - config.stop_days)
    stop = events.dia_visita.between(cut - config.stop_days + 1, cut)
    train = events.dia_visita.le(cut)

    def sample_weights(mask, anchor):
        weight = np.exp2(
            (events.loc[mask, "dia_visita"].to_numpy() - anchor)
            / parameters["half_life"]
        )
        return weight / weight.mean()

    transformer = preprocessing(features.loc[inner])
    a = transformer.fit_transform(features.loc[inner])
    b = transformer.transform(features.loc[stop])
    selector = estimator(parameters, config.max_trees, config)
    selector.fit(
        a,
        target[inner],
        sample_weight=sample_weights(inner, cut - config.stop_days),
        eval_X=b,
        eval_y=target[stop],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(config.stopping_rounds, verbose=False),
            log_evaluation(0),
        ],
    )
    trees = max(1, selector.best_iteration_)
    model = make_pipeline(
        preprocessing(features.loc[train]), estimator(parameters, trees, config)
    )
    model.fit(
        features.loc[train],
        target[train],
        lgbmclassifier__sample_weight=sample_weights(train, cut),
    )
    return {
        "model": model,
        "columns": list(features),
        "training_end": cut,
        "iterations": trees,
        "executed_selection_iterations": len(
            selector.evals_result_["valid_0"]["binary_logloss"]
        ),
    }


def predict_frame(events, features, bundle, start, end):
    mask = events.dia_visita.between(start, end)
    frame = (
        events.loc[mask, ["id", "dia_visita", "incidencia_compra"]]
        .rename(columns={"incidencia_compra": "y"})
        .copy()
    )
    frame["p"] = bundle["model"].predict_proba(features.loc[mask, bundle["columns"]])[
        :, 1
    ]
    return frame


def inner_predictions(events, features, parameters, cut, config):
    def fit_fold(item):
        number, (train_end, end) = item
        bundle = train_window(events, features, parameters, train_end, config)
        frame = predict_frame(events, features, bundle, train_end + 1, end)
        frame["inner_fold"] = number
        return frame

    windows = list(enumerate(INNER_FOLDS[cut], 1))
    if config.fold_workers == 1:
        return list(map(fit_fold, windows))
    # Ordered results keep threshold selection and objective aggregation stable.
    with ThreadPoolExecutor(max_workers=config.fold_workers) as executor:
        return list(executor.map(fit_fold, windows))


def suggest(trial):
    depth = trial.suggest_int("max_depth", 2, 8)
    return {
        "max_depth": depth,
        "num_leaves": trial.suggest_int("num_leaves", 4, min(64, 2**depth)),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 600),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "half_life": trial.suggest_float("half_life", 90, 720, log=True),
    }


def tune_fold(events, features, cut, config):
    """A new in-memory study; no timeout, trial cap, pruning or saved winners."""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=15),
    )

    def objective(trial):
        parameters = suggest(trial)
        frames = inner_predictions(events, features, parameters, cut, config)
        trial.set_user_attr(
            "fold_logloss", [float(log_loss(frame.y, frame.p)) for frame in frames]
        )
        return float(np.mean(trial.user_attrs["fold_logloss"]))

    def progress(study, trial):
        if (trial.number + 1) % 10 == 0:
            print(
                f"Corte {cut}: {trial.number + 1} trials; mejor log loss {study.best_value:.6f}",
                flush=True,
            )

    started = perf_counter()
    study.optimize(objective, callbacks=[StagnationStopper(config), progress], n_jobs=1)
    study.set_user_attr("seconds", perf_counter() - started)
    return study


def evaluate_bundle(events, features, bundle, start, end):
    frame = predict_frame(events, features, bundle, start, end)
    return {"n": len(frame), **metrics(frame.y, frame.p, bundle["threshold"])}, frame


def compact_features(events, features, config, count=17):
    """Rank and remove redundant features using only days 1–120."""
    parameters = {
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "reg_lambda": 10.0,
        "half_life": 180.0,
    }
    pilot = train_window(events, features, parameters, 120, config)
    past = features.loc[events.dia_visita.le(120)]
    sample = past.sample(n=min(1000, len(past)), random_state=config.seed)
    transformer, model = pilot["model"].steps[0][1], pilot["model"].steps[-1][1]
    contributions = model.predict(transformer.transform(sample), pred_contrib=True)[
        :, :-1
    ]
    importance = {column: 0.0 for column in features}
    originals = sorted(features, key=len, reverse=True)
    for encoded, value in zip(
        transformer.get_feature_names_out(),
        np.abs(contributions).mean(axis=0),
        strict=True,
    ):
        raw = encoded.split("__", 1)[1]
        column = next(
            name for name in originals if raw == name or raw.startswith(name + "_")
        )
        importance[column] += float(value)
    ranking = pd.DataFrame(
        {"variable": list(importance), "pilot_shap": list(importance.values())}
    )
    ranking = ranking.sort_values(["pilot_shap", "variable"], ascending=[False, True])
    candidates = ranking.variable.head(25).tolist()
    correlation = (
        features.loc[events.dia_visita.le(120), candidates]
        .corr(method="spearman")
        .abs()
    )
    columns = []
    for column in candidates:
        if not any(correlation.loc[column, previous] >= 0.90 for previous in columns):
            columns.append(column)
    columns = columns[:count]
    return features[columns].copy(), ranking.reset_index(drop=True)


def historical_frames(events, features, cut):
    frames = []
    for train_end, end in INNER_FOLDS[cut]:
        mask = events.dia_visita.between(train_end + 1, end)
        frame = events.loc[mask, ["incidencia_compra"]].rename(
            columns={"incidencia_compra": "y"}
        )
        frame["p"] = features.loc[mask, "tasa_previa"]
        frames.append(frame)
    return frames


def score_visits(bundle, history, visits):
    """Score one future day using complete prior store history, ignoring current outcomes."""
    from .feature_research import reviewed_features

    if (
        visits.dia_visita.nunique() != 1
        or history.dia_visita.max() >= visits.dia_visita.min()
    ):
        raise ValueError("Provide one scoring day and strictly earlier store history")
    future = visits.copy()
    future[["incidencia_compra", "id_marca", "cantidad"]] = np.nan
    joined = (
        pd.concat([history, future], ignore_index=True)
        .sort_values(["id", "dia_visita"])
        .reset_index(drop=True)
    )
    events, features = reviewed_features(joined)
    mask = events.dia_visita.eq(visits.dia_visita.iloc[0])
    result = events.loc[mask, ["id", "dia_visita"]].copy()
    result["probability"] = bundle["model"].predict_proba(
        features.loc[mask, bundle["columns"]]
    )[:, 1]
    result["prediction"] = (result.probability >= bundle["threshold"]).astype(int)
    return result
