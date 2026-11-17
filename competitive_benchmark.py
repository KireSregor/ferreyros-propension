"""Compare models with nested temporal validation and a chronological ensemble."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ferreyros_propension.data import load_data
from ferreyros_propension.feature_research import monthly_trend_features
from final_audit import metrics

OUTPUT = Path("outputs/competitive_benchmark")
OUTER_FOLDS = [(404, 512), (512, 620)]
INNER_FOLDS = {
    404: [(220, 296), (296, 404)],
    512: [(296, 404), (404, 512)],
}


def load_features():
    raw = load_data("Dataset/compras_data.csv")
    events, features = monthly_trend_features(raw.loc[raw.dia_visita.le(620)])
    excluded = {"cliente_conocido", "dia_relativo"}
    columns = [c for c in features if c not in excluded and not c.startswith("fase_")]
    features = features[columns].apply(pd.to_numeric, errors="coerce")
    return events, features.replace([np.inf, -np.inf], np.nan)


def suggest(trial, family):
    common = {
        "half_life": trial.suggest_int("half_life", 120, 360, step=60),
    }
    if family == "logistic":
        common["C"] = trial.suggest_float("C", 0.0001, 0.1, log=True)
    else:
        common["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.02, 0.08, log=True
        )
        common["depth"] = trial.suggest_int("depth", 2, 5)
        common["regularization"] = trial.suggest_float(
            "regularization", 1, 100, log=True
        )
        if family == "lightgbm":
            common["min_samples"] = trial.suggest_int("min_samples", 150, 700, step=50)
        if family == "xgboost":
            common["min_samples"] = trial.suggest_int("min_samples", 10, 150, step=20)
    return common


def make_model(family, params, iterations=600, stopping=False):
    if family == "logistic":
        return make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            StandardScaler(),
            LogisticRegression(C=params["C"], max_iter=350, random_state=42),
        )
    if family == "lightgbm":
        return LGBMClassifier(
            n_estimators=iterations,
            learning_rate=params["learning_rate"],
            num_leaves=min(2 ** params["depth"], 16),
            max_depth=params["depth"],
            min_child_samples=params["min_samples"],
            reg_lambda=params["regularization"],
            n_jobs=2,
            verbosity=-1,
            random_state=42,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=iterations,
            learning_rate=params["learning_rate"],
            max_depth=params["depth"],
            min_child_weight=params["min_samples"],
            reg_lambda=params["regularization"],
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=2,
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=35 if stopping else None,
        )
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["regularization"],
        thread_count=2,
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    )


def fit_fold(events, features, family, params, cut, end):
    target = events.incidencia_compra
    train = events.dia_visita.le(cut)
    valid = events.dia_visita.between(cut + 1, end)
    weights = np.exp2((events.loc[train, "dia_visita"] - cut) / params["half_life"])
    iterations = None
    if family != "logistic":
        inner = events.dia_visita.le(cut - 40)
        stop = events.dia_visita.between(cut - 39, cut)
        inner_weights = np.exp2(
            (events.loc[inner, "dia_visita"] - cut + 40) / params["half_life"]
        )
        estimator = make_model(family, params, stopping=True)
        fit_args = {
            "sample_weight": inner_weights,
            "eval_set": [(features[stop], target[stop])],
        }
        if family == "lightgbm":
            fit_args["callbacks"] = [
                early_stopping(35, verbose=False),
                log_evaluation(0),
            ]
        elif family == "xgboost":
            fit_args["verbose"] = False
        else:
            fit_args["early_stopping_rounds"] = 35
        estimator.fit(features[inner], target[inner], **fit_args)
        if family == "lightgbm":
            iterations = max(1, estimator.best_iteration_)
        elif family == "xgboost":
            iterations = estimator.best_iteration + 1
        else:
            iterations = max(1, estimator.best_iteration_ + 1)
    model = make_model(family, params, iterations=iterations or 600)
    fit_args = (
        {"logisticregression__sample_weight": weights}
        if family == "logistic"
        else {"sample_weight": weights}
    )
    model.fit(features[train], target[train], **fit_args)
    return (
        model.predict_proba(features[train])[:, 1],
        model.predict_proba(features[valid])[:, 1],
        iterations,
    )


def threshold_for(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    score = np.minimum(precision[:-1], recall[:-1])
    best = int(np.argmax(score))
    return float(thresholds[best]), float(score[best])


def run_task(family, cut, end, trials):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    started = perf_counter()
    events, features = load_features()
    target = events.incidencia_compra
    cache = {}

    def objective(trial):
        params = suggest(trial, family)
        predictions = []
        truth = []
        for inner_cut, inner_end in INNER_FOLDS[cut]:
            _, probability, _ = fit_fold(
                events, features, family, params, inner_cut, inner_end
            )
            mask = events.dia_visita.between(inner_cut + 1, inner_end)
            predictions.extend(probability)
            truth.extend(target[mask])
        threshold, score = threshold_for(np.asarray(truth), np.asarray(predictions))
        trial.set_user_attr("threshold", threshold)
        cache[trial.number] = np.asarray(predictions)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    train_p, valid_p, iterations = fit_fold(
        events, features, family, best.params, cut, end
    )
    masks = {
        "train": events.dia_visita.le(cut),
        "validation": events.dia_visita.between(cut + 1, end),
    }
    rows = []
    for split, probability in [("train", train_p), ("validation", valid_p)]:
        rows.append(
            {
                "model": family,
                "cut": cut,
                "end": end,
                "split": split,
                "threshold": best.user_attrs["threshold"],
                "n": int(masks[split].sum()),
                **metrics(
                    target[masks[split]], probability, best.user_attrs["threshold"]
                ),
            }
        )
    inner = pd.concat(
        [
            events.loc[
                events.dia_visita.between(start + 1, stop),
                ["id", "dia_visita", "incidencia_compra"],
            ]
            for start, stop in INNER_FOLDS[cut]
        ]
    ).copy()
    inner["p"] = cache[best.number]
    valid = events.loc[
        masks["validation"], ["id", "dia_visita", "incidencia_compra"]
    ].copy()
    valid["p"] = valid_p
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inner.to_csv(OUTPUT / f"{family}_{cut}_inner.csv", index=False)
    valid.to_csv(OUTPUT / f"{family}_{cut}_validation.csv", index=False)
    study.trials_dataframe().to_csv(OUTPUT / f"{family}_{cut}_trials.csv", index=False)
    details = {
        "model": family,
        "cut": cut,
        "parameters": best.params,
        "threshold": best.user_attrs["threshold"],
        "inner_score": best.value,
        "iterations": iterations,
        "seconds": perf_counter() - started,
        "rows": rows,
    }
    (OUTPUT / f"{family}_{cut}.json").write_text(json.dumps(details, indent=2))
    return details


def add_ensemble(cut, end, families):
    inner = [pd.read_csv(OUTPUT / f"{family}_{cut}_inner.csv") for family in families]
    valid = [
        pd.read_csv(OUTPUT / f"{family}_{cut}_validation.csv") for family in families
    ]
    for frames in [inner, valid]:
        keys = ["id", "dia_visita", "incidencia_compra"]
        if any(not frames[0][keys].equals(frame[keys]) for frame in frames[1:]):
            raise ValueError("Predictions are not aligned across models")
    inner_p = np.mean([frame.p for frame in inner], axis=0)
    valid_p = np.mean([frame.p for frame in valid], axis=0)
    threshold, score = threshold_for(inner[0].incidencia_compra, inner_p)
    return {
        "model": "equal_ensemble",
        "cut": cut,
        "end": end,
        "split": "validation",
        "threshold": threshold,
        "n": len(valid_p),
        "inner_score": score,
        **metrics(valid[0].incidencia_compra, valid_p, threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = perf_counter()
    families = ["logistic", "lightgbm", "xgboost", "catboost"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_task, family, cut, end, args.trials)
            for cut, end in OUTER_FOLDS
            for family in families
        ]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["rows"])
            print(
                f"{result['model']} cut={result['cut']}: {result['seconds']:.1f}s",
                flush=True,
            )
    rows.extend(add_ensemble(cut, end, families) for cut, end in OUTER_FOLDS)
    pd.DataFrame(rows).to_csv(OUTPUT / "metrics.csv", index=False)
    (OUTPUT / "runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "workers": args.workers,
                "threads_per_tree": 2,
                "trials_per_study": args.trials,
            },
            indent=2,
        )
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"TOTAL_SECONDS={perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
