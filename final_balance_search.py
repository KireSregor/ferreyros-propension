"""Final temporal search: normalized prices, recency weighting, balanced threshold.

The later check is previously inspected, so success would remain exploratory.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

from ferreyros_propension.data import load_data
from ferreyros_propension.feature_research import monthly_trend_features


def scores(y, p, threshold):
    labels = p >= threshold
    return {
        "precision": precision_score(y, labels),
        "recall": recall_score(y, labels),
        "f1": f1_score(y, labels),
        "threshold": float(threshold),
    }


OUT = Path("outputs/regularized_search/final_balance")


def main(grid=False, warm_tpe=False, finalize=False, split480=False):
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_data("Dataset/compras_data.csv")
    events, x = monthly_trend_features(raw[raw.dia_visita.le(730 if finalize else 620)])
    remove = [
        c
        for c in x
        if c.startswith("fase_")
        or c
        in {
            "cliente_conocido",
            "dia_relativo",
            "antiguedad_observada",
            "visitas_previas",
            "compras_previas",
        }
    ]
    x = x.drop(columns=remove)
    # Daily prices are shared across customers: normalize using earlier days only.
    extra = {}
    for brand in range(1, 6):
        column = f"precio_marca_{brand}"
        daily = events.groupby("dia_visita")[column].first().sort_index()
        prior = daily.shift().rolling(60, min_periods=5).median()
        extra[f"price_vs_prior60_{brand}"] = events[column] / events.dia_visita.map(
            prior
        )
    x = pd.concat([x, pd.DataFrame(extra, index=events.index)], axis=1)
    candidates = []

    def fit(params, half_life, cut, end):
        train = events.dia_visita.between(1 if split480 else cut - 269, cut)
        inner = train & events.dia_visita.le(cut - 45)
        stop = events.dia_visita.between(cut - 44, cut)
        valid = events.dia_visita.between(cut + 1, end)

        def frames(mask, other):
            a, b = x[mask].copy(), x[other].copy()
            for column in a.select_dtypes(include=["object", "string"]).columns:
                categories = a[column].unique()
                a[column] = pd.Categorical(a[column], categories=categories)
                b[column] = pd.Categorical(b[column], categories=categories)
            return a, b

        a, b = frames(inner, stop)
        weights = np.exp2((events.loc[inner, "dia_visita"] - (cut - 45)) / half_life)
        model = lgb.LGBMClassifier(
            **params, n_estimators=1000, random_state=915, n_jobs=4, verbosity=-1
        )
        model.fit(
            a,
            events.loc[inner, "incidencia_compra"],
            sample_weight=weights,
            eval_set=[(b, events.loc[stop, "incidencia_compra"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iterations = model.best_iteration_
        a, b = frames(train, valid)
        model = lgb.LGBMClassifier(
            **params, n_estimators=iterations, random_state=915, n_jobs=4, verbosity=-1
        )
        model.fit(
            a,
            events.loc[train, "incidencia_compra"],
            sample_weight=np.exp2((events.loc[train, "dia_visita"] - cut) / half_life),
        )
        return {
            "cut": cut,
            "iterations": iterations,
            "train_y": events.loc[train, "incidencia_compra"].to_numpy(),
            "train_p": model.predict_proba(a)[:, 1],
            "y": events.loc[valid, "incidencia_compra"].to_numpy(),
            "p": model.predict_proba(b)[:, 1],
            "model": model,
        }

    if finalize:
        chosen = json.loads(Path("outputs/final_model/configuration.json").read_text())[
            "selected"
        ]
        validation = fit(
            chosen["parameters"],
            chosen["half_life"],
            480 if split480 else 512,
            580 if split480 else 620,
        )
        if split480:
            precision, recall, thresholds = precision_recall_curve(
                validation["y"], validation["p"]
            )
            chosen["threshold"] = float(
                thresholds[np.argmax(np.minimum(precision[:-1], recall[:-1]))]
            )
        final = fit(
            chosen["parameters"], chosen["half_life"], 580 if split480 else 620, 730
        )
        rows = []
        for label, y, p in [
            ("train_243_512", validation["train_y"], validation["train_p"]),
            ("validation_513_620", validation["y"], validation["p"]),
            ("final_train_351_620", final["train_y"], final["train_p"]),
            ("test_621_730_previously_seen", final["y"], final["p"]),
        ]:
            if split480:
                label = {
                    "train_243_512": "train_1_480",
                    "validation_513_620": "validation_481_580",
                    "final_train_351_620": "final_train_1_580",
                    "test_621_730_previously_seen": "test_581_730_previously_seen",
                }[label]
            rows.append(
                {
                    "period": label,
                    "n": len(y),
                    "purchase_rate": float(np.mean(y)),
                    **scores(y, p, chosen["threshold"]),
                    "auc": roc_auc_score(y, p),
                    "ap": average_precision_score(y, p),
                    "logloss": log_loss(y, p),
                }
            )
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
        (OUT / "configuration.json").write_text(
            json.dumps(
                {
                    "selected": chosen,
                    "final_iterations": final["iterations"],
                    "columns": list(x),
                    "scope": "Previously inspected621-730, not independent test; threshold fixed before evaluation.",
                },
                indent=2,
            )
        )
        joblib.dump(
            {
                "model": final["model"],
                "columns": list(x),
                "threshold": chosen["threshold"],
                "parameters": chosen["parameters"],
                "half_life": chosen["half_life"],
                "feature_function": "monthly_trend_features plus causal60dayprice normalization as in final_balance_search.py",
                "warning": "Requires exact feature construction; not compatible with core score_visits.",
            },
            OUT / "model.joblib",
        )
        pd.DataFrame({"y": final["y"], "p": final["p"]}).to_csv(
            OUT / "test_predictions.csv", index=False
        )
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        return

    def objective(trial):
        params = (
            {}
            if grid or warm_tpe
            else {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.07, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 3, 12),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "min_child_samples": trial.suggest_int("min_child_samples", 200, 900),
                "reg_lambda": trial.suggest_float("reg_lambda", 10, 500, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1, 60, log=True),
            }
        )
        half_life = (
            90
            if grid or warm_tpe
            else trial.suggest_categorical("half_life", [45, 90, 180, 10000])
        )
        if warm_tpe:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.05, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 1, 3),
                "min_child_samples": trial.suggest_int("min_child_samples", 250, 1000),
                "reg_lambda": trial.suggest_float("reg_lambda", 30, 1000, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 5, 150, log=True),
            }
            params["num_leaves"] = 2 ** params["max_depth"]
            half_life = trial.suggest_float("half_life", 30, 365, log=True)
        if grid:
            params = {
                "learning_rate": trial.suggest_categorical(
                    "grid_learning_rate", [0.015, 0.035, 0.07]
                ),
                "max_depth": trial.suggest_categorical("grid_depth", [1, 2, 3]),
                "min_child_samples": trial.suggest_categorical(
                    "grid_min_child", [200, 500, 900]
                ),
                "reg_lambda": trial.suggest_categorical("grid_lambda", [10, 300]),
                "reg_alpha": trial.suggest_categorical("grid_alpha", [1, 50]),
            }
            params["num_leaves"] = 2 ** params["max_depth"]
            half_life = trial.suggest_categorical("grid_half_life", [90, 10000])
        folds = [
            fit(params, half_life, cut, end) for cut, end in [(296, 404), (404, 512)]
        ]
        y, p = (
            np.concatenate([f["y"] for f in folds]),
            np.concatenate([f["p"] for f in folds]),
        )
        precision, recall, thresholds = precision_recall_curve(y, p)
        threshold = float(
            thresholds[np.argmax(np.minimum(precision[:-1], recall[:-1]))]
        )
        metrics = []
        for fold in folds:
            train = scores(fold["train_y"], fold["train_p"], threshold)
            valid = scores(fold["y"], fold["p"], threshold)
            gap = max(
                0.0,
                train["precision"] - valid["precision"],
                train["recall"] - valid["recall"],
            )
            metrics.append(
                {
                    "cut": fold["cut"],
                    "iterations": fold["iterations"],
                    "train": train,
                    "validation": valid,
                    "gap": gap,
                }
            )
        value = min(
            min(m["validation"]["precision"], m["validation"]["recall"])
            - max(0.0, m["gap"] - 0.05)
            for m in metrics
        )
        candidates.append(
            {
                "trial": trial.number,
                "parameters": params,
                "half_life": half_life,
                "threshold": threshold,
                "folds": metrics,
                "objective": value,
            }
        )
        print("trial", trial.number, round(value, 4), flush=True)
        (OUT / "trials.json").write_text(json.dumps(candidates, indent=2))
        return value

    search_space = {
        "grid_learning_rate": [0.015, 0.035, 0.07],
        "grid_depth": [1, 2, 3],
        "grid_min_child": [200, 500, 900],
        "grid_lambda": [10, 300],
        "grid_alpha": [1, 50],
        "grid_half_life": [90, 10000],
    }
    sampler = (
        optuna.samplers.GridSampler(search_space, seed=915)
        if grid
        else optuna.samplers.TPESampler(seed=915)
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if warm_tpe:
        winner = json.loads(
            Path("outputs/regularized_search/exhaustive_grid/summary.json").read_text()
        )["selected"]
        start_parameters = {
            k: v for k, v in winner["parameters"].items() if k != "num_leaves"
        }
        start_parameters["half_life"] = winner["half_life"]
        study.enqueue_trial(start_parameters)
    study.optimize(objective, n_trials=216 if grid else (60 if warm_tpe else 40))
    chosen = max(candidates, key=lambda row: row["objective"])
    later = fit(chosen["parameters"], chosen["half_life"], 512, 620)
    result = {
        "selected": chosen,
        "check": {
            "train": scores(later["train_y"], later["train_p"], chosen["threshold"]),
            "validation": scores(later["y"], later["p"], chosen["threshold"]),
            "iterations": later["iterations"],
        },
        "seconds": perf_counter() - started,
        "exhaustive_grid": search_space if grid else None,
        "scope": "Selection297-512; previously inspected513-620; no621+. Exploratory, not independent confirmation.",
    }
    pd.DataFrame({"y": later["y"], "p": later["p"]}).to_csv(
        OUT / "check.csv", index=False
    )
    (OUT / "trials.json").write_text(json.dumps(candidates, indent=2))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--warm-tpe", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--split480", action="store_true")
    arguments = parser.parse_args()
    if arguments.grid:
        OUT = Path("outputs/regularized_search/exhaustive_grid")
    if arguments.warm_tpe:
        OUT = Path("outputs/regularized_search/warm_tpe")
    if arguments.finalize:
        OUT = Path("outputs/final_model")
    if arguments.split480:
        arguments.finalize = True
        OUT = Path("outputs/final_model/split480")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with threadpool_limits(limits=4):
        main(arguments.grid, arguments.warm_tpe, arguments.finalize, arguments.split480)
