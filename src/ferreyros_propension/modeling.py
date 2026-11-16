"""Modelos, validación temporal y búsqueda persistente con MLflow."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, features, load_data
from .evaluation import bootstrap_delta, daily_capacity, metrics


def make_model(model_name, parameters, columns, seed=42, threads=4):
    """Cada familia recibe su tratamiento categórico y límite de hilos."""
    categorical_columns = [c for c in CATEGORICAL if c in columns]
    numeric_columns = [c for c in columns if c not in categorical_columns]
    if model_name == "catboost":
        # CatBoost maneja NaN numéricos y categorías sin one-hot externo.
        return CatBoostClassifier(
            **parameters,
            cat_features=categorical_columns,
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
            task_type="CPU",
        )
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler() if model_name == "logistic" else "passthrough",
                ),
                numeric_columns,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
    estimators = {
        "logistic": lambda: LogisticRegression(
            **parameters, max_iter=1500, random_state=seed
        ),
        "random_forest": lambda: RandomForestClassifier(
            **parameters, random_state=seed, n_jobs=threads
        ),
        "hist": lambda: HistGradientBoostingClassifier(
            **parameters, random_state=seed, early_stopping=False
        ),
        "xgboost": lambda: XGBClassifier(
            **parameters,
            random_state=seed,
            n_jobs=threads,
            tree_method="hist",
            device="cpu",
            verbosity=0,
        ),
        "lightgbm": lambda: LGBMClassifier(
            **parameters, random_state=seed, n_jobs=threads, verbosity=-1
        ),
    }
    if model_name not in estimators:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return make_pipeline(preprocessor, estimators[model_name]())


def suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    """Define espacios conservadores para un panel de 500 clientes.

    Profundidades y hojas moderadas limitan sobreajuste a visitas repetidas.
    Las escalas logarítmicas exploran órdenes de magnitud de regularización.
    El número de árboles se optimiza explícitamente; no hay early stopping
    sobre el fold externo, pues contaminaría la evaluación.
    """
    if model_name == "logistic":
        return {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": 200,
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 150),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

    if model_name == "hist":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 350, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 25),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.1, 30.0, log=True
            ),
        }

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        }

    if model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"Modelo no soportado: {model_name}")

    parameters = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    if model_name == "xgboost":
        parameters.update(
            {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 5.0, 100.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        )
    else:
        parameters.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 5, 25),
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            }
        )
    return parameters


def cross_validate(model_name, parameters, events, predictors, folds, config):
    """Evalúa ventanas expansivas con preprocesamiento ajustado solo en train.

    Las variables históricas se actualizaron secuencialmente antes de cada
    visita. Las etiquetas externas nunca participan en el ajuste del modelo.
    """
    fold_results = []
    prediction_frames = []
    for fold_number, (training_end, validation_end) in enumerate(folds):
        training_mask = events.dia_visita.le(training_end)
        validation_mask = events.dia_visita.between(training_end + 1, validation_end)
        training_target = events.loc[training_mask, "incidencia_compra"]
        validation_target = events.loc[validation_mask, "incidencia_compra"]
        if training_target.nunique() != 2 or validation_target.nunique() != 2:
            raise ValueError(f"El fold {fold_number} no contiene ambas clases.")

        model = make_model(
            model_name, parameters, list(predictors), config["seed"], config["threads"]
        )
        model.fit(predictors[training_mask], training_target)
        probabilities = model.predict_proba(predictors[validation_mask])[:, 1]
        fold_results.append(
            {
                "fold": fold_number,
                "train_end": training_end,
                "validation_end": validation_end,
                **metrics(validation_target, probabilities),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": events.loc[validation_mask, "id"],
                    "dia_visita": events.loc[validation_mask, "dia_visita"],
                    "y": validation_target,
                    "p": probabilities,
                    "baseline": events.loc[validation_mask, "tasa_previa"],
                    "fold": fold_number,
                }
            )
        )
    return pd.DataFrame(fold_results), pd.concat(prediction_frames)


def optimize_family(
    model_name,
    events,
    predictors,
    folds,
    config,
    output_dir,
    version,
    phase="search",
    budget=None,
):
    """Reanuda una búsqueda y vincula cada trial a su run de MLflow.

    Los trials fallidos no agotan el presupuesto de trials completos.
    Todos los trials completan los folds para una comparación homogénea.
    """
    storage = "sqlite:///" + (output_dir / "optuna.db").as_posix()
    study = optuna.create_study(
        study_name=f"{model_name}_{version}_{phase}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"]),
    )

    def objective(trial):
        parameters = suggest_parameters(trial, model_name)
        started_at = time.perf_counter()
        with mlflow.start_run(run_name=f"{phase}_{model_name}_{trial.number}") as run:
            mlflow.set_tags(
                {
                    "family": model_name,
                    "version": version,
                    "phase": phase,
                    "study": study.study_name,
                    "trial": trial.number,
                    "device": "cpu",
                }
            )
            mlflow.log_params(parameters)
            mlflow.log_params({"seed": config["seed"], "folds": str(folds)})
            results, _ = cross_validate(
                model_name, parameters, events, predictors, folds, config
            )
            for record in results.to_dict("records"):
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in record.items()
                        if key not in {"fold", "train_end", "validation_end"}
                    },
                    step=int(record["fold"]),
                )
            mlflow.log_metrics(
                {
                    "cv_ap_mean": results.ap.mean(),
                    "cv_ap_std": results.ap.std(),
                    "cv_brier_mean": results.brier.mean(),
                    "seconds": time.perf_counter() - started_at,
                }
            )
            mlflow.log_dict(
                {
                    "columns": list(predictors),
                    "fold_metrics": results.to_dict("records"),
                },
                "validation.json",
            )
            trial.set_user_attr("run_id", run.info.run_id)
            trial.set_user_attr("full_params", parameters)
            return results.ap.mean()

    target_trials = budget if budget is not None else config["trials"][model_name]
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, target_trials - completed)
    print(f"{phase} / {model_name}: {remaining} trials pendientes", flush=True)
    study.optimize(objective, n_trials=remaining)
    return study.best_trial.user_attrs["full_params"]


def probability_logits(probabilities):
    """Transforma probabilidades finitas a log-odds para calibración sigmoide."""
    clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_calibrator(oof_predictions, output_dir):
    """Selecciona calibración cronológica en desarrollo; no consulta el test."""
    from sklearn.metrics import log_loss

    calibration_mask = oof_predictions.fold.le(1)
    comparison = oof_predictions.loc[~calibration_mask]
    calibrator = LogisticRegression(C=1000)
    calibrator.fit(
        probability_logits(oof_predictions.loc[calibration_mask, "p"]),
        oof_predictions.loc[calibration_mask, "y"],
    )
    calibrated = calibrator.predict_proba(probability_logits(comparison.p))[:, 1]
    rows = [
        {"variant": "raw", **metrics(comparison.y, comparison.p)},
        {"variant": "sigmoid", **metrics(comparison.y, calibrated)},
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "tables/calibration_selection.csv", index=False
    )
    if log_loss(comparison.y, calibrated) >= log_loss(comparison.y, comparison.p):
        return None
    # Una vez elegido el procedimiento, se ajusta con todo el OOF disponible.
    calibrator.fit(probability_logits(oof_predictions.p), oof_predictions.y)
    return calibrator


def evaluate_final(
    model_name,
    parameters,
    events,
    predictors,
    oof_predictions,
    config,
    output_dir,
    version,
):
    """Congela el candidato y evalúa el benchmark previamente consultado."""
    development_mask = events.dia_visita.le(config["development_end"])
    benchmark_mask = ~development_mask
    calibrator = fit_calibrator(oof_predictions, output_dir)
    model = make_model(
        model_name, parameters, list(predictors), config["seed"], config["threads"]
    )
    model.fit(
        predictors[development_mask], events.loc[development_mask, "incidencia_compra"]
    )
    raw_probabilities = model.predict_proba(predictors[benchmark_mask])[:, 1]
    probabilities = raw_probabilities
    if calibrator is not None:
        probabilities = calibrator.predict_proba(probability_logits(raw_probabilities))[
            :, 1
        ]

    predictions = events.loc[benchmark_mask, ["id", "dia_visita"]].copy()
    predictions["y"] = events.loc[benchmark_mask, "incidencia_compra"]
    predictions["p"] = probabilities
    predictions["raw_p"] = raw_probabilities
    predictions["baseline"] = events.loc[benchmark_mask, "tasa_previa"]
    predictions.to_csv(output_dir / "tables/test_predictions.csv", index=False)
    constant = np.full(
        len(predictions), events.loc[development_mask, "incidencia_compra"].mean()
    )
    comparison = pd.DataFrame(
        [
            {"model": model_name, **metrics(predictions.y, probabilities)},
            {"model": "historical", **metrics(predictions.y, predictions.baseline)},
            {"model": "constant", **metrics(predictions.y, constant)},
        ]
    )
    comparison.to_csv(output_dir / "tables/test_metrics.csv", index=False)
    (output_dir / "uncertainty.json").write_text(
        json.dumps(bootstrap_delta(predictions), indent=2)
    )
    (output_dir / "daily_capacity.json").write_text(
        json.dumps(daily_capacity(predictions), indent=2)
    )
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "columns": list(predictors),
        "version": version,
        "family": model_name,
        "params": parameters,
        "development_end": config["development_end"],
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    with mlflow.start_run(run_name="final_" + model_name):
        mlflow.set_tags(
            {
                "phase": "final",
                "version": version,
                "family": model_name,
                "benchmark": "previously_seen",
            }
        )
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics(predictions.y, probabilities))
        mlflow.log_artifact(str(output_dir / "model.joblib"))
        mlflow.log_artifact(str(output_dir / "tables/test_predictions.csv"))
    return bundle


def diagnose_finalists(
    ranking, parameters, events, predictors, config, output_dir, version
):
    """Ablaciones sin retuning y búsquedas internas de los dos finalistas.

    La selección de familias ya observó los folds externos: este diagnóstico
    anidado no es una estimación completamente independiente de selección.
    """
    from .data import PRICES, PROFILE, PROMOS

    winner = ranking.index[0]
    commercial = PRICES + PROMOS + ["precio_min", "rango_precio", "n_promociones"]
    commercial += [f"precio_relativo_{brand}" for brand in range(1, 6)]
    groups = {
        "demografia": PROFILE,
        "comercial": commercial,
        "historial": [
            column for column in predictors if column not in PROFILE + commercial
        ],
    }
    ablations = []
    for group_name, removed_columns in groups.items():
        reduced = predictors.drop(columns=removed_columns)
        results, _ = cross_validate(
            winner, parameters[winner], events, reduced, config["folds"], config
        )
        ablations.append(
            {
                "removed": group_name,
                "ap": results.ap.mean(),
                "brier": results.brier.mean(),
            }
        )
    pd.DataFrame(ablations).to_csv(output_dir / "tables/ablations.csv", index=False)

    nested_results = []
    for model_name in ranking.index[:2]:
        for training_end, validation_end in config["folds"][1:]:
            inner_folds = [
                [training_end - 144, training_end - 72],
                [training_end - 72, training_end],
            ]
            phase = f"nested_{training_end}"
            selected = optimize_family(
                model_name,
                events,
                predictors,
                inner_folds,
                config,
                output_dir,
                version,
                phase,
                budget=5,
            )
            results, _ = cross_validate(
                model_name,
                selected,
                events,
                predictors,
                [[training_end, validation_end]],
                config,
            )
            nested_results.append({"model": model_name, **results.iloc[0].to_dict()})
    pd.DataFrame(nested_results).to_csv(output_dir / "tables/nested.csv", index=False)


def persist_finalists(
    ranking, selected_parameters, events, predictors, out_of_fold, config, output_dir
):
    """Conserva OOF y modelo base de dos finalistas, sin modelos por trial/fold."""
    destination = output_dir / "finalists"
    destination.mkdir(exist_ok=True)
    development_mask = events.dia_visita.le(config["development_end"])
    for model_name in ranking.index[:2]:
        out_of_fold[model_name].to_csv(
            destination / f"{model_name}_oof.csv", index=False
        )
        candidate = make_model(
            model_name,
            selected_parameters[model_name],
            list(predictors),
            config["seed"],
            config["threads"],
        )
        candidate.fit(
            predictors[development_mask],
            events.loc[development_mask, "incidencia_compra"],
        )
        joblib.dump(candidate, destination / f"{model_name}_base.joblib")


def run_experiments(root: Path):
    """Punto de entrada: datos → búsqueda → selección → evaluación → diagnósticos."""
    from threadpoolctl import threadpool_limits

    root = Path(root).resolve()
    config = yaml.safe_load((root / "config.yml").read_text())
    output_dir = root / "outputs"
    for folder in ["tables", "figures", "tracking"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    raw = load_data(root / "Dataset/compras_data.csv")
    events, predictors = features(raw)
    sources = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "src/ferreyros_propension").glob("*.py"))
    )
    version = hashlib.sha256(
        (sources + (root / "config.yml").read_text()).encode()
    ).hexdigest()[:12]
    data_hash = hashlib.sha256(
        (root / "Dataset/compras_data.csv").read_bytes()
    ).hexdigest()
    manifest = {
        "version": version,
        "data_sha256": data_hash,
        "config": config,
        "features": list(predictors),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    mlflow.set_tracking_uri(
        "sqlite:///" + (output_dir / "tracking/mlflow.db").as_posix()
    )
    experiment = mlflow.get_experiment_by_name("propension_cpu")
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            "propension_cpu",
            artifact_location=(output_dir / "tracking/artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selected_parameters = {}
    fold_tables = []
    out_of_fold = {}
    with threadpool_limits(limits=config["threads"]):
        for model_name in config["trials"]:
            selected_parameters[model_name] = optimize_family(
                model_name,
                events,
                predictors,
                config["folds"],
                config,
                output_dir,
                version,
            )
            results, predictions = cross_validate(
                model_name,
                selected_parameters[model_name],
                events,
                predictors,
                config["folds"],
                config,
            )
            results["model"] = model_name
            fold_tables.append(results)
            out_of_fold[model_name] = predictions
            print(f"{model_name}: AP={results.ap.mean():.4f}", flush=True)
            # Checkpoint de lectura: conserva resultados de familias terminadas.
            pd.concat(fold_tables).to_csv(output_dir / "tables/cv.csv", index=False)

        cross_validation = pd.concat(fold_tables)
        ranking = (
            cross_validation.groupby("model")
            .agg(
                ap_mean=("ap", "mean"),
                ap_std=("ap", "std"),
                brier_mean=("brier", "mean"),
            )
            .sort_values("ap_mean", ascending=False)
        )
        ranking.to_csv(output_dir / "tables/leaderboard.csv")
        persist_finalists(
            ranking,
            selected_parameters,
            events,
            predictors,
            out_of_fold,
            config,
            output_dir,
        )
        winner = ranking.index[0]
        out_of_fold[winner].to_csv(output_dir / "tables/oof.csv", index=False)
        (output_dir / "best_params.json").write_text(
            json.dumps(selected_parameters, indent=2)
        )
        bundle = evaluate_final(
            winner,
            selected_parameters[winner],
            events,
            predictors,
            out_of_fold[winner],
            config,
            output_dir,
            version,
        )
        diagnose_finalists(
            ranking,
            selected_parameters,
            events,
            predictors,
            config,
            output_dir,
            version,
        )
        mlflow.search_runs(
            [experiment_id], filter_string=f"tags.version = '{version}'"
        ).to_csv(output_dir / "tables/experiments.csv", index=False)
    print("Experimentos terminados. Modelo seleccionado:", winner, flush=True)
    return bundle


if __name__ == "__main__":
    run_experiments(Path.cwd())
