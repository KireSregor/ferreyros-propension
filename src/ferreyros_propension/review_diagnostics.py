"""Development-only explanations and controlled feature-group ablations."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .data import PROFILE
from .review import (
    classification_metrics,
    development_data,
    feature_sets,
    fingerprint,
    fit_classic,
    inner_predictions,
    optimal_threshold,
    predictions,
    read_config,
)

CLASSICAL_FAMILIES = ("logistic", "lightgbm", "xgboost", "catboost")
METRICS = ("precision", "recall", "f1", "auc")


def _selected_results(config):
    """Select each fold's family by internal evidence, never outer labels."""
    output = Path(config["output"])
    selections = []
    for cut, end in config["outer_folds"]:
        candidates = []
        for family in CLASSICAL_FAMILIES:
            path = output / f"{family}_{cut}" / "result.json"
            if not path.exists():
                raise RuntimeError(f"Wait for classical study {family}_{cut} to finish")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["cut"] != cut or result["end"] != end or end > 620:
                raise ValueError("Unexpected development fold in result")
            candidates.append((result, path.parent))
        chosen = max(candidates, key=lambda item: item[0]["inner_score"])
        selections.append(chosen)
    return selections


def _context(config):
    if config["development_end"] > 620:
        raise ValueError("Diagnostics must exclude the reserved test period")
    data = development_data(config)
    events, features, groups = feature_sets(data)
    identity = fingerprint(data, config)
    selections = _selected_results(config)
    if any(result["fingerprint"] != identity for result, _ in selections):
        raise ValueError("Diagnostic inputs differ from the fitted models' fingerprint")
    return events, features, groups, selections, identity


def _permutation_groups(columns, original_columns):
    new = [column for column in columns if column not in original_columns]
    profile = [column for column in columns if column in PROFILE]
    offers = [
        column
        for column in columns
        if column not in new and ("precio" in column or "promo" in column)
    ]
    assigned = set(new + profile + offers)
    history = [column for column in columns if column not in assigned]
    return {
        "customer_history": history,
        "profile": profile,
        "offers": offers,
        "store_trend_and_new": new,
    }


def _permutation_record(model, frame, target, threshold, columns, baseline, rng):
    permuted = frame.copy()
    order = rng.permutation(len(permuted))
    # Shuffle a group jointly so its within-group relationships are preserved.
    for column in columns:
        permuted[column] = frame[column].iloc[order].to_numpy()
    probability = model.predict_proba(permuted)[:, 1]
    values = classification_metrics(target, probability, threshold)
    return {
        **{f"baseline_{metric}": baseline[metric] for metric in METRICS},
        **{f"permuted_{metric}": values[metric] for metric in METRICS},
        **{f"drop_{metric}": baseline[metric] - values[metric] for metric in METRICS},
    }


def _oracle_curve(result, directory, events):
    """Exterior-label oracle for diagnosis only, never a deployable threshold."""
    frame = pd.read_csv(directory / "validation_predictions.csv")
    keys = ["id", "dia_visita"]
    expected = events.loc[
        events.dia_visita.between(result["cut"] + 1, result["end"]),
        keys + ["incidencia_compra"],
    ]
    checked = expected.merge(
        frame, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if (
        not checked._merge.eq("both").all()
        or not checked.incidencia_compra.eq(checked.y).all()
    ):
        raise ValueError(
            "Oracle diagnostic predictions disagree with development visits"
        )
    precision, recall, thresholds = precision_recall_curve(frame.y, frame.p)
    balanced = np.minimum(precision[:-1], recall[:-1])
    index = int(np.argmax(balanced))
    deployed = classification_metrics(frame.y, frame.p, result["threshold"])
    return {
        "model": result["model"],
        "cut": result["cut"],
        "end": result["end"],
        "n": len(frame),
        "deployed_threshold": result["threshold"],
        "deployed_precision": deployed["precision"],
        "deployed_recall": deployed["recall"],
        "deployed_f1": deployed["f1"],
        "deployed_auc": deployed["auc"],
        "deployed_balance": min(deployed["precision"], deployed["recall"]),
        "oracle_balance": float(balanced[index]),
        "oracle_precision": float(precision[index]),
        "oracle_recall": float(recall[index]),
        "oracle_threshold_diagnostic_only": float(thresholds[index]),
        "oracle_gap": float(
            balanced[index] - min(deployed["precision"], deployed["recall"])
        ),
        "curve_reaches_60_60": bool(
            ((precision[:-1] > 0.6) & (recall[:-1] > 0.6)).any()
        ),
        "use": "EXTERIOR_LABEL_ORACLE_DIAGNOSTIC_ONLY_NEVER_FOR_SELECTION",
    }


def explain_candidates(config, max_seconds=None):
    """Permute only after internal selection; do not feed diagnostics to selection."""
    started = perf_counter()
    events, features, groups, selections, _ = _context(config)
    output = Path(config["output"])
    settings = config.get("diagnostics", {})
    sample_size = min(2000, settings.get("explanation_rows", 2000))
    max_seconds = (
        settings.get("explanation_seconds", 60) if max_seconds is None else max_seconds
    )
    rng = np.random.default_rng(config["seed"])
    records, statuses, calibration_rows, oracle_rows = [], [], [], []
    for result, directory in selections:
        oracle_rows.append(_oracle_curve(result, directory, events))
        bundle = joblib.load(directory / "model.joblib")
        if bundle.get("feature_builder_version") != "review_v1":
            raise ValueError("Unsupported explanation artifact version")
        columns = bundle["columns"]
        if columns != result["columns"]:
            raise ValueError("Artifact and result feature schemas differ")
        indices = events.index[
            events.dia_visita.between(result["cut"] + 1, result["end"])
        ]
        indices = rng.choice(
            indices, size=min(sample_size, len(indices)), replace=False
        )
        frame = features.loc[indices, columns].copy()
        target = events.loc[indices, "incidencia_compra"].to_numpy()
        threshold = result["threshold"]
        model = bundle["model"]
        base_probability = model.predict_proba(frame)[:, 1]
        baseline = classification_metrics(target, base_probability, threshold)
        calibration = pd.DataFrame({"y": target, "p": base_probability})
        calibration["decile"] = np.ceil(
            calibration.p.rank(method="first") / len(calibration) * 10
        ).astype(int)
        for decile, part in calibration.groupby("decile"):
            calibration_rows.append(
                {
                    "model": result["model"],
                    "cut": result["cut"],
                    "end": result["end"],
                    "decile": int(decile),
                    "n": len(part),
                    "p_min": part.p.min(),
                    "p_max": part.p.max(),
                    "mean_probability": part.p.mean(),
                    "purchase_rate": part.y.mean(),
                    "purchases": int(part.y.sum()),
                    "threshold": threshold,
                }
            )
        tasks = []
        for name, selected in _permutation_groups(columns, set(groups["all"])).items():
            if selected:
                tasks.extend(("group", name, selected, repeat) for repeat in range(3))
        tasks.extend(("individual", column, [column], 0) for column in columns)
        complete = True
        for kind, name, selected, repeat in tasks:
            if perf_counter() - started >= max_seconds:
                complete = False
                break
            values = _permutation_record(
                model, frame, target, threshold, selected, baseline, rng
            )
            records.append(
                {
                    "model": result["model"],
                    "cut": result["cut"],
                    "end": result["end"],
                    "kind": kind,
                    "feature_group": name,
                    "repeat": repeat,
                    "n": len(frame),
                    "threshold": threshold,
                    "columns": "|".join(selected),
                    **values,
                }
            )
        statuses.append({"cut": result["cut"], "complete": complete})
    pd.DataFrame(records).to_csv(output / "explanation_importance.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output / "calibration_diagnostic.csv", index=False
    )
    pd.DataFrame(oracle_rows).to_csv(
        output / "oracle_curve_diagnostic.csv", index=False
    )
    (output / "explanation_status.json").write_text(
        json.dumps(
            {
                "folds": statuses,
                "seconds": perf_counter() - started,
                "max_seconds": max_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(config)
    return records


def _ablation_result(
    config, events, features, groups, source, source_dir, identity, group
):
    cut, end = source["cut"], source["end"]
    output = Path(config["output"]) / f"ablation_{group}_{cut}"
    output.mkdir(exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result["fingerprint"] != identity
            or result["parameters"] != source["parameters"]
        ):
            raise ValueError("Existing ablation has incompatible configuration")
        return result
    started = perf_counter()
    columns = groups[group]
    if columns == source["columns"]:
        result = {**source, "metrics": [row.copy() for row in source["metrics"]]}
        for filename in [
            "inner_predictions.csv",
            "train_predictions.csv",
            "validation_predictions.csv",
            "model.joblib",
        ]:
            shutil.copy2(source_dir / filename, output / filename)
        result["reused_selected_group"] = True
    else:
        family, params = source["model"], source["parameters"]
        inner, _ = inner_predictions(
            events, features[columns], family, params, cut, config
        )
        threshold, score, _ = optimal_threshold(inner)
        pd.concat(inner).to_csv(output / "inner_predictions.csv", index=False)
        model, iterations = fit_classic(
            events, features[columns], family, params, cut, config
        )
        rows = []
        for split, start, stop in [("train", 1, cut), ("validation", cut + 1, end)]:
            frame = predictions(events, model, features[columns], start, stop, 0)
            frame.to_csv(output / f"{split}_predictions.csv", index=False)
            rows.append(
                {
                    "model": f"ablation_{group}",
                    "cut": cut,
                    "end": end,
                    "split": split,
                    "threshold": threshold,
                    "n": len(frame),
                    **classification_metrics(frame.y, frame.p, threshold),
                }
            )
        joblib.dump(
            {
                "model": model,
                "feature_builder_version": "review_v1",
                "columns": columns,
                "threshold": threshold,
                "training_end": cut,
            },
            output / "model.joblib",
        )
        result = {
            "threshold": threshold,
            "inner_score": score,
            "iterations": iterations,
            "metrics": rows,
        }
    result.update(
        model=f"ablation_{group}",
        cut=cut,
        end=end,
        fingerprint=identity,
        parameters=source["parameters"],
        feature_group=group,
        columns=columns,
        trials=0,
        selected_family=source["model"],
        selection_basis="inner_score_only",
        seconds=perf_counter() - started,
    )
    for row in result["metrics"]:
        row["model"] = f"ablation_{group}"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def ablate_features(config, max_seconds=None):
    """Hold selected family/parameters fixed; recalibrate each group's threshold internally."""
    started = perf_counter()
    events, features, groups, selections, identity = _context(config)
    seconds = (
        config.get("diagnostics", {}).get("ablation_seconds", 240)
        if max_seconds is None
        else max_seconds
    )
    results = []
    skipped = []
    # Round-robin groups ensure a limited budget does not only cover one fold.
    for group in ["history", "history_offers", "all", "enhanced"]:
        for source, directory in selections:
            if perf_counter() - started >= seconds:
                skipped.append(
                    {"group": group, "cut": source["cut"], "reason": "time_budget"}
                )
                continue
            result = _ablation_result(
                config, events, features, groups, source, directory, identity, group
            )
            results.append(result)
    output = Path(config["output"])
    pd.DataFrame([row for result in results for row in result["metrics"]]).to_csv(
        output / "ablation_metrics.csv", index=False
    )
    (output / "ablation_status.json").write_text(
        json.dumps(
            {
                "seconds": perf_counter() - started,
                "max_seconds": seconds,
                "completed": len(results),
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(config)
    return results


def _write_report(config):
    output = Path(config["output"])
    lines = [
        "# Diagnóstico de variables y ablaciones controladas",
        "",
        "Solo desarrollo hasta el día 620; no se consultan etiquetas ni artefactos de test. "
        "En cada fold se elige la familia clásica por su puntuación interna, nunca por su validación exterior.",
        "",
        "## Importancia por permutación",
        "",
        "La muestra exterior contiene como máximo 2.000 visitas por fold. Se conserva el umbral seleccionado "
        "internamente. Las caídas positivas indican empeoramiento al permutar; se reportan precisión, recall, "
        "F1 y AUC. Los grupos se permutan conjuntamente, con tres repeticiones; cada variable individual "
        "tiene una repetición exploratoria. Este diagnóstico usa etiquetas exteriores y no se utiliza "
        "para seleccionar modelos ni parámetros.",
        "",
        "La permutación puede crear combinaciones poco plausibles y la correlación entre variables puede "
        "ocultar importancia. No mide causalidad ni importancia universal. Los resultados individuales "
        "son orientativos y no sustituyen una ablación temporal.",
        "",
    ]
    importance_path = output / "explanation_importance.csv"
    if importance_path.exists() and importance_path.stat().st_size > 1:
        importance = pd.read_csv(importance_path)
        if not importance.empty:
            summary = (
                importance.loc[importance.kind.eq("group")]
                .groupby(["cut", "model", "feature_group"])
                .drop_auc.mean()
                .reset_index()
            )
            lines.extend(
                f"- Fold con corte {r.cut}, {r.model}: {r.feature_group}, "
                f"caída media de AUC {r.drop_auc:+.4f}."
                for r in summary.itertuples()
            )
    lines.extend(
        [
            "",
            "## Ablaciones",
            "",
            "Se fija la familia y los hiperparámetros ganadores internos de cada fold; únicamente cambia "
            "el conjunto de variables. El umbral se recalibra con los mismos folds internos y la cantidad "
            "de árboles se vuelve a seleccionar mediante la misma parada interna. Por ello se comparan "
            "procedimientos completos y no un efecto aislado del umbral. El presupuesto se comprueba "
            "entre tareas; una tarea iniciada puede terminar después del límite.",
            "",
        ]
    )
    lines.extend(
        [
            "## Cota de la curva precisión–recall: oráculo diagnóstico",
            "",
            "`oracle_curve_diagnostic.csv` recorre todos los umbrales sobre las predicciones exteriores "
            "del candidato elegido solo mediante puntuación interna. Compara su umbral desplegado con "
            "el máximo de min(precisión, recall) alcanzable en esa curva específica. El oráculo usa "
            "las respuestas exteriores y es optimista: NO es una mejora lograda por el modelo, "
            "NO se usa para escoger modelos ni cambiar umbrales y NO es una cota de todos los modelos posibles. "
            "Una brecha orienta sobre transferencia del umbral; una curva limitada sugiere investigar el ranking. "
            "Ninguno de estos resultados identifica por sí solo una causa.",
            "",
        ]
    )
    oracle_path = output / "oracle_curve_diagnostic.csv"
    if oracle_path.exists() and oracle_path.stat().st_size > 1:
        for row in pd.read_csv(oracle_path).itertuples():
            lines.append(
                f"- Corte {row.cut}, {row.model}: equilibrio desplegado "
                f"{row.deployed_balance:.1%}; oráculo diagnóstico {row.oracle_balance:.1%}; "
                f"brecha {row.oracle_gap:.1%}. No es una métrica de despliegue."
            )
    lines.extend(
        [
            "## Diagnóstico de calibración",
            "",
            "`calibration_diagnostic.csv` agrupa la misma muestra exterior por deciles de probabilidad "
            "predicha y contrasta probabilidad media con compras observadas, junto a su soporte. "
            "Los empates de score se reparten por orden para mantener grupos de tamaño comparable. "
            "No se ajusta ningún calibrador con las etiquetas exteriores. Las visitas de un cliente "
            "no son independientes: no deben aplicarse intervalos binomiales ingenuos ni atribuir "
            "causalidad a las diferencias observadas.",
            "",
        ]
    )
    path = output / "ablation_metrics.csv"
    if path.exists() and path.stat().st_size > 1:
        ablation = pd.read_csv(path)
        for r in ablation.loc[ablation.split.eq("validation")].itertuples():
            lines.append(
                f"- Corte {r.cut}, {r.model}: precisión {r.precision:.1%}, recall "
                f"{r.recall:.1%}, F1 {r.f1:.1%}, AUC {r.auc:.3f}."
            )
    lines.extend(
        [
            "",
            "Los estados JSON indican coberturas incompletas por tiempo. Estos períodos ya se "
            "habían revisado en el proyecto: las conclusiones siguen siendo exploratorias. "
            "Ninguna caída o mejora identifica por sí sola la causa de un error.",
        ]
    )
    (output / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def neural_seed_sensitivity(config, max_seconds=300):
    """Diagnostic seed repetitions; preserve selected parameters and original threshold."""
    from .review_neural import _fit_window, _frame, _identity

    started = perf_counter()
    if config["development_end"] > 620:
        raise ValueError("Seed diagnostics must exclude the reserved test")
    data = development_data(config)
    events, features, _ = feature_sets(data)
    identity = _identity(data, config)
    output = Path(config["output"])
    rows, statuses = [], []
    for cut, end in config["outer_folds"]:
        path = output / f"neural_{cut}" / "result.json"
        if not path.exists():
            raise RuntimeError(f"Wait for neural_{cut} to complete")
        source = json.loads(path.read_text(encoding="utf-8"))
        if (
            source["fingerprint"] != identity
            or source["cut"] != cut
            or source["end"] != end
        ):
            raise ValueError("Seed diagnostic source does not match development/config")
        rows.extend(
            {**row, "seed": config["seed"], "diagnostic_only": True}
            for row in source["metrics"]
        )
        directory = output / f"neural_seed_diagnostics_{cut}"
        directory.mkdir(exist_ok=True)
        for seed in [137, 915]:
            cache = directory / f"seed_{seed}_metrics.json"
            digest = hashlib.sha256(
                path.read_bytes() + Path(__file__).read_bytes() + str(seed).encode()
            ).hexdigest()
            if cache.exists():
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["fingerprint"] != digest:
                    raise ValueError("Incompatible cached seed provenance")
                rows.extend(saved["metrics"])
                statuses.append({"cut": cut, "seed": seed, "status": "cached"})
                continue
            if perf_counter() - started >= max_seconds:
                statuses.append({"cut": cut, "seed": seed, "status": "skipped_budget"})
                continue
            seed_config = {**config, "seed": seed}
            records = []
            fitted = _fit_window(
                events,
                features[source["columns"]],
                source["parameters"],
                cut,
                end,
                seed_config,
                directory,
                f"seed_{seed}",
                records,
                artifact=True,
            )
            metrics = []
            for split, mask in [
                ("train", events.dia_visita.le(cut)),
                ("validation", events.dia_visita.between(cut + 1, end)),
            ]:
                frame = _frame(events, mask, fitted["probabilities"][split], 0)
                frame.to_csv(
                    directory / f"seed_{seed}_{split}_predictions.csv", index=False
                )
                template = next(
                    row for row in source["metrics"] if row["split"] == split
                )
                metric = classification_metrics(frame.y, frame.p, source["threshold"])
                metrics.append(
                    {**template, **metric, "seed": seed, "diagnostic_only": True}
                )
            payload = dict(fingerprint=digest, metrics=metrics)
            cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.extend(metrics)
            statuses.append({"cut": cut, "seed": seed, "status": "completed"})
        pd.DataFrame(rows).to_csv(output / "neural_seed_sensitivity.csv", index=False)
        status = dict(
            seconds=perf_counter() - started,
            max_seconds=max_seconds,
            coverage=statuses,
            base_seed=config["seed"],
            selection="None; fixed original threshold and parameters",
        )
        (output / "neural_seed_sensitivity_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
    return rows


def explain_leader(config, max_seconds=120):
    """Describe the final learned leader, including ensemble/calibration wrappers."""
    started = perf_counter()
    if config["development_end"] > 620:
        raise ValueError("Leader explanation must exclude the reserved test")
    output = Path(config["output"])
    leaderboard = pd.read_csv(output / "leaderboard.csv")
    complete = leaderboard.loc[leaderboard.complete.eq(True)]
    sources = {}
    for path in output.rglob("result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["end"] > 620:
            raise ValueError("A reported result extends beyond development")
        sources[(result["model"], result["cut"])] = (result, path.parent)
    learned = None
    for candidate in complete.model:
        if candidate in {"historical", "constant"}:
            continue
        available = [sources.get((candidate, cut)) for cut, _ in config["outer_folds"]]
        if all(item and (item[1] / "model.joblib").exists() for item in available):
            learned = candidate
            break
    if learned is None:
        raise RuntimeError("No complete learned procedure has artifacts for all folds")
    events, features, groups = feature_sets(development_data(config))
    rng = np.random.default_rng(config["seed"])
    records, incomplete = [], []
    for cut, end in config["outer_folds"]:
        result, directory = sources[(learned, cut)]
        bundle = joblib.load(directory / "model.joblib")
        if bundle.get("feature_builder_version") != "review_v1":
            raise ValueError("Unsupported learned-leader feature builder")
        indices = events.index[events.dia_visita.between(cut + 1, end)]
        indices = rng.choice(indices, min(1500, len(indices)), replace=False)
        frame = features.loc[indices, bundle["columns"]].copy()
        target = events.loc[indices, "incidencia_compra"].to_numpy()
        model, threshold = bundle["model"], result["threshold"]
        baseline = classification_metrics(
            target, model.predict_proba(frame)[:, 1], threshold
        )
        for group, columns in _permutation_groups(
            frame.columns, set(groups["all"])
        ).items():
            if not columns:
                continue
            for repeat in range(3):
                if perf_counter() - started >= max_seconds:
                    incomplete.append({"cut": cut, "group": group, "repeat": repeat})
                    break
                values = _permutation_record(
                    model, frame, target, threshold, columns, baseline, rng
                )
                records.append(
                    dict(
                        model=learned,
                        cut=cut,
                        end=end,
                        group=group,
                        repeat=repeat,
                        n=len(frame),
                        threshold=threshold,
                        **values,
                    )
                )
    pd.DataFrame(records).to_csv(output / "leader_group_importance.csv", index=False)
    overall = complete.iloc[0].model
    lines = [
        "# Explicación descriptiva del líder aprendido",
        "",
        f"Primer procedimiento completo del ranking: `{overall}`. "
        f"Procedimiento aprendido explicado con artefactos de los tres folds: `{learned}`.",
        "",
        "La elección del líder es retrospectiva a la tabla de desarrollo; esta explicación no "
        "constituye validación independiente. Si gana una referencia no aprendida, se explica "
        "el mejor candidato aprendido, sin presentarlo como ganador general.",
        "",
        "Se permutan conjuntamente historial, perfil, ofertas y variables nuevas cuando están "
        "presentes, con tres repeticiones sobre un máximo de 1.500 visitas por fold y el umbral "
        "original fijo. Cada caída positiva indica pérdida de la métrica respecto al mismo "
        "modelo sin permutación. Los cambios son descriptivos, no causales; variables correlacionadas "
        "y combinaciones artificiales pueden alterar su interpretación. No se usa este análisis "
        "para ajustar parámetros ni umbrales. No se consultó test.",
        "",
    ]
    if records:
        summary = (
            pd.DataFrame(records)
            .groupby(["cut", "group"])[[f"drop_{metric}" for metric in METRICS]]
            .mean()
        )
        lines.extend(
            f"- Corte {cut}, {group}: caída media de precisión {r.drop_precision:+.1%}, "
            f"recall {r.drop_recall:+.1%}, F1 {r.drop_f1:+.1%}, AUC {r.drop_auc:+.4f}."
            for (cut, group), r in summary.iterrows()
        )
    lines.extend(
        [
            "",
            f"Tiempo: {perf_counter() - started:.1f} segundos; "
            f"tareas incompletas por presupuesto: {len(incomplete)}.",
        ]
    )
    (output / "LEADER_EXPLANATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["explain", "ablate", "both"], default="both")
    arguments = parser.parse_args()
    configuration = read_config()
    if arguments.mode in {"explain", "both"}:
        explain_candidates(configuration)
    if arguments.mode in {"ablate", "both"}:
        ablate_features(configuration)
