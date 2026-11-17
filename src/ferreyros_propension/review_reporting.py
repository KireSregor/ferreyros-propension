"""Aggregate reviewed development results without consulting the reserved test."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .data import PROMOS
from .review import development_data, feature_sets

METRICS = ["precision", "recall", "f1", "auc", "ap", "logloss"]
KEYS = ["id", "dia_visita"]


def _load_results(output, end_day):
    rows, sources, times = [], {}, []
    for path in sorted(output.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result["end"]) > end_day:
            raise ValueError(f"Result extends beyond development: {path.parent.name}")
        key = (result["model"], int(result["cut"]))
        if key in sources:
            raise ValueError(f"Duplicate model/fold result: {key}")
        sources[key] = path.parent
        rows.extend(result["metrics"])
        times.append(
            {
                "model": key[0],
                "cut": key[1],
                "seconds": result.get("seconds"),
                "trials": result.get("trials", 0),
                "iterations": result.get("iterations"),
            }
        )
    if not rows:
        raise ValueError("No completed reviewed results found")
    metrics = pd.DataFrame(rows)
    if metrics.duplicated(["model", "cut", "split"]).any():
        raise ValueError("Duplicate metrics for model/fold/split")
    metrics["balance"] = metrics[["precision", "recall"]].min(axis=1)
    metrics["meets_60_60"] = metrics.precision.gt(0.6) & metrics.recall.gt(0.6)
    return metrics, sources, pd.DataFrame(times)


def _predictions(directory, split, expected_y):
    frame = pd.read_csv(directory / f"{split}_predictions.csv")
    required = KEYS + ["y", "p"]
    if not set(required).issubset(frame) or frame.duplicated(KEYS).any():
        raise ValueError(f"Invalid prediction keys/schema in {directory.name}")
    if not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all():
        raise ValueError(f"Invalid probabilities in {directory.name}")
    aligned = frame.merge(
        expected_y,
        on=KEYS,
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if (
        not aligned._merge.eq("both").all()
        or not aligned.y.eq(aligned.y_expected).all()
    ):
        raise ValueError(f"Prediction rows/labels disagree in {directory.name}")
    return aligned[required].sort_values(KEYS).reset_index(drop=True)


def _weighted_scores(y, p, threshold, weights):
    positive = p >= threshold
    tp = weights[(y == 1) & positive].sum()
    fp = weights[(y == 0) & positive].sum()
    fn = weights[(y == 1) & ~positive].sum()
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    auc = roc_auc_score(y, p, sample_weight=weights)
    return np.array([precision, recall, f1, auc])


def _calibration_and_transfer(frame, threshold):
    """Post-hoc outer-label diagnostics; never return an oracle threshold."""
    frame = frame.copy()
    frame["bin"] = np.minimum((frame.p * 10).astype(int), 9)
    calibration = []
    for bin_id in range(10):
        group = frame.loc[frame.bin.eq(bin_id)]
        predicted_mean = float(group.p.mean()) if len(group) else np.nan
        observed_rate = float(group.y.mean()) if len(group) else np.nan
        calibration.append(
            {
                "bin_low": bin_id / 10,
                "bin_high": (bin_id + 1) / 10,
                "n": len(group),
                "customers": group.id.nunique(),
                "purchases": int(group.y.sum()),
                "mean_predicted": predicted_mean,
                "observed_rate": observed_rate,
                "observed_minus_predicted": observed_rate - predicted_mean,
            }
        )
    predicted = frame.p.ge(threshold)
    true_positive = int((frame.y.eq(1) & predicted).sum())
    precision = true_positive / predicted.sum() if predicted.sum() else 0
    recall = true_positive / frame.y.sum() if frame.y.sum() else 0
    deployed = float(min(precision, recall))
    curve_precision, curve_recall, _ = precision_recall_curve(frame.y, frame.p)
    oracle = float(np.minimum(curve_precision[:-1], curve_recall[:-1]).max())
    transfer = {
        "n": len(frame),
        "deployed_balance": deployed,
        "oracle_balance_fixed_predictions": oracle,
        "oracle_minus_deployed": max(0.0, oracle - deployed),
        "fixed_curve_can_exceed_60_60": oracle > 0.6,
        "posthoc_outer_labels": True,
        "selection_allowed": False,
    }
    return calibration, transfer


def _paired_bootstrap(
    frame, baseline, threshold, baseline_threshold, seed=42, reps=200
):
    if not np.array_equal(
        frame[KEYS + ["y"]].to_numpy(), baseline[KEYS + ["y"]].to_numpy()
    ):
        raise ValueError("Paired bootstrap requires identical sorted visits and labels")
    y, p, baseline_p = frame.y.to_numpy(), frame.p.to_numpy(), baseline.p.to_numpy()
    _, customer = np.unique(frame.id, return_inverse=True)
    count = customer.max() + 1
    rng = np.random.default_rng(seed)
    original = np.ones(len(y))
    delta = _weighted_scores(y, p, threshold, original) - _weighted_scores(
        y, baseline_p, baseline_threshold, original
    )
    samples = []
    for _ in range(reps):
        multiplicity = np.bincount(rng.integers(0, count, count), minlength=count)
        weights = multiplicity[customer]
        if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
            continue
        samples.append(
            _weighted_scores(y, p, threshold, weights)
            - _weighted_scores(y, baseline_p, baseline_threshold, weights)
        )
    if not samples:
        raise ValueError("No bootstrap samples contain both classes")
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return [
        {
            "metric": metric,
            "delta": value,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_reps": len(samples),
        }
        for metric, value, low, high in zip(
            METRICS[:4], delta, lower, upper, strict=True
        )
    ]


def _segments(events, features):
    frame = events[KEYS].copy()
    recency = features.dias_desde_compra
    frame["recencia"] = (
        pd.cut(
            recency,
            [-np.inf, 7, 30, 60, np.inf],
            labels=["hasta7", "8a30", "31a60", "mas60"],
        )
        .astype("string")
        .fillna("sin_compra_previa")
    )
    frame["frecuencia"] = pd.cut(
        events.visitas_previas,
        [-1, 5, 15, 30, np.inf],
        labels=["0a5", "6a15", "16a30", "mas30"],
    ).astype("string")
    frame["edad"] = pd.cut(events.edad, [-np.inf, 29, 39, 49, 59, np.inf]).astype(
        "string"
    )
    frame["promocion"] = events[PROMOS].max(axis=1).map({0: "sin", 1: "con"})
    frame["genero"] = events.genero.astype("string")
    frame["bloque30"] = ((events.dia_visita - 1) // 30 + 1).astype("string")
    return frame


def _error_segments(frame, segments, threshold, model, cut):
    joined = frame.merge(segments, on=KEYS, how="left", validate="one_to_one")
    predicted = joined.p.ge(threshold)
    joined["tp"] = joined.y.eq(1) & predicted
    joined["fp"] = joined.y.eq(0) & predicted
    joined["fn"] = joined.y.eq(1) & ~predicted
    joined["tn"] = joined.y.eq(0) & ~predicted
    records = []
    for column in segments.columns.difference(KEYS):
        for label, group in joined.groupby(column, dropna=False, observed=True):
            tp, fp, fn, tn = (int(group[c].sum()) for c in ["tp", "fp", "fn", "tn"])
            records.append(
                {
                    "model": model,
                    "cut": cut,
                    "dimension": column,
                    "segment": str(label),
                    "n": len(group),
                    "customers": group.id.nunique(),
                    "purchase_rate": group.y.mean(),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": tp / (tp + fp) if tp + fp else 0,
                    "recall": tp / (tp + fn) if tp + fn else 0,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                    "small_support": len(group) < 100 or group.id.nunique() < 20,
                }
            )
    return records


def _plot(metrics, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    validation = metrics.loc[metrics.split.eq("validation")]
    ranking = validation.groupby("model").agg(
        folds=("cut", "nunique"),
        balance=("balance", "mean"),
        f1=("f1", "mean"),
        logloss=("logloss", "mean"),
    )
    ranking = ranking.loc[ranking.folds.eq(validation.cut.nunique())].sort_values(
        ["balance", "f1", "logloss"], ascending=[False, False, True]
    )
    leaders = [
        name for name in ranking.index if name not in {"historical", "reference"}
    ][:5]
    order = ["historical", "reference", *leaders]
    for cut, frame in metrics.groupby("cut"):
        present = [name for name in order if name in set(frame.model)]
        fig, axes = plt.subplots(2, 2, figsize=(max(12, len(present) * 0.8), 8))
        for metric, axis in zip(METRICS[:4], axes.flat, strict=True):
            for split, color, marker in [
                ("train", "#8593A0", "o"),
                ("validation", "#2251FF", "D"),
            ]:
                values = (
                    frame.loc[frame.split.eq(split)]
                    .set_index("model")[metric]
                    .reindex(present)
                )
                axis.plot(
                    range(len(present)),
                    values,
                    marker=marker,
                    color=color,
                    label="Train" if split == "train" else "Validación",
                )
            axis.set_xticks(range(len(present)), present, rotation=45, ha="right")
            axis.set_title(
                metric.upper() if metric == "auc" else metric.capitalize(),
                loc="left",
                color="#051C2C",
            )
            axis.grid(axis="y", color="#E8EDF1")
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False)
        end = int(frame.end.max())
        fig.suptitle(
            f"Train 1–{int(cut)} · Validación {int(cut) + 1}–{end}", color="#051C2C"
        )
        fig.tight_layout()
        fig.savefig(figures / f"fold_{int(cut)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _markdown_table(frame, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
            for c in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plot_neural_curves(config, output):
    """Plot internal stopping losses, never exterior-validation losses."""
    families = [
        family
        for family in ["neural", "neural_wide"]
        if any(
            (output / f"{family}_{cut}" / "outer_selection_curves.csv").exists()
            for cut, _ in config["outer_folds"]
        )
    ]
    if not families:
        return
    folds = config["outer_folds"]
    fig, axes = plt.subplots(
        len(families),
        len(folds),
        squeeze=False,
        figsize=(5 * len(folds), 4.2 * len(families)),
    )
    stop_days = config["early_stopping"]["days"]
    for row, family in enumerate(families):
        for column, (cut, _) in enumerate(folds):
            axis = axes[row, column]
            directory = output / f"{family}_{cut}"
            curves_path = directory / "outer_selection_curves.csv"
            fit_path = directory / "outer_selection_fit.json"
            if not curves_path.exists() or not fit_path.exists():
                axis.text(0.5, 0.5, "Curva no disponible", ha="center", va="center")
                axis.set_axis_off()
                continue
            curve = pd.read_csv(curves_path)
            details = json.loads(fit_path.read_text(encoding="utf-8"))
            axis.plot(
                curve.epoch, curve.train_loss, color="#8593A0", label="Entrenamiento"
            )
            axis.plot(
                curve.epoch,
                curve.validation_loss,
                color="#2251FF",
                label="Ventana interna de parada",
            )
            selected = details["best_epochs"]
            axis.axvline(
                selected,
                linestyle="--",
                color="#008C95",
                linewidth=1.2,
                label=f"Época elegida: {selected}",
            )
            architecture = details.get("architecture", family)
            axis.set_title(
                f"{family} · {architecture}\nTrain 1–{cut - stop_days}; parada {cut - stop_days + 1}–{cut}",
                fontsize=10,
                loc="left",
                color="#051C2C",
            )
            axis.text(
                0.98,
                0.97,
                f"Elegidas {selected} / ejecutadas {details['executed_epochs']}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            axis.set_xlabel("Época")
            axis.set_ylabel("Log loss · menor es mejor")
            axis.grid(axis="y", color="#E8EDF1")
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Curvas de selección de épocas: la ventana de parada es interna, no el fold exterior",
        color="#051C2C",
        fontsize=12,
    )
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "neural_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parameter_importances(sources, output):
    """Descriptive fANOVA importances of tuning scores, not feature effects."""
    import optuna

    records = {}
    for (family, cut), directory in sources.items():
        if family not in {"logistic", "lightgbm", "xgboost", "catboost"}:
            continue
        database = directory / "study.db"
        if not database.exists():
            continue
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        study = optuna.load_study(
            study_name=f"{family}_{cut}_{result['fingerprint'][:12]}",
            storage=f"sqlite:///{database.resolve().as_posix()}",
        )
        completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if len(completed) < 10:
            continue
        try:
            importance = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
            )
            records[f"{family}_{cut}"] = {
                "completed_trials": len(completed),
                "importance": importance,
            }
        except ValueError as error:
            records[f"{family}_{cut}"] = {
                "completed_trials": len(completed),
                "unavailable_reason": str(error),
            }
    (output / "parameter_importances.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def _plot_selected_diagnostics(candidate, validation, frames, config, output):
    """Describe the highest observed balance; do not optimize on these curves."""
    folds = config["outer_folds"]
    fig, axes = plt.subplots(2, len(folds), squeeze=False, figsize=(5 * len(folds), 8))
    styles = [
        (candidate, "Candidato", "#2251FF"),
        ("historical", "Histórico", "#8593A0"),
        ("reference", "LightGBM de referencia", "#008C95"),
    ]
    for column, (cut, end) in enumerate(folds):
        axis = axes[0, column]
        axis.fill_between(
            [0.6, 1], 0.6, 1, color="#E2F0E9", label="Zona objetivo 60/60"
        )
        plotted = set()
        for model, label, color in styles:
            if model in plotted or (model, cut) not in frames:
                continue
            plotted.add(model)
            frame = frames[(model, cut)]
            precision, recall, _ = precision_recall_curve(frame.y, frame.p)
            axis.plot(recall, precision, label=label, color=color, linewidth=1.5)
        frame = frames[(candidate, cut)]
        row = validation.loc[
            validation.model.eq(candidate) & validation.cut.eq(cut)
        ].iloc[0]
        predicted = frame.p.ge(row.threshold)
        actual = frame.y.eq(1)
        tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
        fn, tn = int((actual & ~predicted).sum()), int((~actual & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        axis.scatter(
            [recall],
            [precision],
            color="#2251FF",
            edgecolors="#051C2C",
            s=55,
            zorder=5,
            label="Decisión con umbral fijado",
        )
        axis.set_title(
            f"Fold {column + 1} · días {cut + 1}–{end}\nPrecisión {precision:.1%} · Recall {recall:.1%}",
            fontsize=11,
            color="#051C2C",
            loc="left",
        )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Recall · compras detectadas / compras reales",
            ylabel="Precisión · aciertos / alertas de compra",
        )
        axis.grid(color="#E8EDF1")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7, loc="lower left")
        matrix = np.array([[tn, fp], [fn, tp]])
        axis = axes[1, column]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for i in range(2):
            for j in range(2):
                label = f"{matrix[i, j]:,}".replace(",", " ")
                axis.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#051C2C",
                )
        axis.set_xticks([0, 1], ["No compra", "Compra"])
        axis.set_yticks([0, 1], ["No compra", "Compra"])
        axis.set_xlabel("Predicción del modelo")
        axis.set_ylabel("Resultado real")
        axis.set_title(
            f"Visitas · n={len(frame):,}".replace(",", " "),
            fontsize=11,
            color="#051C2C",
            loc="left",
        )
    fig.suptitle(
        f"Mayor equilibrio observado: {candidate}\nCurvas diagnósticas; el punto operativo mantiene el umbral interno",
        fontsize=12,
        color="#051C2C",
    )
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(
        figures / "selected_candidate_diagnostics.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _logged_neural_fits(directory):
    """Count archived executions while ignoring byte-equivalent backup copies."""
    records, seen, duplicate_files = [], set(), 0
    for path in sorted(directory.rglob("*_fit.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        signature = hashlib.sha256(
            json.dumps(
                {"filename": path.name, "record": record}, sort_keys=True
            ).encode()
        ).hexdigest()
        if signature in seen:
            duplicate_files += 1
            continue
        seen.add(signature)
        records.append(record)
    return records, duplicate_files


def write_experiment_inventory(config):
    """Count completed work, separating logged neural fits from inferred fits."""
    output = Path(config["output"])
    rows, counted_groups = [], set()
    for path in sorted(output.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["end"] > config["development_end"]:
            raise ValueError("Inventory encountered a result beyond development")
        model, cut = result["model"], int(result["cut"])
        inner_count = len(config["inner_folds"][cut])
        record = {
            "model": model,
            "cut": cut,
            "source": path.parent.name,
            "category": "main",
            "complete_trials": 0,
            "failed_trials": 0,
            "pruned_trials": 0,
            "running_trials": 0,
            "manual_variants": 0,
            "training_fits": 0,
            "base_fits": 0,
            "calibrator_fits": 0,
            "executed_epochs": 0,
            "selected_epochs": 0,
            "peak_gpu_memory_bytes": 0,
            "duplicate_fit_files_ignored": 0,
            "fit_count_basis": "inferred",
        }
        trials_path = path.parent / "trials.csv"
        if trials_path.exists():
            states = pd.read_csv(trials_path, usecols=["state"]).state.value_counts()
            for state, field in [
                ("COMPLETE", "complete_trials"),
                ("FAIL", "failed_trials"),
                ("PRUNED", "pruned_trials"),
                ("RUNNING", "running_trials"),
            ]:
                record[field] = int(states.get(state, 0))
        elif model in {"logistic", "lightgbm", "xgboost", "catboost", "neural"}:
            record["complete_trials"] = int(
                result.get("complete_trials", result.get("trials", 0))
            )
        if model in {"logistic", "lightgbm", "xgboost", "catboost"}:
            multiplier = 1 if model == "logistic" else 2
            record["training_fits"] = multiplier * (
                inner_count * record["complete_trials"] + 1
            )
            record["base_fits"] = record["training_fits"]
        elif model == "reference":
            record["training_fits"] = record["base_fits"] = inner_count + 1
        elif model in {"neural", "neural_wide"}:
            fits, duplicate_files = _logged_neural_fits(path.parent)
            record.update(
                training_fits=len(fits),
                fit_count_basis="logged",
                duplicate_fit_files_ignored=duplicate_files,
                executed_epochs=sum(r["executed_epochs"] for r in fits),
                selected_epochs=sum(r["best_epochs"] for r in fits),
                peak_gpu_memory_bytes=max(
                    (r.get("peak_gpu_memory_bytes", 0) for r in fits), default=0
                ),
            )
            if model == "neural_wide":
                record["manual_variants"] = len(result.get("variants_completed", []))
        elif model.startswith("seedmean_"):
            family = result["underlying_family"]
            extra_seeds = sum(seed != config["seed"] for seed in config["seeds"])
            record["training_fits"] = (
                extra_seeds * (inner_count + 1) * (1 if family == "logistic" else 2)
            )
            record["base_fits"] = record["training_fits"]
        elif model.startswith("chronological_"):
            group = result.get("shared_training_group", f"chronological_{cut}")
            record["shared_training_group"] = group
            if group not in counted_groups:
                record.update(training_fits=22, base_fits=21, calibrator_fits=1)
                counted_groups.add(group)
            else:
                record["fit_count_basis"] = "shared_work_counted_elsewhere"
        elif model.startswith("ablation_"):
            record["category"] = "diagnostic"
            family = result["selected_family"]
            source = output / f"{family}_{cut}" / "result.json"
            selected = json.loads(source.read_text(encoding="utf-8"))
            reused = (
                result.get("reused_selected_group", False)
                or result["feature_group"] == selected["feature_group"]
            )
            if not reused:
                record["training_fits"] = (inner_count + 1) * (
                    1 if family == "logistic" else 2
                )
                record["base_fits"] = record["training_fits"]
            else:
                record["fit_count_basis"] = "reused_source_no_new_fit"
        elif model == "calibrated_reference":
            record.update(
                base_fits=int(result["base_fits"]),
                calibrator_fits=int(result["calibrator_fits"]),
                training_fits=int(result["total_fits"]),
            )
        rows.append(record)
    for directory in sorted(output.glob("neural_seed_diagnostics_*")):
        fits, duplicate_files = _logged_neural_fits(directory)
        rows.append(
            {
                "model": "neural_seed_diagnostics",
                "cut": int(directory.name.rsplit("_", 1)[1]),
                "source": directory.name,
                "category": "diagnostic",
                "complete_trials": 0,
                "training_fits": len(fits),
                "fit_count_basis": "logged",
                "duplicate_fit_files_ignored": duplicate_files,
                "executed_epochs": sum(r["executed_epochs"] for r in fits),
                "selected_epochs": sum(r["best_epochs"] for r in fits),
                "peak_gpu_memory_bytes": max(
                    (r.get("peak_gpu_memory_bytes", 0) for r in fits), default=0
                ),
            }
        )
    pilot_path = output / "gpu_pilot_selection.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": "catboost_gpu_pilot",
                "cut": pilot["cut"],
                "source": pilot_path.name,
                "category": "pilot",
                "complete_trials": pilot.get("complete_trials", 1),
                "training_fits": pilot.get("actual_fits", 8),
                "fit_count_basis": "logged_result",
                "executed_rounds": pilot.get("executed_rounds"),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "experiment_inventory.csv", index=False)
    main = inventory.loc[inventory.category.eq("main")]
    summary = {
        "main_complete_optuna_trials": int(main.complete_trials.sum()),
        "main_manual_variants": int(main.manual_variants.fillna(0).sum()),
        "main_training_fits": int(main.training_fits.sum()),
        "diagnostic_training_fits": int(
            inventory.loc[inventory.category.eq("diagnostic"), "training_fits"].sum()
        ),
        "pilot_training_fits": int(
            inventory.loc[inventory.category.eq("pilot"), "training_fits"].sum()
        ),
        "total_training_fits_accounted": int(inventory.training_fits.sum()),
        "neural_executed_epochs_logged": int(inventory.executed_epochs.fillna(0).sum()),
        "neural_selected_epochs_logged": int(inventory.selected_epochs.fillna(0).sum()),
        "peak_gpu_memory_bytes_logged": int(
            inventory.peak_gpu_memory_bytes.fillna(0).max()
        ),
        "failed_trials_captured": int(inventory.failed_trials.fillna(0).sum()),
        "pruned_trials_captured": int(inventory.pruned_trials.fillna(0).sum()),
        "running_trials_captured": int(inventory.running_trials.fillna(0).sum()),
        "duplicate_neural_fit_files_ignored": int(
            inventory.duplicate_fit_files_ignored.fillna(0).sum()
        ),
        "notes": [
            "Classical fit counts inferred from completed trials plus final fit; failed/active partial fits may be unobservable.",
            "Neural fits counted from *_fit.json including initial_selection and resumed_selection archives and seed diagnostics; identical backup copies of the same named fit metadata are counted once.",
            "Trial states come from the current cumulative trials.csv only; archived selection.json files are not additional trials or model results.",
            "Chronological shared training counted once: 21 base fits plus one meta-logistic fit per outer cut.",
            "GPU pilot is separate from main study trials; synthetic tests and small-sample overfit diagnostics excluded.",
            "Weighted blends, thresholds and store corrections add no model training fits.",
            "Counts summarize completed recorded work, not a claim of exhaustive search. Task times must not be summed as wall time.",
        ],
    }
    (output / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return inventory


def write_report(config):
    """Write reviewed metrics, paired uncertainty, error segments and figures."""
    output = Path(config["output"])
    metrics, sources, times = _load_results(output, config["development_end"])
    runtime = {}
    for path in sorted(output.glob("*runtime*.json")):
        if path.name != "runtime_summary.json":
            runtime[path.name] = json.loads(path.read_text(encoding="utf-8"))
    (output / "runtime_summary.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    validation = metrics.loc[metrics.split.eq("validation")]
    leaderboard = (
        validation.groupby("model")
        .agg(
            folds=("cut", "nunique"),
            balance_mean=("balance", "mean"),
            balance_std=("balance", "std"),
            balance_min=("balance", "min"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            auc_mean=("auc", "mean"),
            logloss_mean=("logloss", "mean"),
            folds_60_60=("meets_60_60", "sum"),
        )
        .reset_index()
    )
    leaderboard["complete"] = leaderboard.folds.eq(len(config["outer_folds"]))
    leaderboard = leaderboard.sort_values(
        ["complete", "balance_mean", "f1_mean", "logloss_mean"],
        ascending=[False, False, False, True],
    )
    complete = leaderboard.loc[leaderboard.complete]
    highlighted = complete.iloc[0].model if len(complete) else None
    diagnostic_models = {highlighted, "historical", "reference"}
    diagnostic_frames = {}
    train = metrics.loc[metrics.split.eq("train")]
    gaps = train.merge(
        validation,
        on=["model", "cut"],
        suffixes=("_train", "_validation"),
        validate="one_to_one",
    )
    for metric in METRICS:
        gaps[f"{metric}_gap"] = gaps[f"{metric}_train"] - gaps[f"{metric}_validation"]
    data = development_data(config)
    events, features, _ = feature_sets(data)
    segments = _segments(events, features)
    intervals, errors, calibration, transfer_diagnostics = [], [], [], []
    for cut, fold in validation.groupby("cut"):
        baseline_row = fold.loc[fold.model.eq("historical")]
        mask = events.dia_visita.between(int(cut) + 1, int(fold.end.max()))
        expected = events.loc[mask, KEYS + ["incidencia_compra"]].rename(
            columns={"incidencia_compra": "y"}
        )
        baseline = (
            _predictions(sources[("historical", int(cut))], "validation", expected)
            if len(baseline_row)
            else None
        )
        for _, row in fold.iterrows():
            frame = _predictions(sources[(row.model, int(cut))], "validation", expected)
            if row.model in diagnostic_models:
                diagnostic_frames[(row.model, int(cut))] = frame
            bins, transfer = _calibration_and_transfer(frame, row.threshold)
            context = {"model": row.model, "cut": int(cut), "end": int(row.end)}
            calibration.extend({**context, **bin_record} for bin_record in bins)
            transfer_diagnostics.append({**context, **transfer})
            errors.extend(
                _error_segments(frame, segments, row.threshold, row.model, int(cut))
            )
            if baseline is not None and row.model != "historical":
                comparisons = _paired_bootstrap(
                    frame,
                    baseline,
                    row.threshold,
                    float(baseline_row.iloc[0].threshold),
                )
                intervals.extend(
                    {"model": row.model, "cut": int(cut), **record}
                    for record in comparisons
                )
    for name, table in {
        "metrics": metrics,
        "leaderboard": leaderboard,
        "gaps": gaps,
        "times": times,
        "bootstrap_vs_historical": pd.DataFrame(intervals),
        "error_segments": pd.DataFrame(errors),
        "calibration_by_model": pd.DataFrame(calibration),
        "threshold_transfer_diagnostic": pd.DataFrame(transfer_diagnostics),
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    _plot(metrics, output)
    _plot_neural_curves(config, output)
    _parameter_importances(sources, output)
    write_experiment_inventory(config)
    if highlighted is not None:
        _plot_selected_diagnostics(
            highlighted, validation, diagnostic_frames, config, output
        )
    summary = _markdown_table(
        leaderboard,
        [
            "model",
            "folds",
            "balance_mean",
            "balance_min",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "auc_mean",
            "folds_60_60",
        ],
    )
    text = f"""# Revisión de propensión: resultados de desarrollo

## Comparación

Ranking: promedio simple por fold de min(precisión, recall); desempate F1 y menor log loss. Solo candidatos con todos los folds completados se consideran comparables para el ranking completo. El umbral se elige internamente; las tablas exteriores no participan en esa búsqueda. Un procedimiento puede escoger diferentes configuraciones en distintos cortes.

{summary}

Las métricas están en escala 0–1. `metrics.csv` contiene train y validación por fold, `gaps.csv` las diferencias train menos validación y `figures/` su comparación visual. Histórico y referencia fija se evalúan con los mismos períodos y sus propios umbrales internos. No se selecciona automáticamente un artefacto final.

`figures/neural_curves.png` muestra pérdidas del entrenamiento y de la ventana interna de parada, con épocas elegidas y ejecutadas. Estas curvas no utilizan etiquetas del fold exterior. `parameter_importances.json` describe la importancia de hiperparámetros para las puntuaciones internas de Optuna cuando hay al menos diez pruebas completadas; no es importancia de variables ni una interpretación causal.

`figures/selected_candidate_diagnostics.png` destaca el procedimiento completo con mayor equilibrio observado, sin declararlo ganador confirmado. Compara sus curvas precisión–recall con histórico y referencia, marca la decisión con el umbral interno ya fijado y muestra matrices de confusión en visitas. La zona verde indica precisión y recall de al menos 60%; recorrer la curva es diagnóstico, no selección de otro umbral.

## Incertidumbre y errores

`bootstrap_vs_historical.csv` usa 200 remuestras pareadas de clientes, semilla 42, para diferencias de precisión, recall, F1 y AUC frente al histórico. Los intervalos percentiles de 95% describen variación entre clientes en los períodos observados; no corrigen dependencia tienda/día ni selección repetida de modelos. No constituyen garantía de mejora futura.

`error_segments.csv` muestra TP/FP/FN/TN, clientes, visitas y métricas por recencia, frecuencia, edad, género, promoción y bloque de 30 días. Segmentos con menos de 100 visitas o 20 clientes se marcan de bajo soporte. Son asociaciones diagnósticas, no causas demostradas. Las segmentaciones se superponen y no deben sumarse entre dimensiones.

`calibration_by_model.csv` compara probabilidad media y tasa observada en diez intervalos fijos de probabilidad, con visitas y clientes como soporte. `threshold_transfer_diagnostic.csv` compara el equilibrio desplegado con el mejor equilibrio alcanzable al recorrer la curva de esas mismas predicciones exteriores. Este segundo valor es un diagnóstico oráculo posterior al resultado, optimista por construcción: no elige modelos ni umbrales, no se usa para entrenar y no demuestra un límite de los datos. No se exporta el umbral oráculo. Ambos diagnósticos utilizan etiquetas exteriores después de la evaluación; cualquier hipótesis motivada por ellos debe etiquetarse como exploratoria.

## Supervisión y límites

Skills aplicadas en la revisión: negocio y EDA para contrato/disponibilidad y diagnóstico; competitive-data-science y machine-learning para variables/modelos/ensambles; scikit-learn para preprocesamiento local; tuning para búsquedas reanudables; guía neuronal para curvas y parada; clean-code/deslop para legibilidad; prompt-engineering para criterios explícitos. Las skills orientan decisiones, no certifican resultados.

Los días 621–730 no se consultan por este informe. Fueron examinados históricamente; tampoco los folds de desarrollo son evidencia completamente independiente de la historia del proyecto. No se afirma ausencia de sobreajuste ni imposibilidad de alcanzar 60/60. Las medias exteriores resumen evaluaciones temporales, no validan el artefacto reajustado final.

`times.csv` recoge duración y pruebas por tarea; `runtime_summary.json` conserva los registros de reloj global disponibles ({", ".join(runtime) if runtime else "todavía no disponibles"}). Tareas paralelas se solapan: sus tiempos no se suman como tiempo de reloj. Los contadores de pruebas no equivalen al número de entrenamientos; consultar configuraciones y trazas para árboles, épocas y reajustes.

`experiment_inventory.csv` y `inventory_summary.json` separan pruebas completas, ajustes inferidos de modelos clásicos y ajustes neuronales registrados, incluyendo épocas y memoria máxima. El trabajo compartido de ensambles se cuenta una vez; el piloto GPU se presenta aparte. No incluyen entrenamientos sintéticos y no pueden reconstruir ajustes parciales fallidos sin registro.
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    return leaderboard
