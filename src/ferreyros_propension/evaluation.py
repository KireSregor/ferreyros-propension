"""Métricas de ranking y probabilidades con incertidumbre por cliente."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def metrics(target, probabilities) -> dict[str, float]:
    """Métricas comunes; requiere ambas clases y probabilidades finitas."""
    target, probabilities = np.asarray(target), np.asarray(probabilities)
    if target.shape != probabilities.shape or len(target) == 0:
        raise ValueError(
            "Target y probabilidades deben ser vectores de igual longitud."
        )
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilidades fuera de [0, 1] o no finitas.")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("La evaluación requiere ambas clases binarias.")
    result = dict(
        ap=average_precision_score(target, probabilities),
        auc=roc_auc_score(target, probabilities),
        brier=brier_score_loss(target, probabilities),
        logloss=log_loss(target, probabilities),
        prevalencia=target.mean(),
    )
    for fraction in [0.05, 0.1, 0.2]:
        selected_count = max(1, int(np.ceil(len(target) * fraction)))
        hits = target[np.argsort(-probabilities, kind="stable")[:selected_count]].sum()
        precision = hits / selected_count if np.ptp(probabilities) else target.mean()
        recall = (
            hits / target.sum()
            if np.ptp(probabilities)
            else selected_count / len(target)
        )
        suffix = int(fraction * 100)
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"lift_{suffix}"] = precision / target.mean()
    return result


def bootstrap_delta(frame, iterations=300, seed=42):
    """Intervalo percentil de diferencia AP modelo-histórico, clientes emparejados."""
    random_generator = np.random.default_rng(seed)
    groups = [s for _, s in frame.groupby("id")]
    values = []
    for _ in range(iterations):
        sample = pd.concat(
            [groups[i] for i in random_generator.integers(0, len(groups), len(groups))]
        )
        if sample.y.nunique() == 2:
            values.append(
                average_precision_score(sample.y, sample.p)
                - average_precision_score(sample.y, sample.baseline)
            )
    return {
        "delta_ap": average_precision_score(frame.y, frame.p)
        - average_precision_score(frame.y, frame.baseline),
        "ci_low": np.quantile(values, 0.025),
        "ci_high": np.quantile(values, 0.975),
        "bootstrap": len(values),
    }


def daily_capacity(frame, fraction=0.2):
    """Selecciona visitas cada día, incluyendo redondeo de cupos."""
    chosen = frame.groupby("dia_visita", group_keys=False).apply(
        lambda s: s.nlargest(max(1, int(np.ceil(len(s) * fraction))), "p")
    )
    return {
        "seleccionadas": len(chosen),
        "fraccion_real": len(chosen) / len(frame),
        "precision": chosen.y.mean(),
        "recall": chosen.y.sum() / frame.y.sum(),
        "lift": chosen.y.mean() / frame.y.mean(),
    }


def explain_model(bundle, predictors, output_dir, sample_size=400):
    """SHAP del estimador base y permutación; no atribuye efectos causales.

    Los SHAP explican el score del estimador, no la calibración posterior.
    Se registra la muestra y el espacio de salida mediante verificación aditiva.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import shap
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.pipeline import Pipeline

    from .plotting import BLUE, TEAL, explanation_waterfall, save

    output_dir = Path(output_dir)
    sample = predictors.sample(min(sample_size, len(predictors)), random_state=42)
    model = bundle["model"]
    if isinstance(model, Pipeline):
        transformed = model[:-1].transform(sample)
        names = model[:-1].get_feature_names_out()
        estimator = model[-1]
    else:
        transformed = sample
        names = sample.columns
        estimator = model
    if bundle["family"] == "logistic":
        explanation = shap.Explainer(estimator, transformed)(transformed)
    else:
        explanation = shap.TreeExplainer(estimator)(transformed)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    explanation.feature_names = list(names)
    shap_ranking = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=names
    ).sort_values(ascending=False)
    shap_ranking.rename("mean_absolute_shap").to_csv(
        output_dir / "tables/shap_importance.csv"
    )
    plt.figure()
    color_scale = LinearSegmentedColormap.from_list(
        "feature_value", [BLUE, "#D8E3EA", TEAL]
    )
    shap.plots.beeswarm(explanation, max_display=15, color=color_scale, show=False)
    plt.xlabel("Contribución SHAP al score base (log-odds)")
    save(plt.gcf(), output_dir / "figures", "10_shap_global")
    predicted = model.predict_proba(sample)[:, 1]
    positions = {
        "alta": int(np.argmax(predicted)),
        "baja": int(np.argmin(predicted)),
        "mediana": int(np.argsort(predicted)[len(predicted) // 2]),
    }
    for label, position in positions.items():
        explanation_waterfall(
            explanation[position], output_dir / "figures", f"11_shap_{label}"
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, feature_name in zip(axes, shap_ranking.index[:2], strict=True):
        feature_position = list(names).index(feature_name)
        feature_values = (
            transformed[feature_name].to_numpy()
            if isinstance(transformed, pd.DataFrame)
            else transformed[:, feature_position]
        )
        axis.scatter(
            np.asarray(feature_values, dtype=float),
            explanation.values[:, feature_position],
            alpha=0.5,
            s=15,
        )
        axis.set(
            xlabel=feature_name,
            ylabel="Contribución SHAP al score base",
            title="Dependencia del modelo, no efecto causal",
        )
    save(figure, output_dir / "figures", "15_shap_dependence")
    sample.to_csv(output_dir / "tables/shap_sample.csv")
    additive = explanation.base_values + explanation.values.sum(axis=1)
    probability_match = np.allclose(additive, predicted, atol=1e-4)
    logit = np.log(
        np.clip(predicted, 1e-6, 1 - 1e-6) / (1 - np.clip(predicted, 1e-6, 1 - 1e-6))
    )
    logit_match = np.allclose(additive, logit, atol=1e-3)
    return {
        "output_space": "probability"
        if probability_match
        else "log_odds"
        if logit_match
        else "raw_score_verificar",
        "sample_size": len(sample),
        "top_features": list(shap_ranking.index[:5]),
        "explains": "base_estimator_before_calibration",
    }
