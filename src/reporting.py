"""Tablas y figuras del notebook; no entrena modelos ni lee resultados guardados."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.ticker import PercentFormatter

from .plotting import BLUE, TEAL, apply_theme
from .training import INNER_FOLDS, OUTER_FOLDS


def show_eda(result):
    """Muestra la figura y conserva sus tablas fuente para inspección opcional."""
    figure, tables = result
    display(figure)
    plt.close(figure)
    return tables


def metric_table(table):
    names = {
        "model": "Modelo",
        "cut": "Fin train",
        "split": "Muestra",
        "n": "Visitas",
        "precision": "Precisión",
        "recall": "Recall",
        "f1": "F1",
        "auc": "ROC AUC",
        "ap": "AP",
        "logloss": "Log loss",
    }
    formats = {
        "Precisión": "{:.1%}",
        "Recall": "{:.1%}",
        "F1": "{:.1%}",
        "ROC AUC": "{:.3f}",
        "AP": "{:.3f}",
        "Log loss": "{:.3f}",
        "Visitas": "{:,.0f}",
    }
    display(table.rename(columns=names).style.format(formats).hide(axis="index"))


def show_protocol(events):
    rows = []
    for fold, (cut, end) in enumerate(OUTER_FOLDS, 1):
        rows.append(
            {
                "Fold": fold,
                "Train · días": f"1–{cut}",
                "Validación · días": f"{cut + 1}–{end}",
                "Visitas train": int(events.dia_visita.le(cut).sum()),
                "Visitas validación": int(
                    events.dia_visita.between(cut + 1, end).sum()
                ),
                "Validaciones internas · días": "; ".join(
                    f"{a + 1}–{b}" for a, b in INNER_FOLDS[cut]
                ),
            }
        )
    display(pd.DataFrame(rows).style.hide(axis="index"))


def show_search(run):
    study, bundle = run["study"], run["bundle"]
    display(
        Markdown(
            f"**{len(study.trials)} trials** · Log loss interna **{study.best_value:.4f}**"
            f" · Umbral **{bundle['threshold']:.3f}** · {bundle['iterations']} árboles"
            f" · {study.user_attrs['seconds']:.1f} s"
        )
    )
    metric_table(pd.DataFrame(run["metrics"]))


def show_results(table):
    """Promedios por fold y barras de train/validación contra el histórico."""
    apply_theme()
    validation = table.loc[table.split.eq("Validación")]
    summary = validation.groupby("model", as_index=False)[
        ["precision", "recall", "f1", "auc", "ap", "logloss"]
    ].mean()
    metric_table(summary)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    series = [
        ("LightGBM", "Train", "#A9BBFF"),
        ("LightGBM", "Validación", BLUE),
        ("Histórico", "Train", "#A3D6D7"),
        ("Histórico", "Validación", TEAL),
    ]
    positions = np.arange(3)
    width = 0.19
    for ax, metric, label in zip(
        axes, ["precision", "recall", "f1"], ["Precisión", "Recall", "F1"]
    ):
        for index, (model, split, color) in enumerate(series):
            part = table.loc[table.model.eq(model) & table.split.eq(split)].sort_values(
                "cut"
            )
            bars = ax.bar(
                positions + (index - 1.5) * width,
                part[metric],
                width,
                color=color,
                label=f"{model} · {split}",
            )
            ax.bar_label(
                bars,
                labels=[f"{100 * x:.1f}" for x in part[metric]],
                padding=3,
                fontsize=8,
                rotation=90,
            )
        ax.set(
            title=label,
            xlabel="Fold",
            ylim=(0, 0.82),
            xticks=positions,
            xticklabels=["1\nVal. 297–404", "2\nVal. 405–512", "3\nVal. 513–620"],
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Entrenamiento y validación por fold · comparación con el histórico",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    display(fig)
    plt.close(fig)
    final = validation.loc[validation.cut.eq(512)].set_index("model")
    delta = 100 * (final.loc["LightGBM", "f1"] - final.loc["Histórico", "f1"])
    display(
        Markdown(
            f"**Último fold:** el F1 de LightGBM cambia **{delta:+.2f} puntos** frente al histórico. "
            "La mejora debe juzgarse junto con precisión, recall y estabilidad; no demuestra impacto comercial."
        )
    )


def show_importance(bundle, features, events, config):
    """SHAP nativo sobre una muestra fija de validación; no implica causalidad."""
    sample = features.loc[events.dia_visita.between(513, 620)].sample(
        n=1000, random_state=config.seed
    )
    transformer, estimator = bundle["model"].steps[0][1], bundle["model"].steps[-1][1]
    contributions = estimator.predict(transformer.transform(sample), pred_contrib=True)
    names = transformer.get_feature_names_out()
    importance = (
        pd.Series(np.abs(contributions[:, :-1]).mean(axis=0), index=names)
        .nlargest(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot.barh(ax=ax, color="#2251FF")
    ax.set(
        xlabel="Media |contribución SHAP|",
        title="Las 15 variables con mayor contribución al modelo",
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_correlations(events, candidates, ranking):
    """Correlaciones del período usado por el selector, sin mirar validación."""
    import seaborn as sns

    columns = ranking.variable.head(25).tolist()
    corr = candidates.loc[events.dia_visita.le(120), columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        mask=np.triu(np.ones(corr.shape, dtype=bool), 1),
        cbar_kws={"label": "Correlación Spearman"},
    )
    ax.set_title("Señales de historial comparten información · días 1–120", pad=18)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    pairs = [
        {"Variable A": a, "Variable B": b, "Spearman": corr.loc[a, b]}
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if abs(corr.loc[a, b]) >= 0.90
    ]
    pairs = pd.DataFrame(pairs).sort_values("Spearman", key=abs, ascending=False)
    display(pairs.head(8).style.format({"Spearman": "{:.3f}"}).hide(axis="index"))
    return pairs


def show_final_results(table):
    """Resultados del modelo final y del promedio de validación, sin mezclar muestras."""
    model = table.loc[table.model.eq("LightGBM")]
    columns = ["precision", "recall", "f1", "auc", "ap", "logloss"]
    mean = model.loc[model.split.eq("Validación"), columns].mean().to_dict()
    mean["split"] = "Validación · promedio de 3 folds"
    final = model.loc[
        model.cut.eq(512) & model.split.isin(["Train", "Validación", "Test"]),
        ["split", "n", *columns],
    ].copy()
    final["split"] = final["split"].replace(
        {
            "Train": "Train · 1–512",
            "Validación": "Validación · 513–620",
            "Test": "Test · 621–730",
        }
    )
    metric_table(final)
    metric_table(pd.DataFrame([mean]))


def show_predictions(scored, raw, threshold):
    """Ejemplos legibles; el resultado real se incorpora después de predecir."""
    example = scored.head().merge(
        raw[["id", "dia_visita", "incidencia_compra"]],
        on=["id", "dia_visita"],
        validate="one_to_one",
    )
    example["prediction"] = example.prediction.map({1: "Compra", 0: "No compra"})
    example["incidencia_compra"] = example.incidencia_compra.map(
        {1: "Compra", 0: "No compra"}
    )
    example = example.rename(
        columns={
            "id": "Cliente",
            "dia_visita": "Día",
            "probability": "Probabilidad de compra",
            "prediction": "Predicción",
            "incidencia_compra": "Resultado observado",
        }
    )
    display(Markdown(f"**Umbral de decisión: {threshold:.1%}.**"))
    display(
        example.style.format({"Probabilidad de compra": "{:.1%}"}).hide(axis="index")
    )
