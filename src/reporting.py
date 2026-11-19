"""Tablas y figuras del notebook; no entrena modelos ni lee resultados guardados."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.ticker import PercentFormatter

from .plotting import BLUE, GRAY, apply_theme
from .training import INNER_FOLDS, OUTER_FOLDS

RANK_METRICS = ["auc", "ks", "gini"]


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
        "ks": "KS",
        "gini": "Gini",
        "ap": "AP",
        "logloss": "Log loss",
    }
    formats = {
        "Precisión": "{:.1%}",
        "Recall": "{:.1%}",
        "F1": "{:.1%}",
        "ROC AUC": "{:.3f}",
        "KS": "{:.3f}",
        "Gini": "{:.1%}",
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
    table = pd.DataFrame(run["metrics"])
    metric_table(table.loc[table.model.eq("LightGBM"), ["split", "n", *RANK_METRICS]])


def show_results(table):
    """Compara las reglas en una tabla, con igual peso por fold."""
    splits = ["Train", "Validación", "Test"]
    mean = table.groupby(["model", "split"])[RANK_METRICS].mean()
    comparison = mean.unstack("split")
    comparison = comparison.reindex(
        columns=pd.MultiIndex.from_product([RANK_METRICS, splits])
    )
    comparison = comparison.sort_values(("auc", "Validación"), ascending=False)
    comparison = comparison.rename(
        columns=dict(zip(RANK_METRICS, ["ROC AUC", "KS", "Gini"], strict=True)),
        level=0,
    )
    formats = {
        column: "{:.1%}" if column[0] == "Gini" else "{:.3f}"
        for column in comparison.columns
    }
    display(comparison.rename_axis("Método").style.format(formats))


def decile_summary(rows):
    """Suma deciles de períodos ya ordenados por separado y deriva sus tasas."""
    table = (
        pd.DataFrame(rows)
        .groupby(["model", "decile"], as_index=False)[["visits", "purchases"]]
        .sum()
    )
    by_model = table.groupby("model", sort=False)
    table["conversion"] = table.purchases / table.visits
    table["coverage"] = by_model.visits.cumsum() / by_model.visits.transform("sum")
    table["capture"] = by_model.purchases.cumsum() / by_model.purchases.transform("sum")
    return table


def show_deciles(rows, title):
    """Muestra la ganancia acumulada y la tabla de los diez tramos."""
    apply_theme()
    table = decile_summary(rows)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([0, 1], [0, 1], "--", color=GRAY, label="Al azar")
    axis.plot(
        np.r_[0, table.coverage],
        np.r_[0, table.capture],
        "o-",
        color=BLUE,
        label="LightGBM",
    )
    axis.set(
        title=title,
        xlabel="Visitas priorizadas",
        ylabel="Compras reales capturadas",
        xlim=(0, 1),
        ylim=(0, 1.03),
        xticks=np.linspace(0, 1, 11),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="lower right")
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    counts = table.rename(
        columns={
            "decile": "Decil",
            "visits": "Visitas",
            "purchases": "Compras",
            "conversion": "Conversión",
            "capture": "Compras capturadas",
        }
    )
    display(
        counts[["Decil", "Visitas", "Compras", "Conversión", "Compras capturadas"]]
        .style.format({"Conversión": "{:.1%}", "Compras capturadas": "{:.1%}"})
        .hide(axis="index")
    )


def show_importance(bundle, features, events, config):
    """SHAP nativo sobre una muestra fija de validación; no implica causalidad."""
    sample = features.loc[events.dia_visita.between(513, 620)].sample(
        n=1000, random_state=config.seed
    )
    transformer, estimator = bundle["model"].steps[0][1], bundle["model"].steps[-1][1]
    contributions = estimator.predict(transformer.transform(sample), pred_contrib=True)
    names = transformer.get_feature_names_out()
    importance = pd.Series(
        np.abs(contributions[:, :-1]).mean(axis=0), index=names
    ).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    importance.plot.barh(ax=ax, color="#2251FF")
    ax.set(
        xlabel="Media |contribución SHAP|",
        title=f"Contribución de las {len(importance)} variables del modelo",
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


def show_final_results(table, threshold):
    """Muestra decisiones al umbral interno frente a las compras observadas."""
    test = table.loc[table.split.eq("Test") & table.model.eq("LightGBM")]
    display(
        Markdown(
            f"**Decisión por visita en test.** LightGBM predice compra "
            f"cuando el puntaje alcanza el umbral de {threshold:.1%}."
        )
    )
    row = test.iloc[0]
    counts = pd.DataFrame(
        [
            [row.true_positives, row.false_negatives],
            [row.false_positives, row.true_negatives],
        ],
        index=pd.Index(["Compró", "No compró"], name="Resultado observado"),
        columns=["Predijo compra", "Predijo ausencia de compra"],
        dtype=int,
    )
    counts["Visitas"] = counts.sum(axis=1)
    display(counts)
    metric_table(test[["precision", "recall", "f1", "logloss"]])


def show_predictions(scored, raw, threshold):
    """Ejemplos legibles; el resultado real se incorpora después de predecir."""
    example = (
        scored.sort_values(["probability", "id"], ascending=[False, True])
        .head()
        .merge(
            raw[["id", "dia_visita", "incidencia_compra"]],
            on=["id", "dia_visita"],
            validate="one_to_one",
        )
    )
    example["prediction"] = example.prediction.map({1: "Compra", 0: "No compra"})
    example["incidencia_compra"] = example.incidencia_compra.map(
        {1: "Compra", 0: "No compra"}
    )
    example = example.rename(
        columns={
            "id": "Cliente",
            "dia_visita": "Día",
            "probability": "Probabilidad estimada",
            "prediction": "Predicción",
            "incidencia_compra": "Resultado observado",
        }
    )
    example.insert(0, "Prioridad", range(1, len(example) + 1))
    display(
        Markdown(
            f"**Primeras cinco visitas del día, por puntaje.** "
            f"El umbral de {threshold:.1%} se usa para la decisión binaria."
        )
    )
    display(
        example.style.format({"Probabilidad estimada": "{:.1%}"}).hide(axis="index")
    )
