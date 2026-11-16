"""Visualización del EDA y evaluación con denominadores explícitos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .data import PRICES, PROMOS

# Adaptación propia de publicaciones de McKinsey; no es un tema oficial.
# Referencias y decisiones visuales documentadas en README.
NAVY = "#051C2C"
BLUE = "#2251FF"
TEAL = "#008C95"
GRAY = "#8593A0"
LIGHT = "#E8EDF1"
PALETTE = [BLUE, TEAL, NAVY, "#6784A3", "#87BDD0"]
MODEL_COLORS = {
    "catboost": BLUE,
    "lightgbm": TEAL,
    "xgboost": NAVY,
    "hist": "#6784A3",
    "random_forest": "#A7B6C4",
    "logistic": "#9ABABD",
    "historical": GRAY,
    "constant": "#BCC5CE",
}
EXHIBITS = {
    "01_target_tiempo": (
        "01 · COMPRA Y TIEMPO",
        "La mayoría de las visitas no termina en compra",
        "Días 1–296 · visitas, clientes y evolución de la conversión",
    ),
    "02_perfil": (
        "02 · PERFIL DEL CLIENTE",
        "El perfil cambia según contemos personas o visitas",
        "Días 1–296 · una observación por cliente para describir la población",
    ),
    "03_demografia": (
        "03 · COMPOSICIÓN",
        "Los grupos demográficos tienen coberturas desiguales",
        "Clientes únicos · códigos conservados sin inventar etiquetas",
    ),
    "04_comportamiento": (
        "04 · RECURRENCIA",
        "Frecuencia y conversión varían entre clientes",
        "Días 1–296 · actividad, intervalos y concentración de compras",
    ),
    "05_marcas": (
        "05 · ELECCIÓN DE MARCA",
        "La repetición de marca domina varias transiciones",
        "Visitas con compra · proporciones condicionadas a volver a comprar",
    ),
    "06_comercial": (
        "06 · PRECIO Y PROMOCIÓN",
        "Las condiciones comerciales cambian durante el periodo",
        "Días 1–296 · asociaciones observadas, no efectos causales",
    ),
    "07_asociaciones": (
        "07 · ASOCIACIONES",
        "Precios y promociones contienen información relacionada",
        "Correlación de Spearman · asociación, no causalidad",
    ),
    "08_validacion": (
        "08 · COMPARACIÓN TEMPORAL",
        "Las diferencias entre modelos son pequeñas frente al cambio temporal",
        "Cuatro ventanas de desarrollo · rango observado, no intervalo de confianza",
    ),
    "09_evaluacion": (
        "09 · EVALUACIÓN",
        "El modelo mejora el ranking frente al historial del cliente",
        "Benchmark días 585–730 previamente consultado · no es un test intacto",
    ),
    "10_shap_global": (
        "10 · EXPLICACIÓN GLOBAL",
        "La historia de compra concentra gran parte de la señal del modelo",
        "SHAP del estimador base · muestra reproducible de 400 visitas",
    ),
    "11_shap_alta": (
        "11A · EXPLICACIÓN INDIVIDUAL",
        "Qué sostiene una predicción alta",
        "Azul: aumenta el score; verde azulado: lo reduce. Caso seleccionado por score.",
    ),
    "11_shap_baja": (
        "11B · EXPLICACIÓN INDIVIDUAL",
        "Qué sostiene una predicción baja",
        "Azul: aumenta el score; verde azulado: lo reduce. Caso seleccionado por score.",
    ),
    "11_shap_mediana": (
        "11C · EXPLICACIÓN INDIVIDUAL",
        "Cómo se compone una predicción intermedia",
        "Azul: aumenta el score; verde azulado: lo reduce. Caso seleccionado por score.",
    ),
    "12_edad_incertidumbre": (
        "12 · EDAD E INCERTIDUMBRE",
        "Las tasas por edad deben leerse junto a su incertidumbre",
        "IC del 95 % por bootstrap de clientes · 300 remuestreos",
    ),
    "13_interaccion": (
        "13 · HISTORIAL",
        "La relación entre recencia y conversión depende del historial",
        "Días 1–296 · tasas y soporte de cada combinación",
    ),
    "14_errores": (
        "14 · ERRORES Y ESTABILIDAD",
        "Los scores requieren seguimiento a medida que cambia el periodo",
        "Benchmark previamente consultado · distribución y evolución de probabilidades",
    ),
    "15_shap_dependence": (
        "15 · DEPENDENCIA DEL MODELO",
        "Una atribución predictiva no demuestra un efecto causal",
        "Variables de mayor atribución media · score base antes de calibración",
    ),
}


def apply_theme():
    """Estética editorial: blanco, azul profundo y jerarquía sin decoración."""
    sns.set_theme(style="white", palette=PALETTE)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 16,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#B7C2CB",
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.color": "#526371",
            "ytick.color": "#526371",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def style_table(value):
    """Aplica el mismo sistema a tablas, sin alterar datos ni formatos existentes."""
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        value = value.style.format(precision=3, na_rep="—")
    return value.set_table_styles(
        [
            {
                "selector": "",
                "props": [
                    ("font-family", "Arial, sans-serif"),
                    ("font-size", "12px"),
                    ("border-collapse", "collapse"),
                    ("color", NAVY),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", NAVY),
                    ("color", "white"),
                    ("padding", "9px 12px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("padding", "7px 12px"),
                    ("border-bottom", "1px solid #E8EDF1"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("padding", "7px 12px"),
                    ("text-align", "left"),
                    ("font-weight", "normal"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#F5F8FA")],
            },
        ]
    )


def finish_figure(figure, name):
    """Añade una lectura editorial y fuente sin modificar las marcas de datos."""
    import textwrap

    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    label, title, subtitle = EXHIBITS[name]
    if figure._suptitle is not None:
        figure._suptitle.remove()
        figure._suptitle = None
    # El espacio se reserva en pulgadas para que los títulos no invadan los paneles.
    width, height = figure.get_size_inches()
    figure.set_size_inches(width, height + 1.1, forward=True)
    for axis in figure.axes:
        axis.set_facecolor("white")
        axis.tick_params(length=0, pad=6)
        rate_labels = {
            "Compras / visitas",
            "Conversión",
            "Proporción de clientes",
            "Fracción acumulada de compras",
            "Frecuencia observada",
            "Precision",
            "Verdaderos positivos",
            "Compras capturadas",
            "Proporción",
            "Fracción de visitas",
            "Recall",
            "Falsos positivos",
            "Probabilidad predicha",
            "Fracción seleccionada",
            "Fracción de clientes (mayor a menor compra)",
            "Proporción acumulada",
        }
        if axis.get_ylabel() in rate_labels:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        if axis.get_xlabel() in rate_labels:
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        if axis.get_label() != "<colorbar>":
            axis.grid(False)
            if not axis.collections or axis.lines:
                axis.yaxis.grid(True, color=LIGHT, linewidth=0.6)
                axis.set_axisbelow(True)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    figure.tight_layout(rect=(0.015, 0.075, 0.985, 0.82), pad=1.5, h_pad=3, w_pad=3)
    figure.text(
        0.035,
        0.975,
        label,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=BLUE,
    )
    wrapped_title = textwrap.fill(title, width=max(45, int(width * 8)))
    figure.text(
        0.035,
        0.935,
        wrapped_title,
        ha="left",
        va="top",
        fontsize=18,
        fontfamily="DejaVu Serif",
        color=NAVY,
        linespacing=1.15,
    )
    figure.text(
        0.035, 0.843, subtitle, ha="left", va="bottom", fontsize=9, color="#526371"
    )
    figure.add_artist(
        Line2D(
            [0.035, 0.98],
            [0.06, 0.06],
            transform=figure.transFigure,
            color=NAVY,
            linewidth=0.7,
        )
    )
    figure.text(
        0.035,
        0.027,
        "Fuente: compras_data.csv; análisis del proyecto. Unidades y población indicadas en cada panel.",
        fontsize=8,
        color="#526371",
        va="bottom",
    )


def save(figure, out, name, close=True):
    """Exporta un exhibit coherente; permite mostrarlo también en el notebook."""
    Path(out).mkdir(parents=True, exist_ok=True)
    finish_figure(figure, name)
    figure.savefig(Path(out) / f"{name}.png", dpi=170)
    if close:
        plt.close(figure)


apply_theme()


def explanation_waterfall(explanation, out, name):
    """Representa contribuciones SHAP ya calculadas sin ejes gemelos superpuestos."""
    values = np.asarray(explanation.values)
    order = np.argsort(-np.abs(values))[:9]
    remaining = float(values.sum() - values[order].sum())
    contributions = np.r_[remaining, values[order[::-1]]]
    labels = ["Otras variables"] + [explanation.feature_names[i] for i in order[::-1]]
    base = float(explanation.base_values)
    starts = base + np.r_[0, contributions.cumsum()[:-1]]
    prediction = base + float(values.sum())
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = [BLUE if value >= 0 else TEAL for value in contributions]
    axis.barh(
        np.arange(len(labels)), contributions, left=starts, color=colors, height=0.62
    )
    axis.set_yticks(np.arange(len(labels)), labels)
    extent = max(float(np.ptp(np.r_[starts, starts + contributions])), 0.1)
    for row, (start, value) in enumerate(zip(starts, contributions, strict=True)):
        axis.text(
            start + value + np.sign(value) * extent * 0.018,
            row,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
            color=NAVY,
        )
    axis.axvline(base, linestyle=":", color=GRAY, linewidth=1)
    limits = np.r_[starts, starts + contributions]
    axis.set_xlim(limits.min() - 0.2 * extent, limits.max() + 0.2 * extent)
    axis.set(
        title=f"Score base: {base:.3f} → predicción: {prediction:.3f}",
        xlabel="Contribución acumulada al score del estimador (log-odds)",
    )
    save(figure, out, name)


def eda_figures(data, out):
    """Describe las visitas y clientes de los primeros 296 días."""
    development = data[data.dia_visita.le(296)].copy()
    clients = development.drop_duplicates("id")
    customers = development.groupby("id").agg(
        visitas=("id", "size"),
        compras=("incidencia_compra", "sum"),
        conversion=("incidencia_compra", "mean"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = development.incidencia_compra.value_counts().sort_index()
    axes[0, 0].bar(["No compra", "Compra"], counts)
    axes[0, 0].set_ylim(0, counts.max() * 1.22)
    for i, v in enumerate(counts):
        axes[0, 0].text(
            i, v, f"{v:,}\n{v / len(development):.1%}", ha="center", va="bottom"
        )
    axes[0, 0].set(
        title=f"Compra por visita · n={len(development):,}", ylabel="Visitas"
    )
    customers.conversion.hist(ax=axes[0, 1], bins=20)
    axes[0, 1].set(
        title=f"Conversión por cliente · n={len(customers)}",
        xlabel="Compras / visitas",
        ylabel="Clientes",
    )
    period = (
        development.assign(periodo=(development.dia_visita - 1) // 7 + 1)
        .groupby("periodo")
        .incidencia_compra.agg(["mean", "size"])
    )
    axes[1, 0].plot(period.index, period["mean"], marker="o")
    axes[1, 0].plot(
        period.index,
        period["mean"].rolling(4, min_periods=1).mean(),
        color=TEAL,
        label="Promedio móvil de 4 bloques",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set(
        title="Conversión por bloques de 7 días",
        xlabel="Semana desde apertura · semana 43: 2 días",
        ylabel="Conversión",
    )
    axes[1, 1].bar(period.index, period["size"])
    axes[1, 1].set(
        title="Visitas por bloque de 7 días",
        xlabel="Semana desde apertura · semana 43: 2 días",
        ylabel="Visitas",
    )
    save(figure, out, "01_target_tiempo")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(clients.edad, bins=15, ax=axes[0, 0])
    axes[0, 0].set(
        title="Edad: una observación por cliente", xlabel="Edad", ylabel="Clientes"
    )
    sns.ecdfplot(clients.ingreso_anual, ax=axes[0, 1])
    axes[0, 1].set(
        title="Ingreso anual: distribución acumulada",
        xlabel="Ingreso (moneda desconocida)",
        ylabel="Proporción de clientes",
    )
    sns.histplot(
        data=development,
        x="edad",
        hue="incidencia_compra",
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        bins=15,
        ax=axes[1, 0],
    )
    axes[1, 0].set(title="Edad por resultado: ponderada por visitas", ylabel="Densidad")
    age = (
        development.assign(rango=pd.cut(development.edad, [17, 29, 39, 49, 59, 80]))
        .groupby("rango", observed=True)
        .incidencia_compra.agg(["mean", "size"])
    )
    age["mean"].plot.bar(ax=axes[1, 1])
    axes[1, 1].set(
        title="Conversión por edad y soporte", ylabel="Compras / visitas", xlabel="Edad"
    )
    for i, r in enumerate(age.itertuples()):
        axes[1, 1].text(i, r.mean, f"n={r.size}", ha="center", va="bottom", fontsize=8)
    save(figure, out, "02_perfil")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, axis in zip(
        ["genero", "estado_civil", "nivel_educacion", "ocupacion"], axes.flat
    ):
        clients[col].value_counts().sort_index().plot.bar(ax=axis)
        axis.set(
            title=f"{col}: clientes únicos",
            xlabel="Código sin etiqueta documentada",
            ylabel="Clientes",
        )
    save(figure, out, "03_demografia")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    customers.visitas.hist(ax=axes[0, 0], bins=20)
    axes[0, 0].set(
        title="Frecuencia de visitas por cliente", xlabel="Visitas", ylabel="Clientes"
    )
    gaps = development.groupby("id").dia_visita.diff().dropna()
    sns.ecdfplot(gaps, ax=axes[0, 1])
    axes[0, 1].set(
        title="Tiempo entre visitas observadas",
        xlabel="Días",
        ylabel="Proporción acumulada",
    )
    sns.scatterplot(
        data=customers, x="visitas", y="conversion", alpha=0.5, ax=axes[1, 0]
    )
    axes[1, 0].set(title="Frecuencia y conversión por cliente")
    cumulative = (
        customers.compras.sort_values(ascending=False).cumsum()
        / customers.compras.sum()
    )
    axes[1, 1].plot(np.arange(1, len(cumulative) + 1) / len(cumulative), cumulative)
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 1].set(
        title="Concentración de compras observadas",
        xlabel="Fracción de clientes (mayor a menor compra)",
        ylabel="Fracción acumulada de compras",
    )
    save(figure, out, "04_comportamiento")
    purchases = development[development.incidencia_compra.eq(1)].copy()
    purchases["prev"] = purchases.groupby("id").id_marca.shift()
    transitions = pd.crosstab(purchases.prev, purchases.id_marca)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    purchases.id_marca.value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set(title="Marca por visita con compra", xlabel="Marca", ylabel="Visitas")
    purchases.groupby("id_marca").cantidad.sum().plot.bar(ax=axes[1])
    axes[1].set(title="Unidades por marca", xlabel="Marca", ylabel="Unidades")
    sns.heatmap(
        transitions.div(transitions.sum(axis=1), axis=0),
        annot=True,
        fmt=".0%",
        cmap="Blues",
        ax=axes[2],
    )
    axes[2].set(
        title=f"Transición entre compras · n={transitions.to_numpy().sum()}",
        xlabel="Marca siguiente",
        ylabel="Marca anterior",
    )
    save(figure, out, "05_marcas")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    development.groupby("dia_visita")[PRICES].median().plot(ax=axes[0, 0])
    axes[0, 0].set(
        title="Precios medianos por día",
        ylabel="Precio (moneda desconocida)",
        xlabel="Día",
    )
    sns.boxplot(data=development[PRICES], ax=axes[0, 1])
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].set(title="Variación de precios por visita", ylabel="Precio")
    development[PROMOS].mean().plot.bar(ax=axes[1, 0])
    axes[1, 0].set(
        title="Exposición a promociones por visita", ylabel="Fracción de visitas"
    )
    axes[1, 0].tick_params(axis="x", rotation=35)
    promo = (
        development.assign(promociones=development[PROMOS].sum(axis=1))
        .groupby("promociones")
        .incidencia_compra.agg(["mean", "size"])
    )
    promo["mean"].plot.bar(ax=axes[1, 1])
    axes[1, 1].set(
        title="Promociones simultáneas y compra (asociación)",
        ylabel="Conversión",
        xlabel="Número de marcas promocionadas",
    )
    for i, r in enumerate(promo.itertuples()):
        axes[1, 1].text(i, r.mean, f"n={r.size}", ha="center", va="bottom", fontsize=8)
    save(figure, out, "06_comercial")
    figure, axes = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        development[
            ["edad", "ingreso_anual"] + PRICES + PROMOS + ["incidencia_compra"]
        ].corr(method="spearman"),
        vmin=-1,
        vmax=1,
        cmap=sns.diverging_palette(240, 185, s=85, l=45, as_cmap=True),
        ax=axes,
    )
    axes.set_title("Asociación Spearman · variables con interpretación numérica")
    save(figure, out, "07_asociaciones")
    return customers, period, transitions


def evaluation_figures(cv, pred, out):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(
        data=cv,
        x="fold",
        y="ap",
        hue="model",
        palette=MODEL_COLORS,
        style="model",
        markers=True,
        ax=axes[0],
    )
    axes[0].set(title="AP por ventana temporal", ylabel="Average precision")
    comparison = (
        cv.loc[cv.model.ne("constant")]
        .groupby("model", sort=False)
        .ap.agg(["mean", "min", "max"])
    )
    axes[1].errorbar(
        np.arange(len(comparison)),
        comparison["mean"],
        yerr=[
            comparison["mean"] - comparison["min"],
            comparison["max"] - comparison["mean"],
        ],
        fmt="o",
        capsize=4,
    )
    axes[1].set_xticks(np.arange(len(comparison)), comparison.index, rotation=45)
    axes[1].set(
        title="Media y rango de AP entre cuatro periodos", ylabel="Average precision"
    )
    save(figure, out, "08_validacion")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for label, column in [("Modelo", "p"), ("Tasa histórica", "baseline")]:
        line_color = BLUE if column == "p" else GRAY
        probabilities = pred[column]
        precision, recall, _ = precision_recall_curve(pred.y, probabilities)
        axes[0, 0].plot(recall, precision, label=label, color=line_color)
        fpr, tpr, _ = roc_curve(pred.y, probabilities)
        axes[0, 1].plot(fpr, tpr, label=label, color=line_color)
        obs, prob = calibration_curve(
            pred.y, probabilities, n_bins=10, strategy="quantile"
        )
        axes[1, 0].plot(prob, obs, "o-", label=label, color=line_color)
        ordered = pred.iloc[np.argsort(-probabilities.to_numpy())]
        axes[1, 1].plot(
            np.arange(1, len(pred) + 1) / len(pred),
            ordered.y.cumsum() / pred.y.sum(),
            label=label,
            color=line_color,
        )
    axes[0, 0].axhline(pred.y.mean(), ls="--", color="gray")
    axes[0, 0].set(title="Precision–recall", xlabel="Recall", ylabel="Precision")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(
        title="ROC", xlabel="Falsos positivos", ylabel="Verdaderos positivos"
    )
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set(
        title="Calibración",
        xlabel="Probabilidad predicha",
        ylabel="Frecuencia observada",
    )
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 1].set(
        title="Ganancias acumuladas por visita",
        xlabel="Fracción seleccionada",
        ylabel="Compras capturadas",
    )
    for axis in axes.flat:
        axis.legend()
    figure.suptitle("Benchmark días 585–730 previamente consultado", y=1.02)
    save(figure, out, "09_evaluacion")
