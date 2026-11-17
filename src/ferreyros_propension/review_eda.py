"""Bounded development EDA, with explicit units and temporal diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .data import PROMOS, audit
from .plotting import BLUE, GRAY, LIGHT, NAVY, TEAL, apply_theme
from .review import development_data, feature_sets, read_config


def _finish(figure, output, name, title, subtitle):
    figure.suptitle(title, x=0.07, y=0.99, ha="left", color=NAVY, fontsize=17)
    figure.text(0.07, 0.925, subtitle, color=GRAY, fontsize=10)
    figure.text(
        0.07,
        0.015,
        "Fuente: compras_data.csv · Desarrollo únicamente · Adaptación visual propia, no tema oficial de McKinsey.",
        color=GRAY,
        fontsize=7,
    )
    figure.tight_layout(rect=(0.05, 0.045, 1, 0.89))
    figure.savefig(output / f"{name}.png", dpi=150)
    plt.close(figure)


def _weekly_table(data, end_day):
    working = data.assign(bloque=(data.dia_visita - 1) // 7)
    weekly = (
        working.groupby("bloque")
        .incidencia_compra.agg(visitas="size", compras="sum", conversion="mean")
        .reset_index()
    )
    weekly["dia_inicio"] = weekly.bloque * 7 + 1
    weekly["dia_fin"] = np.minimum(weekly.dia_inicio + 6, end_day)
    weekly["dias_observados"] = weekly.dia_fin - weekly.dia_inicio + 1
    weekly["etiqueta"] = (
        weekly.dia_inicio.astype(str) + "–" + weekly.dia_fin.astype(str)
    )
    return weekly


def _weekly_plot(weekly, output):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].plot(weekly.bloque + 1, weekly.conversion, color=BLUE)
    axes[0].set(title="Compras / visitas por bloque", ylabel="Conversión por visita")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].bar(weekly.bloque + 1, weekly.visitas, color=BLUE, width=0.8)
    axes[1].set(title="Visitas observadas por bloque", ylabel="Número de visitas")
    selected = weekly.iloc[::6].index.union(weekly.tail(1).index)
    for axis in axes:
        axis.set_xticks(
            weekly.loc[selected, "bloque"] + 1,
            weekly.loc[selected, "etiqueta"],
            rotation=35,
        )
        axis.set_xlabel("Días de cada bloque de 7 días; último bloque parcial")
        axis.grid(axis="y", color=LIGHT)
        axis.set_axisbelow(True)
    _finish(
        figure,
        output,
        "01_semanas_iniciales",
        "La conversión cambia durante el período inicial",
        "EDA inicial · Días 1–296 · Una observación es una visita; bloques relativos, no semanas de calendario",
    )


def _customer_analysis(initial, output):
    customers = initial.groupby("id").incidencia_compra.agg(
        visitas="size", compras="sum"
    )
    customers["conversion"] = customers.compras / customers.visitas
    customers.describe().rename_axis("estadistico").to_csv(
        output / "resumen_por_cliente.csv"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].hist(customers.visitas, bins=20, color=BLUE, edgecolor="white")
    axes[0].set(
        title="Visitas acumuladas por cliente",
        xlabel="Visitas de un cliente en días 1–296",
        ylabel="Clientes",
    )
    axes[1].hist(
        customers.conversion, bins=np.linspace(0, 1, 21), color=TEAL, edgecolor="white"
    )
    axes[1].set(
        title="Compras / visitas de cada cliente",
        xlabel="Conversión de un cliente en días 1–296",
        ylabel="Clientes",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.grid(axis="y", color=LIGHT)
        axis.set_axisbelow(True)
    _finish(
        figure,
        output,
        "02_clientes_iniciales",
        "Los clientes presentan distinta actividad y conversión",
        f"EDA inicial · Días 1–296 · Cada observación del histograma es uno de {len(customers):,} clientes",
    )
    return customers


def _history_analysis(events, output):
    frequency_labels = ["0–5", "6–15", "16–30", "31 o más"]
    recency_labels = ["1–7 días", "8–30 días", "31–60 días", "Más de 60 días"]
    recency = pd.cut(
        events.dias_desde_compra, [0, 7, 30, 60, np.inf], labels=recency_labels
    )
    recency = recency.cat.add_categories("Sin compra previa").fillna(
        "Sin compra previa"
    )
    recency = recency.cat.reorder_categories(["Sin compra previa", *recency_labels])
    frequency = pd.cut(
        events.visitas_previas, [-1, 5, 15, 30, np.inf], labels=frequency_labels
    )
    table = (
        events.assign(recencia=recency, frecuencia=frequency)
        .groupby(["recencia", "frecuencia"], observed=False)
        .incidencia_compra.agg(visitas="size", compras="sum", conversion="mean")
        .reset_index()
    )
    table.to_csv(output / "recencia_frecuencia.csv", index=False)
    rates = table.pivot(index="recencia", columns="frecuencia", values="conversion")
    support = table.pivot(index="recencia", columns="frecuencia", values="visitas")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    sns.heatmap(
        rates,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        mask=support.eq(0),
        ax=axes[0],
        cbar_kws={"label": "Compras / visitas"},
    )
    sns.heatmap(
        support,
        annot=True,
        fmt=".0f",
        cmap="Greys",
        ax=axes[1],
        cbar_kws={"label": "Visitas"},
    )
    axes[0].set_title("Conversión observada por visita")
    axes[1].set_title("Soporte de la misma combinación")
    for axis in axes:
        axis.set_xlabel("Número de visitas previas del cliente")
        axis.set_ylabel("Tiempo desde la última compra anterior")
        axis.tick_params(axis="y", labelrotation=0)
        axis.tick_params(axis="x", labelrotation=0)
    _finish(
        figure,
        output,
        "03_historial_inicial",
        "El historial distingue contextos de visita con distinta conversión",
        "EDA inicial · Días 1–296 · Un cliente puede aportar varias visitas y ocupar distintas celdas",
    )


def _associations(initial, output):
    working = initial.assign(
        promocion=initial[PROMOS]
        .max(axis=1)
        .map({0: "Ninguna marca", 1: "Al menos una marca"}),
        edad_grupo=pd.cut(
            initial.edad,
            [0, 29, 39, 49, 59, np.inf],
            labels=["Hasta 29", "30–39", "40–49", "50–59", "60 o más"],
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    tables = {}
    for axis, column, filename in zip(
        axes, ["promocion", "edad_grupo"], ["promociones", "edad"], strict=True
    ):
        table = working.groupby(column, observed=True).agg(
            visitas=("incidencia_compra", "size"),
            compras=("incidencia_compra", "sum"),
            conversion=("incidencia_compra", "mean"),
            clientes=("id", "nunique"),
        )
        table.to_csv(output / f"asociacion_{filename}.csv")
        tables[filename] = table
        axis.bar(
            np.arange(len(table)),
            table.conversion,
            color=BLUE if column == "promocion" else TEAL,
        )
        axis.set_xticks(np.arange(len(table)), table.index.astype(str), rotation=15)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(0, max(0.45, float(table.conversion.max()) + 0.1))
        axis.set_ylabel("Compras / visitas del grupo")
        for position, (_, row) in enumerate(table.iterrows()):
            axis.text(
                position,
                row.conversion + 0.012,
                f"{row.conversion:.1%}\nn={int(row.visitas):,} visitas",
                ha="center",
                fontsize=8,
            )
    axes[0].set_title("Visitas según promoción disponible ese día")
    axes[1].set_title("Visitas según edad del cliente")
    _finish(
        figure,
        output,
        "04_asociaciones_iniciales",
        "Perfil y ofertas describen asociaciones, no efectos causales",
        "EDA inicial · Días 1–296 · Tasas ponderadas por visitas; una persona puede contribuir varias veces",
    )
    return tables


def _temporal_diagnostics(events, config, output):
    bounds = [(1, config["outer_folds"][0][0])]
    bounds.extend((cut + 1, end) for cut, end in config["outer_folds"])
    records = []
    for start, end in bounds:
        part = events.loc[events.dia_visita.between(start, end)]
        records.append(
            {
                "periodo": f"{start}–{end}",
                "visitas": len(part),
                "clientes": part.id.nunique(),
                "compras": int(part.incidencia_compra.sum()),
                "conversion": part.incidencia_compra.mean(),
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(output / "diagnostico_temporal.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].bar(
        table.periodo, table.conversion, color=[TEAL] + [BLUE] * (len(table) - 1)
    )
    axes[0].set(
        title="Conversión de visitas por período",
        ylabel="Compras / visitas",
        xlabel="Días observados",
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(0, 0.4)
    axes[1].bar(table.periodo, table.visitas, color=[TEAL] + [BLUE] * (len(table) - 1))
    axes[1].set(
        title="Volumen observado por período",
        ylabel="Visitas",
        xlabel="Días observados; períodos de distinta duración",
    )
    for position, row in table.iterrows():
        axes[0].text(
            position,
            row.conversion + 0.01,
            f"{row.conversion:.1%}",
            ha="center",
            fontsize=9,
        )
    _finish(
        figure,
        output,
        "05_diagnostico_desarrollo",
        "Las tasas observadas varían entre períodos de desarrollo",
        f"Diagnóstico temporal exploratorio · Días 1–{config['development_end']} · No es EDA inicial ni evaluación de test",
    )
    return table


def write_eda(config):
    """Write aggregate tables, PNG panels and interpretation without test access."""
    if config["development_end"] > 620:
        raise ValueError("La revisión EDA debe excluir los días posteriores al 620.")
    output = Path(config["output"]) / "eda"
    output.mkdir(parents=True, exist_ok=True)
    data = development_data(config)
    events, predictors, groups = feature_sets(data)
    initial = events.loc[events.dia_visita.le(296)].copy()
    quality, definitions = audit(data)
    quality.to_csv(output / "calidad_desarrollo.csv", header=["valor"])
    definitions.to_csv(output / "semantica_historial.csv", index=False)
    weekly = _weekly_table(initial, 296)
    weekly.to_csv(output / "semanas_iniciales.csv", index=False)
    apply_theme()
    _weekly_plot(weekly, output)
    customers = _customer_analysis(initial, output)
    _history_analysis(initial, output)
    _associations(initial, output)
    periods = _temporal_diagnostics(events, config, output)
    profile_changed = sum(
        int(data.groupby("id")[c].nunique().gt(1).sum())
        for c in [
            "genero",
            "estado_civil",
            "edad",
            "nivel_educacion",
            "ingreso_anual",
            "ocupacion",
        ]
    )
    report = [
        "# EDA inicial y diagnóstico temporal de desarrollo",
        "",
        "## Contrato y límites",
        "",
        "Se estima compra antes del resultado de una visita conocida. Perfil y ofertas actuales se consideran disponibles; "
        "los resultados de visitas anteriores se conocen antes de cada nueva visita. Las variables históricas se actualizan "
        "durante validación únicamente con resultados anteriores. Las asociaciones no identifican efectos causales.",
        "Días 621–730 excluidos antes de validaciones y construcción de variables. No se consultaron artefactos de test.",
        "",
        "## EDA inicial: días 1–296",
        "",
        f"- {len(initial):,} visitas de {initial.id.nunique():,} clientes; "
        f"{int(initial.incidencia_compra.sum()):,} visitas con compra ({initial.incidencia_compra.mean():.1%}).",
        f"- Un cliente registra una mediana de {customers.visitas.median():.0f} visitas. "
        f"La mediana de su conversión individual es {customers.conversion.median():.1%}; "
        "no debe confundirse con la conversión agregada de visitas.",
        f"- La conversión de los {len(weekly)} bloques de siete días varía entre "
        f"{weekly.conversion.min():.1%} y {weekly.conversion.max():.1%}. "
        "El último bloque tiene dos días: comparar sus volúmenes brutos exige considerar su menor duración.",
        "- El mapa recencia/frecuencia cuenta visitas, no clientes únicos. Un cliente puede aparecer en distintas celdas; "
        "las celdas con poco soporte no justifican conclusiones firmes.",
        "- Las tablas de promociones y edad incluyen visitas, compras y clientes únicos; las tasas se calculan por visita. "
        "Las mismas personas pueden aparecer en ambos grupos de exposición promocional.",
        "",
        "## Calidad y semántica en desarrollo",
        "",
        f"- {len(data):,} visitas, {data.id.nunique():,} clientes y {len(data.columns)} columnas originales; "
        f"{int(quality['faltantes'])} celdas faltantes y {int(quality['duplicados'])} filas duplicadas.",
        f"- Se construyen {len(predictors.columns)} variables; se comparan {len(groups)} grupos de variables. "
        f"Cambios de campos de perfil dentro del cliente: {profile_changed}.",
        "- `ultima_marca_comprada` coincide con la marca de la visita anterior; "
        "`ultima_cantidad_comprada` coincide con la incidencia de compra de la visita anterior. "
        "La comprobación cuantitativa está en `semantica_historial.csv`.",
        "",
        "## Diagnóstico temporal: no es EDA inicial",
        "",
        "Las siguientes tasas describen los períodos exteriores de desarrollo. Ya examinados históricamente, "
        "sirven para investigar estabilidad; no son una confirmación independiente.",
        "",
    ]
    report.extend(
        f"- Días {r.periodo}: {r.visitas:,} visitas, {r.clientes} clientes y conversión {r.conversion:.1%}."
        for r in periods.itertuples()
    )
    report.extend(
        [
            "",
            "## Lecturas y siguientes hipótesis",
            "",
            "Investigar tendencia de tienda estrictamente rezagada, cambios de frecuencia del cliente y diferencias "
            "en respuesta histórica a ofertas. Contrastarlas mediante ablaciones dentro de validación temporal; "
            "estas visuales no demuestran causalidad ni explican por sí solas los errores del modelo.",
            "",
            "Visuales: `01_semanas_iniciales.png`, `02_clientes_iniciales.png`, `03_historial_inicial.png`, "
            "`04_asociaciones_iniciales.png` y `05_diagnostico_desarrollo.png`. "
            "Tablas CSV conservan denominadores y límites de períodos.",
        ]
    )
    (output / "EDA.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(write_eda(read_config()))
