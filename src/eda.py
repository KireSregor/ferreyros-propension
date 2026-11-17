"""Exploración en memoria: cada función devuelve una figura y sus tablas fuente."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .data import PROMOS, audit, features
from .plotting import BLUE, GRAY, LIGHT, NAVY, TEAL, apply_theme


def _initial(data, end_day):
    selected = data.loc[data.dia_visita.between(1, end_day)].copy()
    if selected.empty:
        raise ValueError("No hay visitas en el período solicitado para el EDA.")
    return selected.sort_values(["id", "dia_visita"]).reset_index(drop=True)


def _period(data):
    return f"Días {int(data.dia_visita.min())}–{int(data.dia_visita.max())}"


def _canvas(title, subtitle, figsize=(13, 5)):
    apply_theme()
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    figure.suptitle(title, x=0.07, y=0.99, ha="left", fontsize=17, color=NAVY)
    figure.text(0.07, 0.91, subtitle, color=GRAY, fontsize=10)
    figure.text(
        0.07,
        0.015,
        "Fuente: datos suministrados · Elaboración propia · Asociaciones descriptivas.",
        color=GRAY,
        fontsize=8,
    )
    return figure, axes


def _finish(figure):
    figure.tight_layout(rect=(0.04, 0.055, 1, 0.88))
    return figure


def _grid(axis):
    axis.grid(axis="y", color=LIGHT)
    axis.set_axisbelow(True)


def weekly_plot(data, end_day=296):
    """Conversión y volumen por bloques relativos de siete días; no guarda archivos."""
    initial = _initial(data, end_day)
    last_day = int(initial.dia_visita.max())
    weekly = (
        initial.assign(bloque=(initial.dia_visita - 1) // 7)
        .groupby("bloque")
        .agg(
            visitas=("incidencia_compra", "size"), compras=("incidencia_compra", "sum")
        )
    )
    weekly = weekly.reindex(range((last_day - 1) // 7 + 1), fill_value=0).reset_index()
    weekly["conversion"] = weekly.compras / weekly.visitas.replace(0, np.nan)
    weekly["dia_inicio"] = weekly.bloque * 7 + 1
    weekly["dia_fin"] = np.minimum(weekly.dia_inicio + 6, last_day)
    weekly["dias_del_bloque"] = weekly.dia_fin - weekly.dia_inicio + 1
    weekly["periodo"] = weekly.dia_inicio.astype(str) + "–" + weekly.dia_fin.astype(str)
    figure, axes = _canvas(
        "La conversión cambia entre bloques de visitas",
        f"{_period(initial)} · Nivel: visita · Bloques de siete días, no semanas de calendario",
    )
    axes[0].plot(weekly.bloque + 1, weekly.conversion, color=BLUE)
    axes[0].set(title="Compras / visitas por bloque", ylabel="Conversión por visita")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].bar(weekly.bloque + 1, weekly.visitas, color=BLUE)
    axes[1].set(title="Número de visitas por bloque", ylabel="Visitas")
    selected = weekly.iloc[:: max(1, len(weekly) // 7)].index.union(
        weekly.tail(1).index
    )
    for axis in axes:
        axis.set_xticks(
            weekly.loc[selected, "bloque"] + 1,
            weekly.loc[selected, "periodo"],
            rotation=35,
        )
        axis.set_xlabel("Días de cada bloque; último bloque puede ser parcial")
        _grid(axis)
    return _finish(figure), {"weekly": weekly}


def customer_plot(data, end_day=296):
    """Distribuye visitas y conversión por cliente, no por registro de visita."""
    initial = _initial(data, end_day)
    customers = initial.groupby("id").incidencia_compra.agg(
        visitas="size", compras="sum"
    )
    customers["conversion"] = customers.compras / customers.visitas
    summary = customers.describe().rename_axis("estadistico").reset_index()
    figure, axes = _canvas(
        "Los clientes presentan distinta frecuencia y conversión",
        f"{_period(initial)} · Nivel: cliente · Cada observación resume una de {len(customers):,} personas",
    )
    axes[0].hist(customers.visitas, bins=20, color=BLUE, edgecolor="white")
    axes[0].set(
        title="Visitas acumuladas de cada cliente",
        xlabel="Visitas por cliente",
        ylabel="Clientes",
    )
    axes[1].hist(
        customers.conversion, bins=np.linspace(0, 1, 21), color=TEAL, edgecolor="white"
    )
    axes[1].set(
        title="Compras / visitas de cada cliente",
        xlabel="Conversión individual",
        ylabel="Clientes",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        _grid(axis)
    return _finish(figure), {"customers": customers, "customer_summary": summary}


def history_plot(data, end_day=296):
    """Cruza historial anterior por visita; devuelve tasas y soporte de cada celda."""
    initial = _initial(data, end_day)
    events, _ = features(initial)
    recency = pd.cut(
        events.dias_desde_compra,
        [0, 7, 30, 60, np.inf],
        labels=["1–7 días", "8–30 días", "31–60 días", "Más de 60 días"],
    )
    recency = recency.cat.add_categories("Sin compra previa").fillna(
        "Sin compra previa"
    )
    recency = recency.cat.reorder_categories(
        ["Sin compra previa", "1–7 días", "8–30 días", "31–60 días", "Más de 60 días"]
    )
    frequency = pd.cut(
        events.visitas_previas,
        [-1, 5, 15, 30, np.inf],
        labels=["0–5", "6–15", "16–30", "31 o más"],
    )
    table = (
        events.assign(recencia=recency, frecuencia=frequency)
        .groupby(["recencia", "frecuencia"], observed=False)
        .incidencia_compra.agg(visitas="size", compras="sum", conversion="mean")
        .reset_index()
    )
    rates = table.pivot(index="recencia", columns="frecuencia", values="conversion")
    support = table.pivot(index="recencia", columns="frecuencia", values="visitas")
    figure, axes = _canvas(
        "El historial distingue contextos de visita con distinta conversión",
        f"{_period(initial)} · Nivel: visita · Un cliente puede ocupar distintas celdas a lo largo del tiempo",
        figsize=(14, 5.8),
    )
    sns.heatmap(
        rates,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        mask=support.eq(0),
        vmin=0,
        vmax=1,
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
        axis.set_xlabel("Número de visitas anteriores del cliente")
        axis.set_ylabel("Tiempo desde la última compra anterior")
        axis.tick_params(axis="y", labelrotation=0)
        axis.tick_params(axis="x", labelrotation=0)
    return _finish(figure), {"history": table}


def association_plot(data, end_day=296):
    """Asociaciones por visita con denominadores; no estima efectos comerciales."""
    initial = _initial(data, end_day)
    working = initial.assign(
        promocion=initial[PROMOS]
        .max(axis=1)
        .map({0: "Ninguna marca", 1: "Alguna marca"}),
        edad_grupo=pd.cut(
            initial.edad,
            [0, 29, 39, 49, 59, np.inf],
            labels=["Hasta 29", "30–39", "40–49", "50–59", "60 o más"],
        ),
    )
    figure, axes = _canvas(
        "Ofertas y perfil describen asociaciones, no efectos causales",
        f"{_period(initial)} · Nivel: visita · Interpretar tasas junto a la cobertura de cada grupo",
    )
    tables = {}
    for axis, column, name in zip(
        axes, ["promocion", "edad_grupo"], ["promotions", "age"], strict=True
    ):
        table = working.groupby(column, observed=True).agg(
            visitas=("incidencia_compra", "size"),
            compras=("incidencia_compra", "sum"),
            conversion=("incidencia_compra", "mean"),
            clientes=("id", "nunique"),
        )
        tables[name] = table.reset_index()
        axis.bar(
            np.arange(len(table)),
            table.conversion,
            color=BLUE if column == "promocion" else TEAL,
        )
        axis.set_xticks(np.arange(len(table)), table.index.astype(str), rotation=15)
        axis.set_ylim(0, max(0.45, float(table.conversion.max()) + 0.1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylabel("Compras / visitas del grupo")
        for position, row in enumerate(table.itertuples()):
            axis.text(
                position,
                row.conversion + 0.012,
                f"{row.conversion:.1%}\nn={row.visitas:,} visitas",
                ha="center",
                fontsize=9,
            )
    axes[0].set_title("Promoción disponible en el día de la visita")
    axes[1].set_title("Edad del cliente que realiza la visita")
    return _finish(figure), tables


def quality_plot(data, end_day=296):
    """Audita calidad y significado empírico del historial usando data.audit."""
    initial = _initial(data, end_day)
    quality, definitions = audit(initial)
    profile = pd.DataFrame(
        {
            "variable": initial.columns,
            "faltantes": initial.isna().sum().to_numpy(),
            "valores_distintos": initial.nunique().to_numpy(),
        }
    )
    figure, axes = _canvas(
        "La calidad y el significado de los campos se comprueban antes de modelar",
        f"{_period(initial)} · Auditoría de la base recibida · El nombre de un campo no demuestra su significado",
        figsize=(14, 6.5),
    )
    axes[0].axis("off")
    labels = {
        "filas": "Visitas",
        "clientes": "Clientes únicos",
        "columnas": "Columnas originales",
        "faltantes": "Celdas faltantes",
        "duplicados": "Filas duplicadas",
        "incoherencia_cantidad": "Incoherencias de cantidad",
        "incoherencia_marca": "Incoherencias de marca",
    }
    table = axes[0].table(
        cellText=[[labels[key], f"{value:,}"] for key, value in quality.items()],
        colLabels=["Comprobación", "Resultado"],
        loc="center",
        cellLoc="left",
        colWidths=[0.75, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(NAVY if row == 0 else (LIGHT if row % 2 else "white"))
        cell.set_text_props(color="white" if row == 0 else NAVY)
    names = {
        "ultima_marca_comprada": "Marca original",
        "ultima_cantidad_comprada": "Cantidad original",
    }
    hypotheses = {
        "visita_anterior": "visita anterior",
        "compra_anterior": "última compra real",
        "evento_actual": "resultado actual",
        "incidencia_visita_anterior": "compra sí/no anterior",
    }
    bar_labels = [
        f"{names[row.variable]}: {hypotheses[row.hipotesis]}"
        for row in definitions.itertuples()
    ]
    axes[1].barh(np.arange(len(definitions)), definitions.coincidencia, color=BLUE)
    axes[1].set_yticks(np.arange(len(definitions)), bar_labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set(
        xlim=(0, 1.14),
        title="Coincidencia con distintas interpretaciones",
        xlabel="Proporción de visitas coincidentes",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    for position, value in enumerate(definitions.coincidencia):
        axes[1].text(value + 0.015, position, f"{value:.1%}", va="center", fontsize=9)
    return _finish(figure), {
        "quality": quality.rename("valor").rename_axis("comprobacion").reset_index(),
        "history_definitions": definitions,
        "columns": profile,
    }
