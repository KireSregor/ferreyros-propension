"""Hipótesis de historial y calendario; todos los resultados se desplazan una visita."""

import numpy as np
import pandas as pd

from .data import PRICES, PROMOS, features


def reviewed_features(data):
    """Version review_v1: customer history plus strictly prior store-day signals."""
    events, predictors = monthly_trend_features(data)
    groups = events.groupby("id", sort=False)
    extra = {}
    calendar = pd.Index(range(1, int(events.dia_visita.max()) + 1), name="dia_visita")
    daily = events.groupby("dia_visita").incidencia_compra.agg(["sum", "count"])
    daily = daily.reindex(calendar, fill_value=0)
    for window in [7, 30, 60]:
        purchases = daily["sum"].shift().rolling(window, min_periods=1).sum()
        visits = daily["count"].shift().rolling(window, min_periods=1).sum()
        rate = (purchases + 1) / (visits + 4)
        extra[f"tienda_tasa_{window}"] = events.dia_visita.map(rate)
    extra["tienda_cambio_tasa"] = extra["tienda_tasa_7"] - extra["tienda_tasa_60"]
    for brand in range(1, 6):
        column = f"precio_marca_{brand}"
        median = events.groupby("dia_visita")[column].first().reindex(calendar)
        median = median.shift().rolling(60, min_periods=5).median()
        extra[f"precio_vs_anterior60_{brand}"] = events[column] / events.dia_visita.map(
            median
        )
    exposure = events[PROMOS].max(axis=1)
    for value in [0, 1]:
        exposed = exposure.eq(value).astype(float)
        hits = exposed * events.incidencia_compra.fillna(0)
        count = exposed.groupby(events.id).cumsum() - exposed
        purchases = hits.groupby(events.id).cumsum() - hits
        extra[f"tasa_exposicion_promo_{value}"] = (
            purchases + 4 * events.tasa_previa
        ) / (count + 4)
        extra[f"soporte_exposicion_promo_{value}"] = count
    extra["asociacion_promo_previa"] = (
        extra["tasa_exposicion_promo_1"] - extra["tasa_exposicion_promo_0"]
    )
    extra["ritmo_visitas_reciente"] = events.visitas_30 / (
        (events.visitas_90 - events.visitas_30) / 2 + 1
    )
    prior = groups.incidencia_compra.shift().fillna(0)
    extra["compra_previa_visita"] = prior
    for value, name in [(1, "compras"), (0, "no_compras")]:
        run = prior.eq(value).astype(int)
        extra[f"racha_{name}"] = run.groupby(events.id).transform(
            lambda s: s.groupby(s.eq(0).cumsum()).cumsum()
        )
    excluded = [
        c
        for c in predictors
        if c.startswith("fase_") or c in {"cliente_conocido", "dia_relativo"}
    ]
    predictors = predictors.drop(columns=excluded)
    return events, pd.concat(
        [predictors, pd.DataFrame(extra, index=events.index)], axis=1
    ).replace([np.inf, -np.inf], np.nan)


def monthly_trend_features(data):
    """Add disjoint 30-day histories before each visit, excluding today's outcome.

    These are relative windows, not calendar months. Empty windows remain missing;
    support columns distinguish sparse estimates from reliable changes.
    """
    events, predictors = enriched_features(data)
    extra = {}
    counts = np.zeros((len(events), 4))
    purchases = np.zeros_like(counts)
    for _, group in events.groupby("id", sort=False):
        days = group.dia_visita.to_numpy()
        cumulative = np.r_[0, group.incidencia_compra.fillna(0).cumsum()]
        for block in range(4):
            left = np.searchsorted(days, days - 30 * (block + 1), side="left")
            right = np.searchsorted(days, days - 30 * block, side="left")
            counts[group.index, block] = right - left
            purchases[group.index, block] = cumulative[right] - cumulative[left]
    rates = np.divide(
        purchases, counts, out=np.full_like(counts, np.nan), where=counts > 0
    )
    for block in range(4):
        label = f"{block * 30 + 1}_{(block + 1) * 30}"
        extra[f"month_visits_{label}"] = counts[:, block]
        extra[f"month_purchases_{label}"] = purchases[:, block]
        extra[f"month_rate_{label}"] = rates[:, block]
    extra["month_rate_change"] = rates[:, 0] - rates[:, 1]
    extra["month_rate_acceleration"] = rates[:, 0] - 2 * rates[:, 1] + rates[:, 2]
    extra["month_visit_change"] = counts[:, 0] - counts[:, 1]
    extra["month_purchase_change"] = purchases[:, 0] - purchases[:, 1]
    extra["month_change_support"] = np.minimum(counts[:, 0], counts[:, 1])
    # Shrink toward the preceding 90 days, rather than lifetime behaviour.
    older_count = counts[:, 1:].sum(axis=1)
    older_purchases = purchases[:, 1:].sum(axis=1)
    prior = (older_purchases + 1) / (older_count + 4)
    extra["month_recent_smoothed"] = (purchases[:, 0] + 5 * prior) / (counts[:, 0] + 5)
    extra["month_recent_vs_prior90"] = extra["month_recent_smoothed"] - prior
    return events, pd.concat(
        [predictors, pd.DataFrame(extra, index=events.index)], axis=1
    )


def enriched_features(data):
    """Extiende variables base sin usar resultados actuales ni futuros.

    Los ciclos de 7/30 días son fases relativas al inicio, no fechas inventadas.
    ID se usa como categoría de clientes conocidos, nunca como magnitud ordinal.
    """
    events, predictors = features(data)
    groups = events.groupby("id", sort=False)
    target = events.incidencia_compra
    previous = groups.incidencia_compra.shift()
    extra = pd.DataFrame(index=events.index)
    extra["cliente_conocido"] = events.id.astype(str)
    extra["dia_relativo"] = events.dia_visita
    extra["antiguedad_observada"] = events.dia_visita - groups.dia_visita.transform(
        "first"
    )
    for period in [7, 30]:
        extra[f"fase_{period}_sin"] = np.sin(2 * np.pi * events.dia_visita / period)
        extra[f"fase_{period}_cos"] = np.cos(2 * np.pi * events.dia_visita / period)
    for lag in [1, 2, 3]:
        extra[f"compra_visita_lag_{lag}"] = groups.incidencia_compra.shift(lag)
    for window in [3, 5, 10, 20]:
        extra[f"tasa_ultimas_{window}_visitas"] = previous.groupby(events.id).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    extra["tasa_ewm_5"] = previous.groupby(events.id).transform(
        lambda s: s.ewm(span=5, adjust=False).mean()
    )
    extra["tasa_ewm_20"] = previous.groupby(events.id).transform(
        lambda s: s.ewm(span=20, adjust=False).mean()
    )
    gaps = groups.dia_visita.diff()
    mean_gap = gaps.groupby(events.id).transform(lambda s: s.expanding().mean().shift())
    extra["intervalo_visita_medio_previo"] = mean_gap
    extra["intervalo_visita_relativo"] = gaps / mean_gap.replace(0, np.nan)
    for window in [7, 14]:
        visits = np.zeros(len(events))
        purchases = np.zeros(len(events))
        for _, sub in groups:
            days = sub.dia_visita.to_numpy()
            left = np.searchsorted(days, days - window)
            pos = np.arange(len(sub))
            cumulative = np.r_[0, sub.incidencia_compra.fillna(0).cumsum()]
            visits[sub.index] = pos - left
            purchases[sub.index] = cumulative[pos] - cumulative[left]
        extra[f"visitas_{window}_dias"] = visits
        extra[f"tasa_{window}_dias"] = (purchases + 1) / (visits + 4)
    purchase_day = events.dia_visita.where(target.eq(1))
    # Compute purchase-to-purchase intervals only at actual purchase rows.
    prior_day = purchase_day.groupby(events.id).ffill().groupby(events.id).shift()
    interval = (events.dia_visita - prior_day).where(target.eq(1))
    mean_interval = interval.groupby(events.id).transform(
        lambda s: s.expanding().mean().shift()
    )
    extra["intervalo_compra_medio_previo"] = mean_interval
    extra["recencia_compra_relativa"] = (
        events.dias_desde_compra / mean_interval.replace(0, np.nan)
    )
    extra["unidades_por_dia_desde_compra"] = events.cantidad_previa / (
        events.dias_desde_compra + 1
    )
    extra["visitas_desde_compra"] = (
        events.visitas_previas
        - events.visitas_previas.where(target.eq(1))
        .groupby(events.id)
        .transform(lambda s: s.ffill().shift())
    )
    preference = events[[f"preferencia_{i}" for i in range(1, 6)]].fillna(0).to_numpy()
    extra["precio_segun_preferencia"] = (preference * events[PRICES].to_numpy()).sum(
        axis=1
    )
    extra["promo_segun_preferencia"] = (preference * events[PROMOS].to_numpy()).sum(
        axis=1
    )
    extra["precio_promedio"] = events[PRICES].mean(axis=1)
    # No total-day conversion or current brand/quantity is admitted here.
    return events, pd.concat([predictors, extra], axis=1)
