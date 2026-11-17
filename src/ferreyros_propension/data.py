"""Auditoría y variables disponibles antes de cada visita, por calendario."""

from pathlib import Path

import numpy as np
import pandas as pd

PROFILE = [
    "genero",
    "estado_civil",
    "edad",
    "nivel_educacion",
    "ingreso_anual",
    "ocupacion",
]
PRICES = [f"precio_marca_{i}" for i in range(1, 6)]
PROMOS = [f"promo_marca_{i}" for i in range(1, 6)]
CATEGORICAL = ["genero", "estado_civil", "nivel_educacion", "ocupacion", "marca_previa"]


def load_data(path: str | Path, end_day: int | None = None) -> pd.DataFrame:
    """Lee sin modificar el origen; la clave cliente/día debe ser inequívoca."""
    if end_day is None:
        data = pd.read_csv(path)
    else:
        data = pd.concat(
            chunk.loc[chunk.dia_visita.le(end_day)]
            for chunk in pd.read_csv(path, chunksize=10000)
        )
    data = data.sort_values(["id", "dia_visita"]).reset_index(drop=True)
    if data.duplicated(["id", "dia_visita"]).any():
        raise ValueError("Hay visitas múltiples por cliente/día: falta orden intradía.")
    if not data.incidencia_compra.isin([0, 1]).all():
        raise ValueError("Target inválido.")
    if data[PRICES].isna().any().any() or (data[PRICES] <= 0).any().any():
        raise ValueError("Los precios deben ser positivos y estar disponibles.")
    if not data[PROMOS].isin([0, 1]).all().all():
        raise ValueError("Los indicadores de promoción deben ser binarios.")
    return data


def audit(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Describe calidad y contrasta posibles significados del historial recibido."""
    summary = {
        "filas": len(data),
        "clientes": data.id.nunique(),
        "columnas": len(data.columns),
        "faltantes": int(data.isna().sum().sum()),
        "duplicados": int(data.duplicated().sum()),
        "incoherencia_cantidad": int(
            ((data.cantidad > 0) != data.incidencia_compra.eq(1)).sum()
        ),
        "incoherencia_marca": int(
            ((data.id_marca > 0) != data.incidencia_compra.eq(1)).sum()
        ),
    }
    comparisons = []
    for given, current in [
        ("ultima_marca_comprada", "id_marca"),
        ("ultima_cantidad_comprada", "cantidad"),
    ]:
        previous = data.groupby("id")[current].shift().fillna(0)
        purchase = (
            data[current]
            .where(data.incidencia_compra.eq(1))
            .groupby(data.id)
            .transform(lambda x: x.ffill().shift())
            .fillna(0)
        )
        for name, values in [
            ("visita_anterior", previous),
            ("compra_anterior", purchase),
            ("evento_actual", data[current]),
        ]:
            comparisons.append(
                {
                    "variable": given,
                    "hipotesis": name,
                    "coincidencia": data[given].eq(values).mean(),
                }
            )
    previous_incidence = data.groupby("id").incidencia_compra.shift().fillna(0)
    comparisons.append(
        {
            "variable": "ultima_cantidad_comprada",
            "hipotesis": "incidencia_visita_anterior",
            "coincidencia": data.ultima_cantidad_comprada.eq(previous_incidence).mean(),
        }
    )
    return pd.Series(summary), pd.DataFrame(comparisons)


def features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula historial estrictamente anterior. Admite targets NaN para scoring.

    Las filas sin resultado no actualizan compras; las visitas sí cuentan como
    observadas. El scoring público exige un único día futuro por lote.
    """
    events = data.sort_values(["id", "dia_visita"]).reset_index(drop=True).copy()
    customer_groups = events.groupby("id", sort=False)
    purchase_indicator = events.incidencia_compra.fillna(0)
    events["visitas_previas"] = customer_groups.cumcount()
    events["compras_previas"] = (
        purchase_indicator.groupby(events.id).cumsum() - purchase_indicator
    )
    # Prior fijo explícito; no se estima con el futuro.
    events["tasa_previa"] = (events.compras_previas + 1) / (events.visitas_previas + 4)
    events["dias_desde_visita"] = events.dia_visita - customer_groups.dia_visita.shift()
    for source, dest in [
        ("dia_visita", "dia_compra_previa"),
        ("id_marca", "marca_previa"),
        ("cantidad", "cantidad_previa"),
    ]:
        events[dest] = (
            events[source]
            .where(purchase_indicator.eq(1))
            .groupby(events.id)
            .transform(lambda x: x.ffill().shift())
        )
    events["dias_desde_compra"] = events.dia_visita - events.dia_compra_previa
    quantities = events.cantidad.fillna(0)
    events["cantidad_media_previa"] = (
        quantities.groupby(events.id).cumsum() - quantities
    ) / events.compras_previas.replace(0, np.nan)
    for window in [30, 60, 90]:
        visits = np.zeros(len(events))
        purchases = np.zeros(len(events))
        for _, sub in events.groupby("id", sort=False):
            days = sub.dia_visita.to_numpy()
            vals = sub.incidencia_compra.fillna(0).to_numpy()
            left = np.searchsorted(days, days - window, side="left")
            positions = np.arange(len(sub))
            cumulative_purchases = np.r_[0, vals.cumsum()]
            visits[sub.index] = positions - left
            purchases[sub.index] = (
                cumulative_purchases[positions] - cumulative_purchases[left]
            )
        events[f"visitas_{window}"] = visits
        events[f"compras_{window}"] = purchases
        events[f"tasa_{window}"] = (purchases + 1) / (visits + 4)
    for brand in range(1, 6):
        bought = (events.id_marca.eq(brand) & purchase_indicator.eq(1)).astype(int)
        events[f"preferencia_{brand}"] = (
            bought.groupby(events.id).cumsum() - bought
        ) / events.compras_previas.replace(0, np.nan)
    events["precio_min"] = events[PRICES].min(axis=1)
    events["rango_precio"] = events[PRICES].max(axis=1) - events.precio_min
    events["n_promociones"] = events[PROMOS].sum(axis=1)
    for i in range(1, 6):
        events[f"precio_relativo_{i}"] = events[f"precio_marca_{i}"] / events.precio_min
    exclude = {
        "id",
        "dia_visita",
        "incidencia_compra",
        "id_marca",
        "cantidad",
        "ultima_marca_comprada",
        "ultima_cantidad_comprada",
        "tamanio_ciudad",
        "dia_compra_previa",
    }
    cols = [c for c in events if c not in exclude]
    predictors = events[cols].copy()
    for col in CATEGORICAL:
        predictors[col] = predictors[col].fillna(-1).astype(int).astype(str)
    return events, predictors


def score_visits(
    bundle: dict, history: pd.DataFrame, visits: pd.DataFrame
) -> pd.DataFrame:
    """Puntúa un día de visitas sin resultado con historia estrictamente previa."""
    if visits.empty or visits.dia_visita.nunique() != 1:
        raise ValueError("El lote debe contener un único día de scoring.")
    required_columns = {"id", "dia_visita", *PROFILE, *PRICES, *PROMOS}
    missing_columns = required_columns.difference(visits.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas de scoring: {sorted(missing_columns)}")
    if visits[PRICES].isna().any().any() or (visits[PRICES] <= 0).any().any():
        raise ValueError("Los precios de scoring deben ser positivos y conocidos.")
    if not visits[PROMOS].isin([0, 1]).all().all():
        raise ValueError("Las promociones de scoring deben ser binarias.")
    if not history.empty and history.dia_visita.max() >= visits.dia_visita.min():
        raise ValueError("El historial debe ser anterior al lote.")
    pending_visits = visits.copy()
    for col in ["incidencia_compra", "id_marca", "cantidad"]:
        pending_visits[col] = np.nan
    joined = pd.concat([history, pending_visits], ignore_index=True)
    if joined.duplicated(["id", "dia_visita"]).any():
        raise ValueError("Clave cliente/día duplicada.")
    events, predictors = features(joined)
    mask = events.dia_visita.eq(visits.dia_visita.iloc[0])
    probabilities = bundle["model"].predict_proba(
        predictors.loc[mask, bundle["columns"]]
    )[:, 1]
    if bundle.get("calibrator") is not None:
        probabilities = bundle["calibrator"].predict_proba(
            np.log(
                np.clip(probabilities, 1e-6, 1 - 1e-6)
                / (1 - np.clip(probabilities, 1e-6, 1 - 1e-6))
            ).reshape(-1, 1)
        )[:, 1]
    result = events.loc[mask, ["id", "dia_visita"]].copy()
    result["probabilidad"] = probabilities
    result["version"] = bundle["version"]
    return result
