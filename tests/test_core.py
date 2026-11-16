from pathlib import Path

import numpy as np
import pandas as pd

from ferreyros_propension.data import features, load_data, score_visits

ROOT = Path(__file__).resolve().parents[1]


def test_model_name_must_be_explicit():
    import optuna
    import pytest

    from ferreyros_propension.modeling import make_model, suggest_parameters

    with pytest.raises(ValueError, match="Modelo no soportado"):
        suggest_parameters(optuna.trial.FixedTrial({}), "typo")
    with pytest.raises(ValueError, match="Modelo no soportado"):
        make_model("typo", {}, ["edad"])


def test_constant_ranker_uses_random_selection_expectation():
    from ferreyros_propension.evaluation import metrics

    target = np.array([0, 0, 0, 1, 1])
    result = metrics(target, np.full(len(target), 0.4))
    assert result["lift_20"] == 1.0
    assert result["recall_20"] == 0.2


def test_future_invariance():
    raw = load_data(ROOT / "Dataset/compras_data.csv")
    early = raw[raw.dia_visita.le(200)]
    d, x = features(raw)
    e, z = features(early)
    pd.testing.assert_frame_equal(x[d.dia_visita.le(200)].reset_index(drop=True), z)


def test_current_target_cannot_change_current_features():
    raw = load_data(ROOT / "Dataset/compras_data.csv").iloc[:30].copy()
    d, x = features(raw)
    changed = raw.copy()
    changed.loc[10, ["incidencia_compra", "cantidad", "id_marca"]] = [1, 99, 5]
    _, z = features(changed)
    pd.testing.assert_series_equal(x.iloc[10], z.iloc[10])


def test_calendar_windows_and_first_visit():
    raw = load_data(ROOT / "Dataset/compras_data.csv").iloc[:3].copy()
    raw["dia_visita"] = [1, 31, 32]
    raw["incidencia_compra"] = [1, 0, 1]
    raw["cantidad"] = [2, 0, 1]
    raw["id_marca"] = [1, 0, 2]
    d, x = features(raw)
    assert list(d.visitas_30) == [0, 1, 1]
    assert list(d.compras_30) == [0, 1, 0]
    assert d.compras_previas.iloc[0] == 0
    assert np.isnan(d.dias_desde_compra.iloc[0])


def test_score_uses_same_features_and_rejects_future():
    raw = load_data(ROOT / "Dataset/compras_data.csv")
    day = 100
    history = raw[raw.dia_visita.lt(day)]
    visits = raw[raw.dia_visita.eq(day)]

    class Dummy:
        def predict_proba(self, x):
            p = x.tasa_previa.to_numpy()
            return np.c_[1 - p, p]

    d, x = features(raw)
    bundle = {"model": Dummy(), "columns": list(x), "version": "test"}
    result = score_visits(bundle, history, visits)
    np.testing.assert_allclose(
        result.probabilidad, d.loc[d.dia_visita.eq(day), "tasa_previa"]
    )
    import pytest

    with pytest.raises(ValueError):
        score_visits(bundle, raw, visits)
