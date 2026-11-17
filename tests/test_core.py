from pathlib import Path

import numpy as np
import pandas as pd

from ferreyros_propension.data import features, load_data, score_visits

ROOT = Path(__file__).resolve().parents[1]


def test_search_stopping_respects_budget_and_meaningful_improvement():
    import optuna

    from ferreyros_propension.modeling import search_has_plateaued

    settings = {"enabled": True, "min_trials": 4, "patience": 2, "min_delta": 0.001}

    def trials(values):
        return [optuna.trial.create_trial(value=value) for value in values]

    assert not search_has_plateaued(trials([0.5, 0.5, 0.5]), settings)
    assert search_has_plateaued(trials([0.5, 0.51, 0.5101, 0.5102]), settings)
    assert not search_has_plateaued(trials([0.5, 0.51, 0.5101, 0.52]), settings)
    assert not search_has_plateaued(trials([0.5] * 10), {"enabled": False})


def test_parallel_cv_matches_sequential():
    from ferreyros_propension.modeling import cross_validate

    raw = load_data(ROOT / "Dataset/compras_data.csv", end_day=620)
    events, predictors = features(raw.loc[raw.id.isin(raw.id.unique()[:25])])
    columns = predictors[["edad", "tasa_previa", "visitas_previas"]]
    config = {"seed": 42, "threads": 1}
    parameters = {"n_estimators": 10, "max_depth": 3}
    folds = [[100, 150], [150, 200]]
    serial_scores, serial_predictions = cross_validate(
        "random_forest", parameters, events, columns, folds, config
    )
    parallel_scores, parallel_predictions = cross_validate(
        "random_forest",
        parameters,
        events,
        columns,
        folds,
        {**config, "fold_workers": 2},
    )
    pd.testing.assert_frame_equal(serial_scores, parallel_scores)
    pd.testing.assert_frame_equal(serial_predictions, parallel_predictions)


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
    raw = load_data(ROOT / "Dataset/compras_data.csv", end_day=620)
    early = raw[raw.dia_visita.le(200)]
    d, x = features(raw)
    e, z = features(early)
    pd.testing.assert_frame_equal(x[d.dia_visita.le(200)].reset_index(drop=True), z)


def test_current_target_cannot_change_current_features():
    raw = load_data(ROOT / "Dataset/compras_data.csv", end_day=620).iloc[:30].copy()
    d, x = features(raw)
    changed = raw.copy()
    changed.loc[10, ["incidencia_compra", "cantidad", "id_marca"]] = [1, 99, 5]
    _, z = features(changed)
    pd.testing.assert_series_equal(x.iloc[10], z.iloc[10])


def test_calendar_windows_and_first_visit():
    raw = load_data(ROOT / "Dataset/compras_data.csv", end_day=620).iloc[:3].copy()
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
    raw = load_data(ROOT / "Dataset/compras_data.csv", end_day=620)
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
