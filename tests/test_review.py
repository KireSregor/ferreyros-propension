"""Synthetic controls for the reviewed development-only temporal workflow."""

import numpy as np
import pandas as pd
import pytest

from ferreyros_propension.data import PRICES, PROMOS, load_data
from ferreyros_propension.feature_research import reviewed_features


@pytest.fixture
def visits():
    rows = []
    for customer in [1, 2]:
        for index, day in enumerate([1, 5, 10, 32, 60, 75]):
            purchase = (index + customer) % 2
            row = {
                "id": customer,
                "dia_visita": day,
                "incidencia_compra": purchase,
                "id_marca": customer if purchase else 0,
                "cantidad": 2 if purchase else 0,
                "genero": customer % 2,
                "estado_civil": customer % 2,
                "edad": 25 + customer,
                "nivel_educacion": 1,
                "ingreso_anual": 10000 + 1000 * customer,
                "ocupacion": 1,
            }
            row.update({column: 10 + brand for brand, column in enumerate(PRICES)})
            row.update({column: int(index % 3 == 0) for column in PROMOS})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["id", "dia_visita"]).reset_index(drop=True)


def test_development_loader_excludes_invalid_future_before_validation(tmp_path, visits):
    invalid = visits.iloc[[0]].copy()
    invalid["dia_visita"] = 621
    invalid["incidencia_compra"] = 9
    invalid[PRICES] = -1
    invalid[PROMOS] = 7
    path = tmp_path / "visits.csv"
    pd.concat([visits, invalid, invalid]).to_csv(path, index=False)
    actual = load_data(path, end_day=620)
    pd.testing.assert_frame_equal(actual, visits)
    with pytest.raises(ValueError):
        load_data(path)


def test_reviewed_features_ignore_current_and_future_outcomes(visits):
    events, original = reviewed_features(visits)
    changed = visits.copy()
    future = changed.dia_visita.ge(32)
    changed.loc[future, "incidencia_compra"] = (
        1 - changed.loc[future, "incidencia_compra"]
    )
    changed.loc[future, "id_marca"] = 5
    changed.loc[future, "cantidad"] = 9
    _, altered = reviewed_features(changed)
    mask = events.dia_visita.le(32)
    pd.testing.assert_frame_equal(original.loc[mask], altered.loc[mask])


def test_reviewed_features_match_truncated_history(visits):
    events, complete = reviewed_features(visits)
    _, truncated = reviewed_features(visits.loc[visits.dia_visita.le(32)])
    pd.testing.assert_frame_equal(
        complete.loc[events.dia_visita.le(32)].reset_index(drop=True), truncated
    )


def test_store_features_do_not_use_same_day_outcomes(visits):
    events, features = reviewed_features(visits)
    same_day = events.dia_visita.eq(32)
    assert features.loc[same_day, "tienda_tasa_30"].nunique() == 1
    assert np.isfinite(features.loc[same_day, "tienda_tasa_30"]).all()


def test_neural_preprocessing_unknown_category_and_artifact_replay():
    import torch

    from ferreyros_propension.neural_training import fit_neural

    train = pd.DataFrame(
        {"edad": [20.0, 30.0, np.nan, 50.0], "genero": ["A", "B", "A", "B"]}
    )
    pending = pd.DataFrame({"edad": [1000.0, np.nan], "genero": ["UNSEEN", "A"]})
    config = {
        "architecture": "mlp",
        "width": 8,
        "depth": 1,
        "dropout": 0,
        "fixed_epochs": 2,
        "device": "cpu",
        "categorical_columns": ["genero"],
        "return_artifact": True,
    }
    result = fit_neural(train, [0, 1, 0, 1], None, None, {"pending": pending}, config)
    artifact = result["artifact"]
    transformed = artifact["preprocessing"].transform(pending).astype("float32")
    with torch.no_grad():
        replay = (
            artifact["model"](torch.as_tensor(transformed))
            .sigmoid()
            .squeeze(-1)
            .numpy()
        )
    np.testing.assert_allclose(result["probabilities"]["pending"], replay)
    assert result["executed_epochs"] == 2
    assert np.isfinite(replay).all()
    scaler = (
        artifact["preprocessing"]
        .named_transformers_["numeric"]
        .named_steps["standardscaler"]
    )
    assert scaler.mean_[0] < 100


def test_threshold_uses_equal_fold_weight_instead_of_pooled_rows():
    from sklearn.metrics import precision_recall_curve

    from ferreyros_propension.review import optimal_threshold

    target = np.array(
        [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    )
    probability = np.array(
        [
            0.1,
            0.5,
            0.5,
            0.3,
            0.3,
            0.9,
            0.5,
            0.9,
            0.9,
            0.3,
            0.3,
            0.9,
            0.7,
            0.3,
            0.7,
            0.5,
            0.9,
            0.3,
            0.9,
            0.9,
            0.1,
            0.1,
            0.5,
            0.1,
        ]
    )
    frame = pd.DataFrame({"y": target, "p": probability})
    threshold, score, _ = optimal_threshold([frame.iloc[:4], frame.iloc[4:]])
    precision, recall, thresholds = precision_recall_curve(target, probability)
    pooled = thresholds[np.argmax(np.minimum(precision[:-1], recall[:-1]))]
    assert threshold == pytest.approx(0.3)
    assert score == pytest.approx(0.5980392156862745)
    assert pooled == pytest.approx(0.5)


def test_classic_preprocessing_keeps_training_statistics_and_unknown_categories():
    from ferreyros_propension.review import preprocessing

    train = pd.DataFrame(
        {"edad": [20.0, 30.0, np.nan, 50.0], "genero": ["A", "B", "A", "B"]}
    )
    future = pd.DataFrame({"edad": [1000.0, np.nan], "genero": ["UNSEEN", "A"]})
    transformer = preprocessing(train, "logistic")
    transformer.fit(train)
    transformed = transformer.transform(future)
    assert transformed.shape[0] == len(future)
    assert np.isfinite(transformed).all()
    numeric = transformer.named_transformers_["numeric"]
    assert numeric.named_steps["standardscaler"].mean_[0] < 100
    np.testing.assert_array_equal(
        transformer.named_transformers_["category"].categories_[0], ["A", "B"]
    )


def test_review_scoring_replays_history_and_checks_schema(visits):
    from ferreyros_propension.review import score_review_visits

    columns = ["tasa_previa", "tienda_tasa_30"]

    class HistoryModel:
        def predict_proba(self, frame):
            assert list(frame.columns) == columns
            probability = frame.mean(axis=1).to_numpy()
            return np.column_stack([1 - probability, probability])

    bundle = {
        "feature_builder_version": "review_v1",
        "columns": columns,
        "threshold": 0.3,
        "model": HistoryModel(),
    }
    events, features = reviewed_features(visits)
    scoring = visits.loc[visits.dia_visita.eq(32)]
    history = visits.loc[visits.dia_visita.lt(32)]
    result = score_review_visits(bundle, history, scoring)
    expected = HistoryModel().predict_proba(
        features.loc[events.dia_visita.eq(32), columns]
    )[:, 1]
    np.testing.assert_allclose(result.probability, expected)
    np.testing.assert_array_equal(result.prediction, expected >= 0.3)
    with pytest.raises(ValueError, match="version"):
        score_review_visits(
            {**bundle, "feature_builder_version": "old"}, history, scoring
        )
    with pytest.raises(ValueError, match="schema"):
        score_review_visits({**bundle, "columns": columns + columns}, history, scoring)
    with pytest.raises(ValueError, match="precede"):
        score_review_visits(bundle, visits, scoring)


@pytest.mark.parametrize(
    "family", ["logistic", "lightgbm", "xgboost", "catboost", "reference"]
)
def test_classic_fit_returns_usable_fitted_model(family, visits):
    from ferreyros_propension.review import fit_classic

    events, all_features = reviewed_features(visits)
    features = all_features[["edad", "genero", "tasa_previa"]]
    params = {
        "half_life": 180,
        "C": 0.1,
        "learning_rate": 0.05,
        "depth": 2,
        "min_samples": 2,
        "reg_lambda": 2,
        "reg_alpha": 0.1,
        "colsample": 0.9,
        "subsample": 0.9,
        "random_strength": 0.5,
    }
    config = {
        "threads_per_model": 1,
        "early_stopping": {"days": 10, "rounds": 2, "max_trees": 10},
    }
    model, selected = fit_classic(events, features, family, params, 60, config)
    pending = features.loc[events.dia_visita.eq(75)].copy()
    pending["genero"] = "UNSEEN"
    probability = model.predict_proba(pending)[:, 1]
    assert probability.shape == (2,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    if family not in {"logistic", "reference"}:
        assert 1 <= selected <= 10


def test_ensemble_alignment_preserves_source_order_and_rejects_mismatch():
    from ferreyros_propension.review_ensembles import aligned

    first = pd.DataFrame(
        {"id": [1, 2], "dia_visita": [10, 10], "y": [0, 1], "p": [0.2, 0.8]}
    )
    second = first.copy()
    second["p"] = [0.3, 0.7]
    np.testing.assert_allclose(aligned([first, second]), [[0.2, 0.3], [0.8, 0.7]])
    with pytest.raises(ValueError, match="aligned"):
        aligned([first, second.iloc[::-1]])
    duplicate = pd.concat([second.iloc[[0]], second.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        aligned([first, duplicate])
    wrong_labels = second.copy()
    wrong_labels["y"] = 1 - wrong_labels.y
    with pytest.raises(ValueError, match="aligned"):
        aligned([first, wrong_labels])


def test_averaged_ensemble_respects_column_contract_and_weights():
    from ferreyros_propension.review_ensembles import AveragedPredictor, ColumnPredictor

    class ColumnModel:
        def __init__(self, name):
            self.name = name

        def predict_proba(self, frame):
            assert list(frame) == [self.name]
            probability = frame[self.name].to_numpy()
            return np.column_stack([1 - probability, probability])

    frame = pd.DataFrame({"a": [0.2, 0.8], "b": [0.6, 0.4], "unused": [99, 99]})
    models = [
        ColumnPredictor(ColumnModel("a"), ["a"]),
        ColumnPredictor(ColumnModel("b"), ["b"]),
    ]
    result = AveragedPredictor(models, [0.25, 0.75]).predict_proba(frame)
    np.testing.assert_allclose(result[:, 1], [0.5, 0.5])
    np.testing.assert_allclose(result.sum(axis=1), 1)
