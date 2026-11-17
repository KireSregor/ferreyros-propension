"""Compare feature ablations with thresholds frozen before later periods."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from ferreyros_propension.data import PRICES, PROFILE, PROMOS, load_data
from ferreyros_propension.feature_research import monthly_trend_features


def metrics(target, probability, threshold):
    prediction = probability >= threshold
    return {
        "precision": precision_score(target, prediction, zero_division=0),
        "recall": recall_score(target, prediction, zero_division=0),
        "f1": f1_score(target, prediction, zero_division=0),
        "auc": roc_auc_score(target, probability),
        "ap": average_precision_score(target, probability),
        "logloss": log_loss(target, probability),
    }


def main():
    started = perf_counter()
    output = Path("outputs/last_attempt")
    output.mkdir(exist_ok=True)
    events, features = monthly_trend_features(load_data("Dataset/compras_data.csv"))
    target = events.incidencia_compra
    history = [column for column in features if column not in PRICES + PROMOS + PROFILE
               and not column.startswith("fase_")
               and column not in {"cliente_conocido", "dia_relativo"}]
    groups = {
        "history": history,
        "history_offers": history + PRICES + PROMOS,
        "history_profile": history + PROFILE,
        "history_offers_profile": history + PRICES + PROMOS + PROFILE,
    }
    # Category dictionaries contain predictor values only, never outcomes.
    for column in features.select_dtypes(include=["object", "string"]):
        features[column] = features[column].astype("category")
    results = []
    predictions = []
    for name, columns in {"baseline": [], **groups}.items():
        for leaves in ([0] if name == "baseline" else [3, 7, 15]):
            threshold = None
            candidate = f"{name}_leaves{leaves}"
            for cut, end in [(296, 404), (404, 512), (512, 620), (620, 730)]:
                train = events.dia_visita.between(1, cut)
                valid = events.dia_visita.between(cut + 1, end)
                if name == "baseline":
                    train_probability = features.loc[train, "tasa_previa"].to_numpy()
                    probability = features.loc[valid, "tasa_previa"].to_numpy()
                else:
                    model = LGBMClassifier(
                        n_estimators=400, learning_rate=0.025, num_leaves=leaves,
                        max_depth=4, min_child_samples=300, reg_lambda=100,
                        reg_alpha=5, verbosity=-1, n_jobs=4, random_state=42,
                    )
                    weights = np.exp2((events.loc[train, "dia_visita"] - cut) / 180)
                    model.fit(features.loc[train, columns], target[train],
                              sample_weight=weights)
                    train_probability = model.predict_proba(features.loc[train, columns])[:, 1]
                    probability = model.predict_proba(features.loc[valid, columns])[:, 1]
                if threshold is None:
                    precision, recall, thresholds = precision_recall_curve(target[valid], probability)
                    threshold = float(thresholds[np.argmax(np.minimum(precision[:-1], recall[:-1]))])
                for split, mask, values in [("train", train, train_probability),
                                             ("validation", valid, probability)]:
                    results.append({"candidate": candidate, "cut": cut, "end": end,
                                    "split": split, "threshold": threshold,
                                    "n": int(mask.sum()), **metrics(target[mask], values, threshold)})
                predictions.append(pd.DataFrame({"candidate": candidate, "cut": cut,
                                                  "id": events.loc[valid, "id"],
                                                  "day": events.loc[valid, "dia_visita"],
                                                  "y": target[valid], "p": probability}))
            pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
            print(f"Finished {candidate}: {perf_counter() - started:.1f}s", flush=True)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    print(f"TOTAL_SECONDS={perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
