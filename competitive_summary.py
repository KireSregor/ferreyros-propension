"""Add historical baseline and the five-model ensemble to the comparison."""

import pandas as pd

from competitive_benchmark import (
    INNER_FOLDS,
    OUTER_FOLDS,
    OUTPUT,
    add_ensemble,
    load_features,
    threshold_for,
)
from final_audit import metrics


def main():
    events, features = load_features()
    target = events.incidencia_compra
    probability = features.tasa_previa
    rows = []
    families = ["logistic", "lightgbm", "xgboost", "catboost", "tabm"]
    for cut, end in OUTER_FOLDS:
        ensemble = add_ensemble(cut, end, families)
        ensemble["model"] = "ensemble_5"
        rows.append(ensemble)
        inner = pd.concat(
            [
                events.loc[events.dia_visita.between(start + 1, stop)]
                for start, stop in INNER_FOLDS[cut]
            ]
        )
        threshold, _ = threshold_for(inner.incidencia_compra, inner.tasa_previa)
        for label, mask in [
            ("train", events.dia_visita.le(cut)),
            ("validation", events.dia_visita.between(cut + 1, end)),
        ]:
            rows.append(
                {
                    "model": "baseline",
                    "cut": cut,
                    "end": end,
                    "split": label,
                    "n": int(mask.sum()),
                    "threshold": threshold,
                    **metrics(target[mask], probability[mask], threshold),
                }
            )
    additional = pd.DataFrame(rows)
    additional.to_csv(OUTPUT / "additional_metrics.csv", index=False)
    combined = pd.concat(
        [
            pd.read_csv(OUTPUT / "metrics.csv"),
            pd.read_csv(OUTPUT / "tabm_metrics.csv"),
            additional,
        ]
    )
    combined.to_csv(OUTPUT / "comparison.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
