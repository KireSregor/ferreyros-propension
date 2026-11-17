"""Mide CV completo en paralelo y comprueba equivalencia de predicciones."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from ferreyros_propension.data import features, load_data
from ferreyros_propension.modeling import cross_validate


def main():
    events, predictors = features(load_data("Dataset/compras_data.csv"))
    config = yaml.safe_load(Path("config.yml").read_text())
    parameters = json.loads(Path("outputs/best_params.json").read_text())
    rows = []
    for family in ["random_forest", "catboost"]:
        reference = None
        for workers, threads in [(1, 4), (3, 2), (3, 4)]:
            candidate = {**config, "fold_workers": workers, "threads": threads}
            started = perf_counter()
            with threadpool_limits(limits=threads):
                scores, predictions = cross_validate(
                    family,
                    parameters[family],
                    events,
                    predictors,
                    config["folds"],
                    candidate,
                )
            elapsed = perf_counter() - started
            if reference is None:
                reference = predictions.p.to_numpy()
            difference = float(np.max(np.abs(predictions.p.to_numpy() - reference)))
            np.testing.assert_allclose(predictions.p, reference, atol=1e-6, rtol=1e-6)
            rows.append(
                {
                    "model": family,
                    "fold_workers": workers,
                    "threads": threads,
                    "seconds": elapsed,
                    "ap": scores.ap.mean(),
                    "max_prediction_delta": difference,
                }
            )
            pd.DataFrame(rows).to_csv(
                "outputs/tables/parallel_benchmark.csv", index=False
            )
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
