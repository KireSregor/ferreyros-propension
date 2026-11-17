"""Run the approved development-only audit and temporal experiments."""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import pandas as pd
import psutil

from ferreyros_propension.data import audit
from ferreyros_propension.review import (
    development_data,
    feature_sets,
    fingerprint,
    read_config,
    run_classic_task,
)


def audit_development(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    data = development_data(config)
    summary, comparisons = audit(data)
    summary.to_csv(out / "data_quality.csv", header=["value"])
    comparisons.to_csv(out / "history_semantics.csv", index=False)
    events, features, groups = feature_sets(data)
    feature_table = pd.DataFrame(
        {
            "variable": features.columns,
            "missing_fraction": features.isna().mean().to_numpy(),
            "unique": features.nunique().to_numpy(),
        }
    )
    feature_table.to_csv(out / "feature_audit.csv", index=False)
    splits = []
    for cut, end in config["outer_folds"]:
        ranges = [("train", 1, cut), ("validation", cut + 1, end)]
        for i, (inner_cut, stop) in enumerate(config["inner_folds"][cut], 1):
            ranges.extend(
                [
                    (f"inner_{i}_train", 1, inner_cut),
                    (f"inner_{i}_stop", inner_cut - 39, inner_cut),
                    (f"inner_{i}_validation", inner_cut + 1, stop),
                ]
            )
        for label, start, stop in ranges:
            part = data.loc[data.dia_visita.between(start, stop)]
            splits.append(
                {
                    "outer_cut": cut,
                    "split": label,
                    "start": start,
                    "end": stop,
                    "visits": len(part),
                    "customers": part.id.nunique(),
                    "purchase_rate": part.incidencia_compra.mean(),
                }
            )
    pd.DataFrame(splits).to_csv(out / "splits.csv", index=False)
    daily = events.groupby("dia_visita").incidencia_compra.agg(["size", "sum", "mean"])
    daily.to_csv(out / "daily_development.csv")
    customer = events.groupby("id").incidencia_compra.agg(["size", "sum", "mean"])
    customer.describe().to_csv(out / "customer_summary.csv")
    manifest = {
        "fingerprint": fingerprint(data, config),
        "development_end": int(data.dia_visita.max()),
        "rows": len(data),
        "customers": int(data.id.nunique()),
        "features": groups,
        "ram_gb": psutil.virtual_memory().total / 2**30,
        "cpu_logical": os.cpu_count(),
        "test_accessed": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"AUDIT: {len(data)} development visits, {len(features.columns)} predictors",
        flush=True,
    )


def run_classical(config):
    started = perf_counter()
    out = Path(config["output"])
    rows, times = [], []
    tasks = [
        (family, cut, end)
        for cut, end in config["outer_folds"]
        for family in [
            "historical",
            "constant",
            "reference",
            "logistic",
            "lightgbm",
            "xgboost",
            "catboost",
        ]
    ]
    replay = all(
        (out / f"{family}_{cut}" / "result.json").exists() for family, cut, _ in tasks
    )
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {
            pool.submit(
                run_classic_task,
                family,
                cut,
                end,
                config,
                config["search"]["stage_timeout_seconds"] / 3,
            ): (family, cut)
            for family, cut, end in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["metrics"])
            times.append(
                {
                    key: result[key]
                    for key in ["model", "cut", "trials", "seconds", "iterations"]
                }
            )
            pd.DataFrame(rows).to_csv(out / "classical_metrics.csv", index=False)
            pd.DataFrame(times).to_csv(out / "classical_times.csv", index=False)
            validation = result["metrics"][-1]
            print(
                f"DONE {result['model']} {result['cut']}: {result['trials']} trials, "
                f"{result['seconds']:.1f}s; P={validation['precision']:.4f}, R={validation['recall']:.4f}",
                flush=True,
            )
    if replay:
        print(
            "Reused existing classical studies; original training times preserved",
            flush=True,
        )
        return
    (out / "classical_runtime.json").write_text(
        json.dumps(
            {
                "wall_seconds": perf_counter() - started,
                "workers": config["workers"],
                "threads_per_model": config["threads_per_model"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "classical",
            "neural",
            "neural-resume",
            "wide",
            "extras",
            "blend",
            "calibrate",
            "neural-seeds",
            "leader",
            "report",
        ],
    )
    parser.add_argument("--config", default="review_config.yml")
    parser.add_argument("--new-budget", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["development_end"] != 620:
        raise ValueError(
            "This review reserves days 621 onward; development_end must remain 620"
        )
    for cut, end in config["outer_folds"]:
        if not 40 < cut < end <= 620:
            raise ValueError("Invalid exterior temporal fold")
        if any(
            not 40 < start < stop <= cut for start, stop in config["inner_folds"][cut]
        ):
            raise ValueError(
                "An internal fold extends beyond its exterior training period"
            )
    output = Path(config["output"])
    clock_path = output / "training_clock.json"
    if args.stage not in {"audit", "report", "leader"} and (
        args.new_budget or not clock_path.exists()
    ):
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        clock_path.write_text(
            json.dumps(
                {
                    "started_epoch": started,
                    "deadline_epoch": started + config["training_budget_seconds"],
                    "budget_seconds": config["training_budget_seconds"],
                },
                indent=2,
            )
        )
    if args.stage == "audit":
        audit_development(config)
    elif args.stage == "classical":
        run_classical(config)
    elif args.stage == "neural":
        from ferreyros_propension.review_neural import _identity, run_neural

        markers = [
            output / f"neural_{cut}" / "result.json" for cut, _ in config["outer_folds"]
        ]
        if all(path.exists() for path in markers):
            expected = _identity(development_data(config), config)
            if any(
                json.loads(path.read_text())["fingerprint"] != expected
                for path in markers
            ):
                raise ValueError("Cached neural studies have another fingerprint")
            print(
                "Reused neural selections; use neural-resume to complete pending trials"
            )
            return
        run_neural(config)
    elif args.stage == "neural-resume":
        from ferreyros_propension.review_neural import _identity, run_neural

        output = Path(config["output"])
        clock = json.loads(clock_path.read_text())
        pending = []
        for cut, _ in config["outer_folds"]:
            marker = output / f"neural_{cut}" / "result.json"
            if (
                not marker.exists()
                or json.loads(marker.read_text())["complete_trials"]
                < config["trials"]["neural"]
            ):
                pending.append(cut)
        if (
            pending
            and clock["deadline_epoch"] - time.time()
            < config["search"]["neural_timeout_seconds"]
        ):
            raise RuntimeError("Insufficient global budget to resume the neural stage")
        if not pending:
            print(
                "Initial neural trial targets already completed; training times preserved"
            )
            return
        expected = _identity(development_data(config), config)
        for cut, _ in config["outer_folds"]:
            directory = output / f"neural_{cut}"
            marker = directory / "result.json"
            if not marker.exists():
                continue
            result = json.loads(marker.read_text())
            if result["fingerprint"] != expected:
                raise ValueError(
                    "Cannot resume neural results with another fingerprint"
                )
            if result["complete_trials"] >= config["trials"]["neural"]:
                continue
            history = directory / "initial_selection"
            history.mkdir(exist_ok=True)
            if not (history / "selection.json").exists():
                shutil.copyfile(marker, history / "selection.json")
                for source in directory.glob("outer_*"):
                    shutil.copyfile(source, history / source.name)
                shutil.copyfile(directory / "model.joblib", history / "model.joblib")
            marker.unlink()
        runtime = output / "neural_runtime.json"
        if runtime.exists() and not (output / "neural_initial_runtime.json").exists():
            shutil.copyfile(runtime, output / "neural_initial_runtime.json")
        run_neural(config)
    elif args.stage == "wide":
        from ferreyros_propension.review_neural_wide import _identity, run_wide

        markers = [
            output / f"neural_wide_{cut}" / "result.json"
            for cut, _ in config["outer_folds"]
        ]
        if all(path.exists() for path in markers):
            expected = _identity(development_data(config), config)
            if any(
                json.loads(path.read_text())["fingerprint"] != expected
                for path in markers
            ):
                raise ValueError("Cached wide studies have another fingerprint")
            print("Reused neural capacity study; original training times preserved")
            return
        run_wide(config)
    elif args.stage == "extras":
        from ferreyros_propension.review_diagnostics import (
            ablate_features,
            explain_candidates,
        )
        from ferreyros_propension.review_ensembles import (
            causal_rate_adjustment,
            chronological_ensembles,
            seed_stability,
        )

        seed_stability(config)
        runtime = output / "ensembles_runtime.json"
        ready = all(
            (output / f"{name}_{cut}" / filename).exists()
            for cut, _ in config["outer_folds"]
            for name in [
                "chronological_equal",
                "chronological_weighted",
                "chronological_stack",
            ]
            for filename in ["result.json", "model.joblib"]
        )
        original_runtime = runtime.read_bytes() if ready and runtime.exists() else None
        chronological_ensembles(config)
        if original_runtime is not None:
            runtime.write_bytes(original_runtime)
        causal_rate_adjustment(config)
        ablate_features(config, max_seconds=900)
        explain_candidates(config, max_seconds=120)
    elif args.stage == "blend":
        from ferreyros_propension.review_blend import blend_neural_trees

        blend_neural_trees(config)
    elif args.stage == "calibrate":
        from ferreyros_propension.review_calibration import run_calibration

        run_calibration(config)
    elif args.stage == "neural-seeds":
        from ferreyros_propension.review_diagnostics import neural_seed_sensitivity

        clock = json.loads(clock_path.read_text())
        remaining = clock["deadline_epoch"] - time.time()
        if remaining <= 180:
            print("Neural seed diagnostic skipped: global training budget exhausted")
            return
        neural_seed_sensitivity(config, max_seconds=min(300, remaining - 120))
    elif args.stage == "leader":
        from ferreyros_propension.review_diagnostics import explain_leader

        explain_leader(config, max_seconds=120)
    else:
        from ferreyros_propension.review_reporting import write_report

        print(write_report(config).to_string(index=False))


if __name__ == "__main__":
    main()
