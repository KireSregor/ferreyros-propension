"""Temporal TabM benchmark with internally chosen epochs and refitting."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabm import TabM

from competitive_benchmark import (
    INNER_FOLDS,
    OUTER_FOLDS,
    OUTPUT,
    load_features,
    threshold_for,
)
from final_audit import metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 1024


def tensors(train, other):
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    scaler = StandardScaler()
    train_array = scaler.fit_transform(imputer.fit_transform(train)).astype("float32")
    other_array = scaler.transform(imputer.transform(other)).astype("float32")
    return (
        torch.as_tensor(train_array, device=DEVICE),
        torch.as_tensor(other_array, device=DEVICE),
    )


def new_model(features):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return TabM.make(
        n_num_features=features,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=2,
        d_block=128,
        dropout=0.2,
        k=8,
    ).to(DEVICE)


def run_epoch(model, optimizer, x, y, weights):
    model.train()
    order = torch.randperm(len(y), device=DEVICE)
    for indices in order.split(BATCH_SIZE):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[indices], None).squeeze(-1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y[indices, None].expand_as(logits), reduction="none"
        ).mean(dim=1)
        (losses * weights[indices]).mean().backward()
        optimizer.step()


@torch.no_grad()
def predict(model, x):
    model.eval()
    parts = []
    for batch in x.split(4096):
        parts.append(model(batch, None).squeeze(-1).sigmoid().mean(dim=1))
    return torch.cat(parts).cpu().numpy()


def fit_fold(events, features, cut, end):
    train = events.dia_visita.le(cut)
    inner = events.dia_visita.le(cut - 40)
    stop = events.dia_visita.between(cut - 39, cut)
    valid = events.dia_visita.between(cut + 1, end)
    target = events.incidencia_compra

    def make_y(mask):
        return torch.as_tensor(target[mask].to_numpy(dtype="float32"), device=DEVICE)

    def make_weights(mask, end_day):
        weights = np.exp2((events.loc[mask, "dia_visita"].to_numpy() - end_day) / 180)
        return torch.as_tensor(weights.astype("float32"), device=DEVICE)

    x_train, x_stop = tensors(features[inner], features[stop])
    model = new_model(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, best_epoch, stagnant = float("inf"), 1, 0
    stop_y = target[stop].to_numpy()
    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(
            model, optimizer, x_train, make_y(inner), make_weights(inner, cut - 40)
        )
        probability = np.clip(predict(model, x_stop), 1e-6, 1 - 1e-6)
        loss = -np.mean(
            stop_y * np.log(probability) + (1 - stop_y) * np.log(1 - probability)
        )
        if loss < best_loss - 0.0001:
            best_loss, best_epoch, stagnant = loss, epoch, 0
        else:
            stagnant += 1
        if stagnant >= PATIENCE:
            break

    x_train, x_valid = tensors(features[train], features[valid])
    model = new_model(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    train_y, weights = make_y(train), make_weights(train, cut)
    for _ in range(best_epoch):
        run_epoch(model, optimizer, x_train, train_y, weights)
    return predict(model, x_train), predict(model, x_valid), best_epoch


def main():
    started = perf_counter()
    torch.set_num_threads(2)
    events, features = load_features()
    target = events.incidencia_compra
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    details = []
    for cut, end in OUTER_FOLDS:
        inner_frames = []
        for inner_cut, inner_end in INNER_FOLDS[cut]:
            _, probability, epoch = fit_fold(events, features, inner_cut, inner_end)
            mask = events.dia_visita.between(inner_cut + 1, inner_end)
            frame = events.loc[mask, ["id", "dia_visita", "incidencia_compra"]].copy()
            frame["p"] = probability
            inner_frames.append(frame)
            details.append({"cut": cut, "inner_cut": inner_cut, "epochs": epoch})
            print(f"TabM fold {cut}: inner {inner_cut}, epochs {epoch}", flush=True)
        inner = pd.concat(inner_frames)
        threshold, inner_score = threshold_for(inner.incidencia_compra, inner.p)
        train_p, valid_p, epoch = fit_fold(events, features, cut, end)
        train = events.dia_visita.le(cut)
        valid = events.dia_visita.between(cut + 1, end)
        for split, mask, p in [
            ("train", train, train_p),
            ("validation", valid, valid_p),
        ]:
            results.append(
                {
                    "model": "tabm",
                    "cut": cut,
                    "end": end,
                    "split": split,
                    "threshold": threshold,
                    "n": int(mask.sum()),
                    **metrics(target[mask], p, threshold),
                }
            )
        inner.to_csv(OUTPUT / f"tabm_{cut}_inner.csv", index=False)
        frame = events.loc[valid, ["id", "dia_visita", "incidencia_compra"]].copy()
        frame["p"] = valid_p
        frame.to_csv(OUTPUT / f"tabm_{cut}_validation.csv", index=False)
        details.append(
            {
                "cut": cut,
                "epochs": epoch,
                "threshold": threshold,
                "inner_score": inner_score,
            }
        )
        print(f"TabM outer {cut}: epochs {epoch}", flush=True)
    pd.DataFrame(results).to_csv(OUTPUT / "tabm_metrics.csv", index=False)
    (OUTPUT / "tabm_configuration.json").write_text(
        json.dumps(
            {
                "folds": details,
                "seconds": perf_counter() - started,
                "device": str(DEVICE),
            },
            indent=2,
        )
    )
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print(f"TABM_SECONDS={perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
