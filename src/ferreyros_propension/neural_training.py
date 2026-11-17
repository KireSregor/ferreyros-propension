"""Neural tabular training with train-only transforms and auditable stopping."""

from collections.abc import Mapping
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn

from .data import CATEGORICAL


def _preprocessor(frame, config, seed):
    categorical = config.get("categorical_columns")
    if categorical is None:
        categorical = [
            column
            for column in frame
            if column in CATEGORICAL or not pd.api.types.is_numeric_dtype(frame[column])
        ]
    numeric = [column for column in frame if column not in categorical]
    scaling = (
        QuantileTransformer(
            n_quantiles=min(1000, len(frame)),
            output_distribution="normal",
            random_state=seed,
        )
        if config.get("quantile_transform", False)
        else StandardScaler()
    )
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                    scaling,
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def _model(feature_count, config):
    width = int(config.get("width", 128))
    depth = int(config.get("depth", 2))
    dropout = float(config.get("dropout", 0.2))
    architecture = config.get("architecture", "tabm")
    if architecture == "tabm":
        from tabm import TabM

        return TabM.make(
            n_num_features=feature_count,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=depth,
            d_block=width,
            dropout=dropout,
            k=int(config.get("k", 16)),
        )
    if architecture != "mlp":
        raise ValueError(f"Unsupported neural architecture: {architecture}")
    layers = []
    previous = feature_count
    for _ in range(depth):
        layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _logits(model, batch, architecture):
    output = model(batch, None) if architecture == "tabm" else model(batch)
    return output.squeeze(-1) if architecture == "tabm" else output


@torch.no_grad()
def _predict(model, values, architecture, batch_size):
    model.eval()
    predictions = []
    for batch in values.split(batch_size):
        predictions.append(_logits(model, batch, architecture).sigmoid().mean(dim=1))
    return torch.cat(predictions).cpu().numpy() if predictions else np.empty(0)


def _binary_loss(target, probability):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return float(
        -np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability))
    )


def fit_neural(train_X, train_y, stop_X, stop_y, predict_frames, config, seed=42):
    """Fit MLP or TabM; return probabilities and per-epoch loss diagnostics.

    ``predict_frames`` is a mapping of names to frames with the training columns.
    Without ``fixed_epochs``, the returned model restores the state with the best
    stopping-set log loss. With it, train exactly that many epochs and do not use
    the stopping set for selection; the caller can pass ``None`` for both stop
    arguments during a full-training refit. Call this function again on the full
    training window with ``fixed_epochs=result['best_epochs']`` to perform refit.

    Configuration: architecture, width, depth, dropout, k, learning_rate,
    weight_decay, batch_size, max_epochs, min_epochs, patience, min_delta,
    quantile_transform, categorical_columns, device, precision (32 or bf16),
    fixed_epochs, threads, return_artifact, and optional sample_weight aligned to
    train_X. ``return_artifact=True`` adds the fitted model on CPU, preprocessing,
    input column order and architecture to the result for subsequent scoring.
    """
    started = perf_counter()
    if not isinstance(predict_frames, Mapping):
        raise TypeError("predict_frames must map names to data frames")
    fixed_epochs = config.get("fixed_epochs")
    if fixed_epochs is None and (stop_X is None or stop_y is None or len(stop_X) == 0):
        raise ValueError("Early stopping requires a nonempty temporal stopping set")
    torch.set_num_threads(int(config.get("threads", 2)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    preprocessing = _preprocessor(train_X, config, seed)
    train_array = preprocessing.fit_transform(train_X).astype(np.float32)
    train_tensor = torch.as_tensor(train_array, device=device)

    def transform(frame):
        array = preprocessing.transform(frame[train_X.columns]).astype(np.float32)
        return torch.as_tensor(array, device=device)

    stop_tensor = transform(stop_X) if stop_X is not None and len(stop_X) else None
    targets = np.asarray(train_y, dtype=np.float32)
    target_tensor = torch.as_tensor(targets, device=device)
    weight_tensor = torch.as_tensor(
        np.asarray(
            config.get("sample_weight", np.ones(len(targets))), dtype=np.float32
        ),
        device=device,
    )
    weight_tensor = weight_tensor / weight_tensor.mean()
    architecture = config.get("architecture", "tabm")
    model = _model(train_tensor.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    batch_size = int(config.get("batch_size", 512))
    max_epochs = int(fixed_epochs or config.get("max_epochs", 300))
    min_epochs = int(config.get("min_epochs", 10))
    patience = int(config.get("patience", 25))
    min_delta = float(config.get("min_delta", 0.0001))
    use_bf16 = device.type == "cuda" and config.get("precision", "32") == "bf16"
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    curves = []
    stop_reason = "fixed_epochs" if fixed_epochs is not None else "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        gradient_norms = []
        for indices in torch.randperm(len(targets), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with context:
                logits = _logits(model, train_tensor[indices], architecture)
                losses = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    target_tensor[indices, None].expand_as(logits),
                    reduction="none",
                ).mean(dim=1)
                loss = (losses * weight_tensor[indices]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite neural training loss at epoch {epoch}"
                )
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            gradient_norms.append(gradient_norm.detach())
            optimizer.step()
        train_probability = _predict(model, train_tensor, architecture, 2048)
        train_loss = _binary_loss(targets, train_probability)
        validation_loss = None
        if stop_tensor is not None:
            stop_probability = _predict(model, stop_tensor, architecture, 2048)
            validation_loss = _binary_loss(np.asarray(stop_y), stop_probability)
        if not np.isfinite(train_loss) or (
            validation_loss is not None and not np.isfinite(validation_loss)
        ):
            raise FloatingPointError(
                f"Nonfinite neural evaluation loss at epoch {epoch}"
            )
        curves.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "seconds": perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_mean": float(torch.stack(gradient_norms).mean()),
                "gradient_norm_max": float(torch.stack(gradient_norms).max()),
            }
        )
        if fixed_epochs is not None:
            best_epoch = epoch
            continue
        if validation_loss < best_loss - min_delta:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            stop_reason = "early_stopping"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    probabilities = {
        name: _predict(model, transform(frame), architecture, 2048)
        for name, frame in predict_frames.items()
    }
    result = {
        "probabilities": probabilities,
        "best_epochs": best_epoch,
        "executed_epochs": len(curves),
        "curves": curves,
        "stop_reason": stop_reason,
        "seconds": perf_counter() - started,
        "device": str(device),
        "transformed_features": train_tensor.shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    if config.get("return_artifact", False):
        result["artifact"] = {
            "model": model.cpu(),
            "preprocessing": preprocessing,
            "columns": list(train_X.columns),
            "architecture": architecture,
        }
    return result
