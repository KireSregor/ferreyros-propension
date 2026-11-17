"""Ejecuta la configuración vigente y actualiza el notebook de resultados."""

import json
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

from ferreyros_propension.modeling import run_experiments


def main():
    root = Path.cwd()
    runtime_path = root / "outputs/runtime.json"
    previous_runtime = (
        json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    )
    started = perf_counter()
    run_experiments(root)
    experiment_seconds = perf_counter() - started
    runtime = {
        "experiment_seconds": experiment_seconds,
        "previous_experiment_seconds": previous_runtime.get("experiment_seconds")
        if previous_runtime
        else None,
        "note": "Incluye búsqueda, ajuste final y diagnósticos; excluye benchmark de velocidad y notebook.",
    }
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    seed_path = root / "outputs/tables/seed_sensitivity.csv"
    seed_path.unlink(missing_ok=True)
    path = root / "notebooks/02_modelado_resultados.ipynb"
    notebook = nbformat.read(path, 4)
    notebook_started = perf_counter()
    notebook = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, path)
    runtime["notebook_seconds"] = perf_counter() - notebook_started
    runtime["total_seconds"] = perf_counter() - started
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print("ACTUALIZACION COMPLETA", json.dumps(runtime), flush=True)


if __name__ == "__main__":
    main()
