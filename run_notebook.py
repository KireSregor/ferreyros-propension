"""Execute the complete notebook with this interpreter and a clean kernel."""

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class ProgressNotebookClient(NotebookClient):
    """Show kernel progress while retaining the same outputs in the notebook."""

    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="lightgbm")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("output-name must be a directory name, not a path")
    root = Path(__file__).resolve().parent
    os.environ["PROPENSION_RUN"] = args.output_name
    spec = root / ".tmp/jupyter/kernels/lightgbm-python"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "LightGBM project",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = str(root / ".tmp/jupyter")
    source = root / "notebooks/propension_lightgbm.ipynb"
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def completed(cell, cell_index, **kwargs):
        print(f"{args.output_name}: celda {cell_index} ejecutada", flush=True)

    client = ProgressNotebookClient(
        notebook,
        kernel_name="lightgbm-python",
        timeout=None,
        resources={"metadata": {"path": str(root)}},
        on_cell_executed=completed,
    )
    target = (
        source
        if args.output_name == "lightgbm"
        else (root / "outputs" / args.output_name / "notebook.executed.ipynb")
    )
    try:
        client.execute()
    finally:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
    print(f"Notebook ejecutado: {target}", flush=True)


if __name__ == "__main__":
    main()
