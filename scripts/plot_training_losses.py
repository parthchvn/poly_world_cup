#!/usr/bin/env python3
"""Plot saved World Cup losses during or after training; no model/GPU imports.

python scripts/plot_training_losses.py --run-dir /workspace/runs/YOUR_RUN
Add --watch 10 to refresh loss_curve.png every ten seconds (Ctrl-C to stop).
Matplotlib is needed only for plotting, not for training or loss logging.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
import warnings


LOSS_KEYS = ("loss", "eval_loss", "smoke_eval_loss")


def read_rows(path):
    """Read complete records only, so a concurrent writer is safe to follow."""
    with Path(path).open(encoding="utf-8") as source:
        lines = source.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    if Path(path).suffix == ".csv":
        return list(csv.DictReader(io.StringIO("".join(lines))))
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            rows.append(row)
        except ValueError as error:
            warnings.warn(f"Skipping invalid metrics record {path}:{number}: {error}")
    return rows


def loss_series(rows):
    """Keep latest values per step and remove abandoned tails on resume.

    'train_loss' is a whole-run summary, not a step loss. Never plot it.
    Legacy metrics.jsonl files without train_begin markers are also supported.
    """
    series = {key: {} for key in LOSS_KEYS}
    nonfinite = 0
    for row in rows:
        try:
            step = float(row["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(step) or step < 0 or not step.is_integer():
            continue
        step = int(step)
        if row.get("event") == "train_begin":
            for key in LOSS_KEYS:
                series[key] = {s: value for s, value in series[key].items() if s <= step}
            continue
        for key in LOSS_KEYS:
            if row.get(key) in (None, ""):
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                nonfinite += 1
                value = math.nan  # Break the curve instead of suggesting a finite loss.
            series[key][step] = value
    if nonfinite:
        warnings.warn(f"{nonfinite} nonfinite loss value(s) in the log; shown as gaps in the curve")
    return {key: sorted(values.items()) for key, values in series.items()}


def plot_losses(source, output, title=None):
    series = loss_series(read_rows(source))
    if not any(math.isfinite(value) for points in series.values() for _, value in points):
        raise ValueError(f"No finite loss values yet in {source}")
    try:
        import matplotlib
        matplotlib.use("Agg")  # Works on headless RunPod machines.
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib: python3 -m pip install matplotlib") from error

    figure, axes = plt.subplots(figsize=(10, 6))
    styles = {
        "loss": {"label": "Training loss", "color": "#2563eb", "linewidth": 1.3},
        "eval_loss": {"label": "Validation loss (full split)", "color": "#ea580c", "marker": "o"},
        "smoke_eval_loss": {"label": "Smoke validation (subset)", "color": "#7c3aed",
                            "marker": "D", "linestyle": "none"},
    }
    try:
        for key, points in series.items():
            if points:
                steps, values = zip(*points)
                axes.plot(steps, values, **styles[key])
        axes.set(xlabel="Optimizer step", ylabel="Loss (assistant-token cross-entropy)",
                 title=title or Path(source).parent.name)
        axes.grid(alpha=0.2)
        axes.legend()
        figure.tight_layout()
        output = Path(output)
        if output.resolve() == Path(source).resolve():
            raise ValueError("Plot output must not overwrite the loss log")
        if output.suffix.lower() not in (".png", ".pdf", ".svg"):
            raise ValueError("Plot output must end with .png, .pdf or .svg")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as target:
            temporary = Path(target.name)
        try:
            figure.savefig(temporary, dpi=160)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        plt.close(figure)
    return {key: len(points) for key, points in series.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="The trainer's --out directory")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/loss_curve.png")
    parser.add_argument("--title")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS", help="Refresh periodically; 0 plots once")
    args = parser.parse_args(argv)
    if not math.isfinite(args.watch) or args.watch < 0:
        parser.error("--watch must be a finite nonnegative number")
    output = args.output or args.run_dir / "loss_curve.png"
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        parser.error("--output must end with .png, .pdf or .svg")
    try:
        while True:
            source = args.run_dir / "metrics.jsonl"
            if not source.exists() and (args.run_dir / "losses.csv").exists():
                source = args.run_dir / "losses.csv"
            if output.resolve() in {(args.run_dir / name).resolve() for name in ("metrics.jsonl", "losses.csv")}:
                parser.error("--output must not overwrite a loss log")
            try:
                counts = plot_losses(source, output, title=args.title)
                print(f"Saved {output} ({counts})", flush=True)
            except (FileNotFoundError, ValueError) as error:
                if not args.watch:
                    parser.error(str(error))
                print(f"Waiting for losses: {error}", flush=True)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("Stopped watching; the last saved plot is retained.")
    except RuntimeError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
