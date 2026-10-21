"""Plot scalar metrics from a TensorBoard event file (install ssgen[plot])."""

import argparse
import math
from pathlib import Path
import sys


def read_scalars(path):
    """Read all scalar samples, including scalar TensorProto summaries."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    series = {}
    for tag in accumulator.Tags().get("scalars", []):
        series[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    for tag in accumulator.Tags().get("tensors", []):
        values = []
        for event in accumulator.Tensors(tag):
            array = tensor_util.make_ndarray(event.tensor_proto)
            if array.size == 1 and array.dtype.kind in "biuf":
                values.append((event.step, float(array.reshape(-1)[0])))
        if values and tag not in series:
            series[tag] = values
    return {tag: sorted((step, value) for step, value in rows if math.isfinite(value))
            for tag, rows in series.items() if any(math.isfinite(v) for _, v in rows)}


def rolling_mean(rows, window):
    """Trailing mean; emit only complete windows."""
    total = 0.0
    result = []
    for index, (step, value) in enumerate(rows):
        total += value
        if index >= window:
            total -= rows[index - window][1]
        if index >= window - 1:
            result.append((step, total / window))
    return result


def plot_log(path, window=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    series = read_scalars(path)
    if not series:
        raise ValueError("No finite scalar metrics found in the event file.")
    preferred = ["loss/average", "loss/current", "lr/unet", "loss/epoch_average"]
    tags = sorted(series, key=lambda tag: (preferred.index(tag) if tag in preferred else len(preferred), tag))
    columns = min(2, len(tags))
    rows = (len(tags) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 4.5 * rows), squeeze=False,
                             layout="constrained")
    try:
        fig.suptitle(f"{path.name}\nTraining scalar metrics", fontsize=14)
        for ax, tag in zip(axes.flat, tags):
            samples = series[tag]
            steps, values = zip(*samples)
            ax.plot(steps, values, color="#4778aa", linewidth=0.8,
                    alpha=0.45 if tag == "loss/current" else 1,
                    marker="o" if len(samples) <= 20 else None, markersize=4,
                    label="Raw")
            if tag == "loss/current" and len(samples) >= window:
                smoothed = rolling_mean(samples, window)
                ax.plot(*zip(*smoothed), color="#df572d", linewidth=1.5,
                        label=f"Trailing mean ({window} samples)")
                ax.legend(fontsize=9)
            epoch = "epoch" in tag.lower()
            ax.set(title=tag, xlabel="Epoch" if epoch else "Step", ylabel="Value")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if tag.startswith("lr/"):
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                ax.set_ylim(bottom=0)
            ax.text(0.99, 0.98, f"Latest: {values[-1]:.6g}  |  {steps[-1]:,}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=9)
        for ax in list(axes.flat)[len(tags):]:
            ax.set_visible(False)
        output = path.with_name(path.name + ".png")
        fig.savefig(output, dpi=150)
        return output
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalar metrics beside the event file as <logfile>.png.")
    parser.add_argument("logfile", type=Path, help="TensorBoard event file")
    parser.add_argument("--window", type=int, default=200, help="Trailing mean window for loss/current (default: 200)")
    args = parser.parse_args(argv)
    if not args.logfile.is_file():
        parser.error(f"File not found: {args.logfile}")
    if args.window < 1:
        parser.error("--window must be at least 1")
    try:
        output = plot_log(args.logfile.resolve(), args.window)
    except ImportError as exc:
        print(f'Plot dependencies are missing: {exc}. Install with: python -m pip install -e ".[plot]"', file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"ssplot: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
