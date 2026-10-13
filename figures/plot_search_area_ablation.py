"""Plot the ship_1 search-area ablation reported in the experiment table."""

import os
import csv
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/acm_tracker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/acm_tracker_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter


FRAME_WIDTH = 224
FRAME_HEIGHT = 261
TARGET_MAX_DIMENSION = 22.0


def two_strip_union_area(padding, max_move, target_size=TARGET_MAX_DIMENSION):
    """Return the unique area covered by two centered perpendicular strips."""
    thickness = target_size + 2 * padding
    length = 2 * max_move + thickness
    return 2 * length * thickness - thickness**2


roi_variants = [
    {
        "label": "p=0, m=10",
        "padding": 0,
        "max_move": 10,
        "oriented_iou": 0.3762,
        "fps": 3.31,
    },
    {
        "label": "p=2, m=20",
        "padding": 2,
        "max_move": 20,
        "oriented_iou": 0.4283,
        "fps": 2.75,
    },
    {
        "label": "p=5, m=50",
        "padding": 5,
        "max_move": 50,
        "oriented_iou": 0.4355,
        "fps": 2.64,
    },
]

for variant in roi_variants:
    variant["area"] = two_strip_union_area(
        variant["padding"], variant["max_move"]
    )

frame_area = FRAME_WIDTH * FRAME_HEIGHT
baseline = {
    "label": "ByteTrack\nfull frame",
    "area": frame_area,
    "oriented_iou": 0.4349,
    "fps": 1.45,
}


def update_logged_metrics():
    """Recompute available runs; smaller variants retain workbook summaries."""
    root = Path(__file__).resolve().parent.parent
    for item, filename in [
        (roi_variants[2], "mytracker_ship_1.csv"),
        (baseline, "baseline_bytetrack_ship_1.csv"),
    ]:
        with (root / "records" / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        item["fps"] = 1000 / (sum(float(row["latency_ms"]) for row in rows) / len(rows))
        item["oriented_iou"] = sum(float(row["oriented_iou"]) for row in rows) / len(rows)


def format_area(area):
    return f"{area / 1000:.2f}k\n({100 * area / frame_area:.1f}%)"


def plot_panel(ax, metric, ylabel, panel_title, ylim):
    areas = [variant["area"] for variant in roi_variants]
    values = [variant[metric] for variant in roi_variants]

    ax.plot(
        areas,
        values,
        color="#276FBF",
        linewidth=2.2,
        marker="o",
        markersize=8,
        label="My tracker ROI variants",
        zorder=3,
    )
    ax.scatter(
        baseline["area"],
        baseline[metric],
        marker="D",
        s=85,
        color="#D1495B",
        edgecolor="white",
        linewidth=0.8,
        label="ByteTrack full-frame baseline",
        zorder=4,
    )

    for index, variant in enumerate(roi_variants):
        offset = (8, 10) if index == 0 else (0, 10)
        alignment = "left" if index == 0 else "center"
        ax.annotate(
            variant["label"],
            (variant["area"], variant[metric]),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=9,
            color="#174A7E",
        )
    ax.annotate(
        baseline["label"],
        (baseline["area"], baseline[metric]),
        xytext=(-9, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#A42F3E",
    )

    all_areas = areas + [baseline["area"]]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(all_areas))
    ax.set_xticklabels([format_area(area) for area in all_areas])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(all_areas) / 1.25, max(all_areas) * 1.2)
    ax.set_xlabel("Nominal strip-union area, pixels (% of frame; log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, loc="left", fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    update_logged_metrics()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlepad": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=False)

    plot_panel(
        axes[0],
        "oriented_iou",
        "Mean oriented IoU (zoomed axis)",
        "A  Search area vs overlap",
        (0.35, 0.45),
    )
    plot_panel(
        axes[1],
        "fps",
        "Tracker FPS (inverse mean latency)",
        "B  Search area vs throughput",
        (1.2, 3.6),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Search-area ablation on OOTB ship_1",
        x=0.5,
        y=1.075,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Area estimate: fixed 22 px target dimension; overlap counted once; no edge clipping.\n"
        "Both crops process the overlap. Source frame: 224×261 px. One sequence; no repeated timings.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.27)

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "search_area_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "search_area_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
