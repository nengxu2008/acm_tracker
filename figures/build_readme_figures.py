"""Build geometric README illustrations and synchronized excerpts of saved videos.

No inference or changes to experiment logs. Run from any working directory.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/acm_tracker_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent
BLUE = "#276FBF"
ORANGE = "#D88926"
INK = "#25364A"
ANGLE = np.deg2rad(28)
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])


def polygon(ax, points, center=(0, 0), **kwargs):
    transformed = np.asarray(points) @ ROTATION.T + np.asarray(center)
    patch = Polygon(transformed, **kwargs)
    ax.add_patch(patch)
    return patch


def box(ax, length, thickness, center=(0, 0), **kwargs):
    return polygon(ax, [(-length / 2, -thickness / 2), (length / 2, -thickness / 2),
                        (length / 2, thickness / 2), (-length / 2, thickness / 2)],
                   center=center, **kwargs)


def ship(ax):
    polygon(ax, [(-1.1, -.28), (.75, -.28), (1.15, 0), (.75, .28), (-1.1, .28)],
            facecolor=INK, edgecolor=INK, zorder=5)
    box(ax, 2.4, .8, fill=False, edgecolor="#209567", linewidth=1.8, zorder=6)


def canvas(ax):
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-4.6, -3.1), 9.2, 6.2, facecolor="#F1F6FA",
                           edgecolor="#B5C6D4", linewidth=1.2))


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def orientation_motion():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax in axes:
        canvas(ax)
        ship(ax)
    axes[0].set_title("1  Observe the target's orientation", loc="left", color=INK,
                      fontsize=13, fontweight="bold", pad=12)
    a, b = np.array([-3.2, 0]) @ ROTATION.T, np.array([3.2, 0]) @ ROTATION.T
    axes[0].annotate("", xy=b, xytext=a,
                     arrowprops=dict(arrowstyle="<->", color=BLUE, linewidth=2), zorder=3)
    axes[0].text(0, -3.55, "Green: previous OBB   •   Blue: orientation axis",
                 ha="center", color=INK, fontsize=10)
    axes[1].set_title("2  Use it as a clue for where to look", loc="left", color=INK,
                      fontsize=13, fontweight="bold", pad=12)
    for offset in [-2.8, 2.8]:
        center = np.array([offset, 0]) @ ROTATION.T
        box(axes[1], 2.4, .8, center=center, fill=False, edgecolor=BLUE,
            linestyle="--", linewidth=1.7, zorder=4)
    axes[1].text(0, -3.55, "Dashed: possible next boxes along either direction",
                 ha="center", color=INK, fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, bottom=.12, top=.84, wspace=.12)
    fig.text(.5, .02, "Illustration only. The box angle does not determine velocity or guarantee the next position.",
             ha="center", fontsize=10, color="#536579")
    save(fig, "orientation_motion")


def two_strip_search():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax in axes:
        canvas(ax)
    axes[0].add_patch(Rectangle((-4.6, -3.1), 9.2, 6.2, facecolor=BLUE,
                                edgecolor=BLUE, alpha=.12))
    axes[0].set_title("Full-frame detector search", loc="left", color=INK,
                      fontsize=13, fontweight="bold", pad=12)
    axes[0].text(0, -3.55, "Run the detector on the complete next frame",
                 ha="center", fontsize=10, color=INK)
    axes[1].set_title("My search: two rotated ROI strips", loc="left", color=INK,
                      fontsize=13, fontweight="bold", pad=12)
    box(axes[1], 6.0, 1.5, facecolor=BLUE, edgecolor=BLUE, alpha=.24, linewidth=1.4)
    box(axes[1], 1.5, 6.0, facecolor=ORANGE, edgecolor=ORANGE, alpha=.24, linewidth=1.4)
    axes[1].text(0, -3.55, "Blue: along the OBB   •   Orange: perpendicular",
                 ha="center", fontsize=10, color=INK)
    for ax in axes:
        ship(ax)
    fig.subplots_adjust(left=.02, right=.98, bottom=.2, top=.84, wspace=.12)
    fig.text(.5, .08, "Previous OBB → crop both strips from the next frame → detect → map corners back → select a box",
             ha="center", fontsize=10, color=INK)
    fig.text(.5, .025, "Schematic, not to scale. Both strips are centered on the last box; their overlap is processed twice.",
             ha="center", fontsize=10, color="#536579")
    save(fig, "two_strip_search")


def video_excerpt(filename, output_name):
    """Video index 99 corresponds to CSV frame 100; retain forty frames at 5 FPS."""
    capture = cv2.VideoCapture(str(ROOT / "records" / filename))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 99)
    frames = []
    try:
        for _ in range(40):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read requested excerpt from {filename}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    frames[0].save(OUTPUT / output_name, save_all=True, append_images=frames[1:],
                   duration=200, loop=0, optimize=True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"})
    orientation_motion()
    two_strip_search()
    video_excerpt("mytracker_ship_1.mp4", "mytracker_ship_1_preview.gif")
    video_excerpt("baseline_bytetrack_ship_1.mp4", "baseline_ship_1_preview.gif")


if __name__ == "__main__":
    main()
