#!/usr/bin/env python3
"""
Heatmap visualizer for the exponential asymmetric-ellipse distance.

Values are normalized to the range [0, 1], where 1 is at the center and
0 is at or beyond the ellipse boundary.

Run: python3 rewards/freeway/visualize_asymmetric_ellipse_exponential_heatmap.py
"""

import numpy as np
import matplotlib.pyplot as plt

CENTER = np.array([50.0, 50.0])
HORIZONTAL_RADIUS = 40.0
FRONT_RADIUS = 12.0
BACK_RADIUS = 6.0
GRID_RES = 600
SAVE_PNG = False
OUTPUT_FILE = "asymmetric_ellipse_exponential_heatmap.png"
ALPHA = 3.0


def asymmetric_ellipse_exponential_distance(
    pts, center, horizontal_radius, front_radius, back_radius, alpha=ALPHA
):
    horizontal_term = (pts[..., 0] - center[0]) / horizontal_radius
    vertical_radius = np.where(pts[..., 1] < center[1], front_radius, back_radius)
    vertical_term = (pts[..., 1] - center[1]) / vertical_radius
    ellipse_distance = np.sqrt(
        horizontal_term * horizontal_term + vertical_term * vertical_term
    )
    ellipse_distance = np.clip(ellipse_distance, 0.0, 1.0)
    exp_min = np.exp(-alpha)
    values = (np.exp(-alpha * ellipse_distance) - exp_min) / (1.0 - exp_min)
    return np.clip(values, 0.0, 1.0)


def main():
    x_min = CENTER[0] - HORIZONTAL_RADIUS * 1.6
    x_max = CENTER[0] + HORIZONTAL_RADIUS * 1.6
    y_extent = max(FRONT_RADIUS, BACK_RADIUS) * 1.6
    y_min = CENTER[1] - y_extent
    y_max = CENTER[1] + y_extent

    x = np.linspace(x_min, x_max, GRID_RES)
    y = np.linspace(y_min, y_max, GRID_RES)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack([xx, yy], axis=-1)

    values = asymmetric_ellipse_exponential_distance(
        pts, CENTER, HORIZONTAL_RADIUS, FRONT_RADIUS, BACK_RADIUS
    )

    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        values,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    plt.colorbar(im, label="exponential distance value")
    plt.contour(
        xx,
        yy,
        values,
        levels=[0.0, 0.25, 0.5, 0.75, 1.0],
        colors="white",
        alpha=0.25,
        linewidths=0.7,
    )
    plt.scatter([CENTER[0]], [CENTER[1]], color="cyan", zorder=5)
    plt.axhline(CENTER[1], color="gray", linestyle="--", linewidth=0.8)

    plt.title(
        f"Asymmetric ellipse exponential heatmap — horiz={HORIZONTAL_RADIUS}, front={FRONT_RADIUS}, back={BACK_RADIUS}, alpha={ALPHA}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if SAVE_PNG:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()
