#!/usr/bin/env python3
"""
Simple visualizer for an asymmetric ellipse like in sample_rf._in_asymmetric_ellipse.

Adjust the global parameters below to explore different shapes.
Run: python3 rewards/freeway/visualize_asymmetric_ellipse.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Global parameters (tweak these)
CENTER = np.array([50.0, 50.0])
LEFT_RADIUS = 40.0
RIGHT_RADIUS = 40.0
FRONT_RADIUS = 12.0  # used when point_y < center_y
BACK_RADIUS = 6.0  # used when point_y >= center_y
GRID_RES = 600
SAVE_PNG = False
OUTPUT_FILE = "asymmetric_ellipse.png"


def in_asymmetric_ellipse(
    pts, center, left_radius, right_radius, front_radius, back_radius
):
    """Vectorized test for points inside the asymmetric ellipse.

    pts: array of shape (..., 2) representing x,y coordinates.
    Returns boolean array of shape pts[...,0].
    """
    horizontal_radius = np.where(pts[..., 0] < center[0], left_radius, right_radius)
    horizontal_term = (pts[..., 0] - center[0]) / horizontal_radius
    vertical_radius = np.where(pts[..., 1] < center[1], front_radius, back_radius)
    vertical_term = (pts[..., 1] - center[1]) / vertical_radius
    ellipse_distance = horizontal_term * horizontal_term + vertical_term * vertical_term
    return ellipse_distance < 1.0


def main():
    x_min = CENTER[0] - max(LEFT_RADIUS, RIGHT_RADIUS) * 1.6
    x_max = CENTER[0] + max(LEFT_RADIUS, RIGHT_RADIUS) * 1.6
    y_extent = max(FRONT_RADIUS, BACK_RADIUS) * 1.6
    y_min = CENTER[1] - y_extent
    y_max = CENTER[1] + y_extent

    x = np.linspace(x_min, x_max, GRID_RES)
    y = np.linspace(y_min, y_max, GRID_RES)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack([xx, yy], axis=-1)

    mask = in_asymmetric_ellipse(
        pts, CENTER, LEFT_RADIUS, RIGHT_RADIUS, FRONT_RADIUS, BACK_RADIUS
    )

    plt.figure(figsize=(6, 6))
    # filled region
    plt.contourf(
        xx, yy, mask, levels=[-0.5, 0.5, 1.5], colors=["white", "#4C72B0"], alpha=0.7
    )
    # boundary
    plt.contour(xx, yy, mask, levels=[0.5], colors=["k"], linewidths=1)

    # center and dividing horizontal line
    plt.scatter([CENTER[0]], [CENTER[1]], color="red", zorder=5)
    plt.axhline(CENTER[1], color="gray", linestyle="--", linewidth=0.8)

    plt.title(
        f"Asymmetric ellipse — left={LEFT_RADIUS}, right={RIGHT_RADIUS}, front={FRONT_RADIUS}, back={BACK_RADIUS}"
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
