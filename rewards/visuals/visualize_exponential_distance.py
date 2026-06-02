#!/usr/bin/env python3
"""Visualize `_exponential_distance` from rewards.freeway.expert_reward.

Usage:
  python scripts/visualize_exponential_distance.py --alpha 3.0 --max 250 --num 1000
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def _exponential_distance(a, b, alpha: float = 3.0) -> jax.Array:
    distance = jnp.sqrt(jnp.sum((a - b) ** 2))
    exp_min = jnp.exp(-alpha)
    val = (jnp.exp(-alpha * distance) - exp_min) / (1.0 - exp_min)

    return jnp.asarray(val, dtype=jnp.float32)


def compute(alpha: float, max_dist: float = 250.0, num: int = 1000):
    xs = jnp.linspace(0.0, max_dist, num)
    b = jnp.array([0.0, 0.0], dtype=jnp.float32)

    def f(x):
        a = jnp.array([x, 0.0], dtype=jnp.float32)
        return _exponential_distance(a, b, alpha)

    vals = jax.vmap(f)(xs)

    return np.asarray(xs), np.asarray(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=3.0, help="alpha parameter")
    parser.add_argument("--max", type=float, default=250.0, help="maximum distance")
    parser.add_argument("--num", type=int, default=1000, help="number of points")
    parser.add_argument(
        "--out", type=str, default=None, help="save plot to file instead of showing"
    )
    args = parser.parse_args()

    xs, ys = compute(args.alpha, args.max, args.num)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, lw=2)
    plt.xlabel("Distance")
    plt.ylabel("_exponential_distance")
    plt.title(f"_exponential_distance(alpha={args.alpha})")
    plt.grid(True)

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
