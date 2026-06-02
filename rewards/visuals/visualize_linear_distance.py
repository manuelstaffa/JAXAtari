import numpy as np
import matplotlib.pyplot as plt


def linear_distance(distance, alpha=0.01):
    return np.maximum(0.0, 1.0 - alpha * distance)


def exponential_distance(distance, alpha=0.05):
    exp_min = np.exp(-alpha)
    return (np.exp(-alpha * distance) - exp_min) / (1.0 - exp_min)


distances = np.linspace(0, 250, 1000)

linear_alphas = [0.02, 0.01, 0.005]
exp_alphas = [0.1, 0.05, 0.02]

plt.figure(figsize=(10, 6))

for alpha in linear_alphas:
    plt.plot(
        distances,
        linear_distance(distances, alpha),
        label=f"Linear α={alpha}",
    )

for alpha in exp_alphas:
    plt.plot(
        distances,
        exponential_distance(distances, alpha),
        "--",
        label=f"Exponential α={alpha}",
    )

plt.xlabel("Distance")
plt.ylabel("Value")
plt.title("Linear and Exponential Distance Falloff")
plt.xlim(0, 250)
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
