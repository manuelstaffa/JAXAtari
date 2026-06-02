import jax
import jax.numpy as jnp


def _linear_distance(a, b, alpha: float = 0.005) -> jax.Array:
    distance = jnp.sqrt(jnp.sum((a - b) ** 2))
    val = jnp.maximum(0.0, 1.0 - alpha * distance)

    return jnp.asarray(val, dtype=jnp.float32)


def _exponential_distance(a, b, alpha: float = 0.05) -> jax.Array:
    distance = jnp.sqrt(jnp.sum((a - b) ** 2))
    exp_min = jnp.exp(-alpha)
    val = (jnp.exp(-alpha * distance) - exp_min) / (1.0 - exp_min)

    return jnp.asarray(val, dtype=jnp.float32)


def _in_ellipse(a, b, horizontal_radius, vertical_radius) -> jax.Array:
    horizontal_term = (a[0] - b[0]) / horizontal_radius
    vertical_term = (a[1] - b[1]) / vertical_radius
    ellipse_distance = horizontal_term * horizontal_term + vertical_term * vertical_term

    return jnp.asarray(ellipse_distance < 1.0, dtype=jnp.bool)


def _in_asymmetric_ellipse(
    a, b, left_radius, right_radius, up_radius, down_radius
) -> jax.Array:
    try:
        horizontal_radius = jnp.where(a[0] < b[0], right_radius, left_radius)
        horizontal_term = (a[0] - b[0]) / horizontal_radius
        vertical_radius = jnp.where(a[1] < b[1], down_radius, up_radius)
        vertical_term = (a[1] - b[1]) / vertical_radius
        ellipse_distance = (
            horizontal_term * horizontal_term + vertical_term * vertical_term
        )

        return jnp.asarray(ellipse_distance < 1.0, dtype=jnp.bool)
    except (TypeError, IndexError):
        return jnp.asarray(0.0, dtype=jnp.float32)


def _asymmetric_ellipse_linear_distance(
    a, b, left_radius, right_radius, up_radius, down_radius
) -> jax.Array:
    try:
        horizontal_radius = jnp.where(a[0] < b[0], right_radius, left_radius)
        horizontal_term = (a[0] - b[0]) / horizontal_radius
        vertical_radius = jnp.where(a[1] < b[1], down_radius, up_radius)
        vertical_term = (a[1] - b[1]) / vertical_radius
        r = jnp.sqrt(horizontal_term * horizontal_term + vertical_term * vertical_term)
        r = jnp.clip(r, 0.0, 1.0)
        val = 1.0 - r

        return jnp.asarray(jnp.clip(val, 0.0, 1.0), dtype=jnp.float32)
    except (TypeError, IndexError):
        return jnp.asarray(0.0, dtype=jnp.float32)


def _asymmetric_ellipse_exponential_distance(
    a, b, left_radius, right_radius, up_radius, down_radius, alpha: float = 3.0
) -> jax.Array:
    try:
        horizontal_radius = jnp.where(a[0] < b[0], right_radius, left_radius)
        horizontal_term = (a[0] - b[0]) / horizontal_radius
        vertical_radius = jnp.where(a[1] < b[1], down_radius, up_radius)
        vertical_term = (a[1] - b[1]) / vertical_radius
        r = jnp.sqrt(horizontal_term * horizontal_term + vertical_term * vertical_term)
        r = jnp.clip(r, 0.0, 1.0)
        exp_min = jnp.exp(-alpha)
        val = (jnp.exp(-alpha * r) - exp_min) / (1.0 - exp_min)

        return jnp.asarray(jnp.clip(val, 0.0, 1.0), dtype=jnp.float32)
    except (TypeError, IndexError):
        return jnp.asarray(0.0, dtype=jnp.float32)


def reward_function(previous_state, state) -> jax.Array:
    reward = jnp.asarray(0.0, dtype=jnp.float32)

    return jnp.asarray(reward, dtype=jnp.float32)
