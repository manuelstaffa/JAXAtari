import jax
import jax.numpy as jnp

GOAL_REWARD = 100.0

MOVE_UP_REWARD = 1.0
MOVE_DOWN_PENALTY = -2.0

CAR_MAX_PENALTY = -10.0
HORIZONTAL_RADIUS = 40.0
FRONT_RADIUS = 20.0
BACK_RADIUS = 10.0
ALPHA = 3.0


def _Linear_distance(a, b) -> jax.Array:
    distance = jnp.sqrt(jnp.sum((a - b) ** 2))

    return jnp.asarray(distance, dtype=jnp.float32)


def _exponential_distance(a, b, alpha: float = 3.0) -> jax.Array:
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
    a, b, horizontal_radius, front_radius, back_radius
) -> jax.Array:
    horizontal_term = (a[0] - b[0]) / horizontal_radius
    vertical_radius = jnp.where(a[1] < b[1], front_radius, back_radius)
    vertical_term = (a[1] - b[1]) / vertical_radius
    ellipse_distance = horizontal_term * horizontal_term + vertical_term * vertical_term

    return jnp.asarray(ellipse_distance < 1.0, dtype=jnp.bool)


def _asymmetric_ellipse_linear_distance(
    a, b, horizontal_radius, front_radius, back_radius
) -> jax.Array:
    horizontal_term = (a[0] - b[0]) / horizontal_radius
    vertical_radius = jnp.where(a[1] < b[1], front_radius, back_radius)
    vertical_term = (a[1] - b[1]) / vertical_radius
    r = jnp.sqrt(horizontal_term * horizontal_term + vertical_term * vertical_term)
    r = jnp.clip(r, 0.0, 1.0)
    val = 1.0 - r

    return jnp.asarray(jnp.clip(val, 0.0, 1.0), dtype=jnp.float32)


def _asymmetric_ellipse_exponential_distance(
    a, b, horizontal_radius, front_radius, back_radius, alpha: float = 3.0
) -> jax.Array:
    horizontal_term = (a[0] - b[0]) / horizontal_radius
    vertical_radius = jnp.where(a[1] < b[1], front_radius, back_radius)
    vertical_term = (a[1] - b[1]) / vertical_radius
    r = jnp.sqrt(horizontal_term * horizontal_term + vertical_term * vertical_term)
    r = jnp.clip(r, 0.0, 1.0)
    exp_min = jnp.exp(-alpha)
    val = (jnp.exp(-alpha * r) - exp_min) / (1.0 - exp_min)

    return jnp.asarray(jnp.clip(val, 0.0, 1.0), dtype=jnp.float32)


def reward_function(previous_state, state) -> jax.Array:
    reward = jnp.asarray(0.0, dtype=jnp.float32)

    delta_score = state.score - previous_state.score
    reward += jnp.where(delta_score > 0, state.score * GOAL_REWARD, 0.0)

    delta_y = state.chicken_y - previous_state.chicken_y
    reward += jnp.where(delta_y < 0, MOVE_UP_REWARD, 0.0)
    reward += jnp.where(delta_y > 0, MOVE_DOWN_PENALTY, 0.0)

    car_positions = jnp.asarray(state.cars, dtype=jnp.float32)
    chicken_pos = jnp.array([44.0, state.chicken_y], dtype=jnp.float32)
    car_penalty_strength = jax.vmap(
        lambda car: _asymmetric_ellipse_exponential_distance(
            chicken_pos,
            car,
            HORIZONTAL_RADIUS,
            FRONT_RADIUS,
            BACK_RADIUS,
            ALPHA,
        )
    )(car_positions)
    reward += jnp.sum(car_penalty_strength) * CAR_MAX_PENALTY

    return jnp.asarray(reward, dtype=jnp.float32)
