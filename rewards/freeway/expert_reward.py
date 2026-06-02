import jax
import jax.numpy as jnp

GOAL_REWARD = 100.0

MOVE_UP_REWARD = 1.0
MOVE_DOWN_PENALTY = -2.0

CAR_MAX_PENALTY = -1.0
HORIZONTAL_RADIUS = 40.0
UP_RADIUS = 20.0
DOWN_RADIUS = 15.0


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

    # Score-based reward
    delta_score = state.score - previous_state.score
    reward += jnp.where(delta_score > 0, state.score * GOAL_REWARD, 0.0)

    # Movement reward/penalty
    delta_y = state.chicken_y - previous_state.chicken_y
    reward += jnp.where(delta_y < 0, MOVE_UP_REWARD, 0.0)
    reward += jnp.where(delta_y > 0, MOVE_DOWN_PENALTY, 0.0)

    # Car proximity penalty
    car_positions = jnp.asarray(state.cars, dtype=jnp.float32)
    chicken_pos = jnp.array([44.0, state.chicken_y], dtype=jnp.float32)
    car_penalty_strength = jax.vmap(
        lambda car: _asymmetric_ellipse_exponential_distance(
            chicken_pos,
            car,
            HORIZONTAL_RADIUS,
            HORIZONTAL_RADIUS,
            UP_RADIUS,
            DOWN_RADIUS,
        )
    )(car_positions)
    reward += jnp.max(car_penalty_strength) * CAR_MAX_PENALTY

    return jnp.asarray(reward, dtype=jnp.float32)
