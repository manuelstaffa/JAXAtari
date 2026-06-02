import jax
import jax.numpy as jnp

SCORE_REWARD = 10.0
BALL_MAX_REWARD = 0.1

LEFT_RADIUS = 80.0
RIGHT_RADIUS = 10.0
VERTICAL_RADIUS = 150.0


def _linear_distance(a, b, alpha: float = 0.005) -> jax.Array:
    try:
        distance = jnp.sqrt(jnp.sum((a - b) ** 2))
        val = jnp.maximum(0.0, 1.0 - alpha * distance)

        return jnp.asarray(val, dtype=jnp.float32)
    except (TypeError, IndexError):
        return jnp.asarray(0.0, dtype=jnp.float32)


def _exponential_distance(a, b, alpha: float = 0.05) -> jax.Array:
    try:
        distance = jnp.sqrt(jnp.sum((a - b) ** 2))
        exp_min = jnp.exp(-alpha)
        val = (jnp.exp(-alpha * distance) - exp_min) / (1.0 - exp_min)

        return jnp.asarray(val, dtype=jnp.float32)
    except (TypeError, IndexError):
        return jnp.asarray(0.0, dtype=jnp.float32)


def _in_ellipse(a, b, horizontal_radius, vertical_radius) -> jax.Array:
    try:
        horizontal_term = (a[0] - b[0]) / horizontal_radius
        vertical_term = (a[1] - b[1]) / vertical_radius
        ellipse_distance = (
            horizontal_term * horizontal_term + vertical_term * vertical_term
        )

        return jnp.asarray(ellipse_distance < 1.0, dtype=jnp.bool)
    except (TypeError, IndexError):
        return jnp.asarray(False, dtype=jnp.bool)


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
        return jnp.asarray(False, dtype=jnp.bool)


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

    # Score-based rewards
    player_delta_score = state.player_score - previous_state.player_score
    reward += jnp.where(player_delta_score > 0, player_delta_score * SCORE_REWARD, 0.0)

    enemy_delta_score = state.enemy_score - previous_state.enemy_score
    reward += jnp.where(enemy_delta_score > 0, enemy_delta_score * -SCORE_REWARD, 0.0)

    # Ball proximity reward
    player_pos = jnp.array([140.0, state.player_y], dtype=jnp.float32)
    ball_pos = jnp.array([state.ball_x, state.ball_y], dtype=jnp.float32)
    ball_y_speed = jnp.abs(state.ball_vel_y)
    ball_x_speed = jnp.abs(state.ball_vel_x)
    proximity_reward_scale = _asymmetric_ellipse_exponential_distance(
        player_pos,
        ball_pos,
        LEFT_RADIUS * ball_x_speed / 2.0,
        RIGHT_RADIUS,
        VERTICAL_RADIUS * ball_y_speed,
        VERTICAL_RADIUS * ball_y_speed,
    )
    reward += BALL_MAX_REWARD * proximity_reward_scale

    return jnp.asarray(reward, dtype=jnp.float32)
