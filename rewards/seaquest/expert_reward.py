import jax
import jax.numpy as jnp

from rewards.kangaroo.expert_reward import (
    COCONUT_LONG_RADIUS,
    COCONUT_MAX_PENALTY,
    COCONUT_SHORT_RADIUS,
)

LIFE_LOSS_PENALTY = -100.0
SURFACE_RESCUE_REWARD = 100.0
SURFACE_LOW_OXYGEN_REWARD = 10.0
SURFACE_HIGH_OXYGEN_PENALTY = -50.0
SURFACE_EMPTY_PENALTY = -100.0

DIVER_MAX_REWARD = 8.0

ENEMY_MAX_PENALTY = -5.0
ENEMY_HORIZONTAL_RADIUS = 50.0
ENEMY_VERTICAL_RADIUS = 20.0

ENEMY_FIRING_REWARD = 0.1
ENEMY_FIRING_Y_TOLERANCE = 8.0
ENEMY_KILL_REWARD = 10.0


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

    player_pos = jnp.array([state.player_x, state.player_y], dtype=jnp.float32)

    # Life loss penalty
    lives_delta = state.lives - previous_state.lives
    reward += jnp.where(lives_delta < 0, lives_delta * LIFE_LOSS_PENALTY, 0.0)

    # Surface reward/penalty
    reward += jnp.where(
        state.just_surfaced,
        jnp.where(
            previous_state.divers_collected > 0,
            jnp.where(
                previous_state.oxygen <= 10,
                SURFACE_LOW_OXYGEN_REWARD,
                SURFACE_HIGH_OXYGEN_PENALTY,
            ),
            SURFACE_EMPTY_PENALTY,
        ),
        0.0,
    )

    successful_rescue_delta = (
        state.successful_rescues - previous_state.successful_rescues
    )
    reward += jnp.where(
        successful_rescue_delta > 0,
        successful_rescue_delta * SURFACE_RESCUE_REWARD,
        0.0,
    )

    # Diver approach reward
    diver_positions = jnp.asarray(state.diver_positions, dtype=jnp.float32)
    diver_reward_strength = jax.vmap(
        lambda diver: _exponential_distance(player_pos, diver)
    )(diver_positions)
    reward += jnp.max(diver_reward_strength) * state.divers_collected * DIVER_MAX_REWARD

    # Enemy proximity penalty
    enemy_positions = jnp.concatenate(
        [
            jnp.asarray(state.shark_positions, dtype=jnp.float32),
            jnp.asarray(state.sub_positions, dtype=jnp.float32),
            jnp.asarray(state.enemy_missile_positions, dtype=jnp.float32),
        ],
        axis=0,
    )
    enemy_penalty_strength = jax.vmap(
        lambda enemy: _asymmetric_ellipse_exponential_distance(
            player_pos,
            enemy,
            ENEMY_HORIZONTAL_RADIUS,
            ENEMY_HORIZONTAL_RADIUS,
            ENEMY_VERTICAL_RADIUS,
            ENEMY_VERTICAL_RADIUS,
        )
    )(enemy_positions)
    reward += jnp.max(enemy_penalty_strength) * ENEMY_MAX_PENALTY

    # Enemy kill reward
    missile_pos = jnp.asarray(state.player_missile_position[:2], dtype=jnp.float32)
    previous_missile_pos = jnp.asarray(
        previous_state.player_missile_position[:2], dtype=jnp.float32
    )
    missile_delta_x = missile_pos[0] - previous_missile_pos[0]

    kill_reward_strength = jax.vmap(
        lambda enemy: jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.abs(missile_pos[1] - enemy[1]) <= ENEMY_FIRING_Y_TOLERANCE,
                    jnp.sign(missile_delta_x) == jnp.sign(enemy[0] - missile_pos[0]),
                ),
                jnp.where(
                    missile_delta_x >= 0,
                    enemy[0] >= missile_pos[0],
                    enemy[0] <= missile_pos[0],
                ),
            ),
            _exponential_distance(missile_pos, enemy[:2]),
            -1.0,
        )
    )(enemy_positions)
    reward += jnp.max(kill_reward_strength) * ENEMY_FIRING_REWARD

    reward += jnp.where(
        jnp.logical_and(state.score > previous_state.score, state.death_counter == 0),
        ENEMY_KILL_REWARD,
        0.0,
    )

    return jnp.asarray(reward, dtype=jnp.float32)
