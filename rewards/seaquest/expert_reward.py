import jax
import jax.numpy as jnp

from rewards.kangaroo.expert_reward import (
    COCONUT_LONG_RADIUS,
    COCONUT_MAX_PENALTY,
    COCONUT_SHORT_RADIUS,
)

LIFE_LOSS_PENALTY = -100.0
SURFACE_RESCUE_REWARD = 100.0
SURFACE_LOW_OXYGEN_REWARD = 20.0
SURFACE_HIGH_OXYGEN_PENALTY = -50.0
SURFACE_EMPTY_PENALTY = -100.0
LOW_OXYGEN_THRESHOLD = 15

DIVER_MAX_REWARD = 0.2
DIVER_HORIZONTAL_RADIUS = 150.0
DIVER_VERTICAL_RADIUS = 100.0

ENEMY_MAX_PENALTY = -0.5
ENEMY_HORIZONTAL_RADIUS = 80.0
ENEMY_VERTICAL_RADIUS = 20.0

ENEMY_KILL_REWARD = 10.0


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
    reward += jnp.where(lives_delta < 0, LIFE_LOSS_PENALTY, 0.0)

    # Surface reward/penalty
    surfaced = jnp.logical_and(state.just_surfaced, previous_state.just_surfaced == 0)
    surface_reward = jnp.where(
        previous_state.divers_collected >= 6,
        SURFACE_RESCUE_REWARD,
        jnp.where(
            previous_state.divers_collected > 0,
            jnp.where(
                previous_state.oxygen <= LOW_OXYGEN_THRESHOLD,
                SURFACE_LOW_OXYGEN_REWARD,
                SURFACE_HIGH_OXYGEN_PENALTY,
            ),
            SURFACE_EMPTY_PENALTY,
        ),
    )
    reward += jnp.where(surfaced, surface_reward, 0.0)

    # Diver approach reward
    diver_positions = jnp.asarray(state.diver_positions, dtype=jnp.float32)
    diver_reward_strength = jax.vmap(
        lambda diver: _asymmetric_ellipse_exponential_distance(
            player_pos,
            diver,
            DIVER_HORIZONTAL_RADIUS,
            DIVER_HORIZONTAL_RADIUS,
            DIVER_VERTICAL_RADIUS,
            DIVER_VERTICAL_RADIUS,
        )
    )(diver_positions)
    reward += jnp.max(diver_reward_strength) * DIVER_MAX_REWARD

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
    reward += jnp.where(
        jnp.logical_and(state.score > previous_state.score, state.death_counter == 0),
        ENEMY_KILL_REWARD,
        0.0,
    )

    return jnp.asarray(reward, dtype=jnp.float32)
