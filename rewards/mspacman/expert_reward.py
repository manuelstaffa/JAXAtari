import jax
import jax.numpy as jnp

PELLET_REWARD = 1.0
POWER_PELLET_REWARD = 5.0
GHOST_EAT_REWARD = 10.0
FRUIT_REWARD = 5.0
LEVEL_CLEAR_REWARD = 100.0
LIFE_LOSS_PENALTY = -100.0

PELLET_MAX_REWARD = 0.1
PELLET_HORIZONTAL_RADIUS = 34.0
PELLET_VERTICAL_RADIUS = 34.0

POWER_PELLET_MAX_REWARD = 0.5
POWER_PELLET_HORIZONTAL_RADIUS = 42.0
POWER_PELLET_VERTICAL_RADIUS = 42.0

FRUIT_MAX_REWARD = 0.2
FRUIT_HORIZONTAL_RADIUS = 40.0
FRUIT_VERTICAL_RADIUS = 30.0

ACTIVE_GHOST_MAX_PENALTY = -0.5
ACTIVE_GHOST_HORIZONTAL_RADIUS = 38.0
ACTIVE_GHOST_VERTICAL_RADIUS = 26.0

POWERED_GHOST_MAX_REWARD = 0.2
POWERED_GHOST_HORIZONTAL_RADIUS = 38.0
POWERED_GHOST_VERTICAL_RADIUS = 26.0

PELLET_GRID_WIDTH = 8.0
PELLET_GRID_HEIGHT = 12.0
PELLET_CENTER_X_OFFSET = 2.0
PELLET_CENTER_Y_OFFSET = 6.0


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

    player_pos = jnp.asarray(state.player.position, dtype=jnp.float32)

    lives_delta = state.lives - previous_state.lives
    reward += jnp.where(lives_delta < 0, LIFE_LOSS_PENALTY, 0.0)

    level_delta = state.level.id - previous_state.level.id
    reward += jnp.where(level_delta > 0, LEVEL_CLEAR_REWARD, 0.0)

    collected_delta = (
        state.level.collected_pellets - previous_state.level.collected_pellets
    )
    power_pellet_delta = jnp.sum(
        previous_state.level.power_pellets.astype(jnp.int32)
    ) - jnp.sum(state.level.power_pellets.astype(jnp.int32))
    regular_pellet_delta = jnp.maximum(0, collected_delta - power_pellet_delta)
    reward += regular_pellet_delta.astype(jnp.float32) * PELLET_REWARD
    reward += (
        jnp.maximum(0, power_pellet_delta).astype(jnp.float32) * POWER_PELLET_REWARD
    )

    ghost_eaten_delta = state.player.eaten_ghosts - previous_state.player.eaten_ghosts
    reward += jnp.maximum(0, ghost_eaten_delta).astype(jnp.float32) * GHOST_EAT_REWARD

    fruit_collected = jnp.logical_and(
        previous_state.fruit.spawned, jnp.logical_not(state.fruit.spawned)
    )
    reward += jnp.where(fruit_collected, FRUIT_REWARD, 0.0)

    pellet_indices = jnp.indices(state.level.pellets.shape, dtype=jnp.float32)
    pellet_positions = jnp.stack(
        [
            pellet_indices[0].reshape(-1) * PELLET_GRID_WIDTH + PELLET_CENTER_X_OFFSET,
            pellet_indices[1].reshape(-1) * PELLET_GRID_HEIGHT + PELLET_CENTER_Y_OFFSET,
        ],
        axis=1,
    )
    pellet_strength = jax.vmap(
        lambda pellet: _asymmetric_ellipse_exponential_distance(
            player_pos,
            pellet,
            PELLET_HORIZONTAL_RADIUS,
            PELLET_HORIZONTAL_RADIUS,
            PELLET_VERTICAL_RADIUS,
            PELLET_VERTICAL_RADIUS,
        )
    )(pellet_positions)
    reward += (
        jnp.max(jnp.where(state.level.pellets.reshape(-1), pellet_strength, 0.0))
        * PELLET_MAX_REWARD
    )

    power_pellet_positions = jnp.array(
        [[4.0, 12.0], [144.0, 12.0], [4.0, 144.0], [144.0, 144.0]],
        dtype=jnp.float32,
    )
    power_pellet_strength = jax.vmap(
        lambda pellet: _asymmetric_ellipse_exponential_distance(
            player_pos,
            pellet,
            POWER_PELLET_HORIZONTAL_RADIUS,
            POWER_PELLET_HORIZONTAL_RADIUS,
            POWER_PELLET_VERTICAL_RADIUS,
            POWER_PELLET_VERTICAL_RADIUS,
        )
    )(power_pellet_positions)
    reward += (
        jnp.max(
            jnp.where(
                state.level.power_pellets.astype(jnp.bool_), power_pellet_strength, 0.0
            )
        )
        * POWER_PELLET_MAX_REWARD
    )

    fruit_strength = _asymmetric_ellipse_exponential_distance(
        player_pos,
        jnp.asarray(state.fruit.position, dtype=jnp.float32),
        FRUIT_HORIZONTAL_RADIUS,
        FRUIT_HORIZONTAL_RADIUS,
        FRUIT_VERTICAL_RADIUS,
        FRUIT_VERTICAL_RADIUS,
    )
    reward += jnp.where(state.fruit.spawned, fruit_strength * FRUIT_MAX_REWARD, 0.0)

    ghost_positions = jnp.asarray(state.ghosts.positions, dtype=jnp.float32)
    ghost_modes = jnp.asarray(state.ghosts.modes)

    active_ghost_strength = jax.vmap(
        lambda ghost: _asymmetric_ellipse_exponential_distance(
            player_pos,
            ghost,
            ACTIVE_GHOST_HORIZONTAL_RADIUS,
            ACTIVE_GHOST_HORIZONTAL_RADIUS,
            ACTIVE_GHOST_VERTICAL_RADIUS,
            ACTIVE_GHOST_VERTICAL_RADIUS,
        )
    )(ghost_positions)
    reward += (
        jnp.max(jnp.where(ghost_modes < 3, active_ghost_strength, 0.0))
        * ACTIVE_GHOST_MAX_PENALTY
    )

    powered_ghost_strength = jax.vmap(
        lambda ghost: _asymmetric_ellipse_exponential_distance(
            player_pos,
            ghost,
            POWERED_GHOST_HORIZONTAL_RADIUS,
            POWERED_GHOST_HORIZONTAL_RADIUS,
            POWERED_GHOST_VERTICAL_RADIUS,
            POWERED_GHOST_VERTICAL_RADIUS,
        )
    )(ghost_positions)
    reward += (
        jnp.max(
            jnp.where(
                (ghost_modes == 3) | (ghost_modes == 4), powered_ghost_strength, 0.0
            )
        )
        * POWERED_GHOST_MAX_REWARD
    )

    return jnp.asarray(reward, dtype=jnp.float32)
