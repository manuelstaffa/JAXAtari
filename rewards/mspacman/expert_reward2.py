import jax
import jax.numpy as jnp

PELLET_REWARD = 1.0
POWER_PELLET_REWARD = 5.0
GHOST_EAT_REWARD = 10.0
FRUIT_REWARD = 5.0
LEVEL_CLEAR_REWARD = 100.0
LIFE_LOSS_PENALTY = -100.0

PELLET_APPROACH_MAX_REWARD = 1.0
POWER_PELLET_APPROACH_MAX_REWARD = 1.0
FRUIT_APPROACH_MAX_REWARD = 2.0
ACTIVE_GHOST_APPROACH_MAX_PENALTY = 1.0
FRIGHTENED_GHOST_APPROACH_MAX_REWARD = 1.0

PELLET_APPROACH_DISTANCE_SCALE = 6.0
POWER_PELLET_APPROACH_DISTANCE_SCALE = 8.0
FRUIT_APPROACH_DISTANCE_SCALE = 48.0
ACTIVE_GHOST_APPROACH_DISTANCE_SCALE = 20.0
FRIGHTENED_GHOST_APPROACH_DISTANCE_SCALE = 20.0

FRUIT_NEARBY_DISTANCE = 56.0
FRUIT_COLLECTION_DISTANCE = 6.0

GHOST_MODE_RANDOM = 0
GHOST_MODE_CHASE = 1
GHOST_MODE_SCATTER = 2
GHOST_MODE_FRIGHTENED = 3
GHOST_MODE_BLINKING = 4

POWER_PELLET_TILES = jnp.asarray(
    [[1.0, 3.0], [36.0, 3.0], [1.0, 36.0], [36.0, 36.0]],
    dtype=jnp.float32,
)


def _to_float32(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _player_position(state):
    return _to_float32(state.player.position)


def _player_tile_position(state):
    player_position = _player_position(state)
    tile_x = (player_position[0] - 2.0) / 8.0
    tile_y = (player_position[1] + 4.0) / 12.0
    return jnp.asarray([tile_x, tile_y], dtype=jnp.float32)


def _nearest_masked_distance(points, mask, origin):
    points = _to_float32(points)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    origin = _to_float32(origin)
    deltas = points - origin[None, :]
    distances = jnp.sqrt(jnp.sum(deltas * deltas, axis=1))
    masked_distances = jnp.where(mask, distances, jnp.inf)
    return jnp.min(masked_distances), jnp.any(mask)


def _nearest_pellet_distance(state):
    pellets = jnp.asarray(state.level.pellets, dtype=jnp.bool_)
    tile_pos = _player_tile_position(state)

    x_idx = jnp.arange(pellets.shape[0], dtype=jnp.float32)[:, None]
    y_idx = jnp.arange(pellets.shape[1], dtype=jnp.float32)[None, :]

    dx = x_idx - tile_pos[0]
    dy = y_idx - tile_pos[1]
    distances = jnp.sqrt(dx * dx + dy * dy)
    masked_distances = jnp.where(pellets, distances, jnp.inf)

    return jnp.min(masked_distances), jnp.any(pellets)


def _nearest_power_pellet_distance(state):
    available = jnp.asarray(state.level.power_pellets, dtype=jnp.bool_)
    tile_pos = _player_tile_position(state)
    return _nearest_masked_distance(POWER_PELLET_TILES, available, tile_pos)


def _approach_signal(previous_distance, current_distance, max_value, distance_scale):
    progress = jnp.maximum(previous_distance - current_distance, 0.0)
    return max_value * jnp.clip(progress / distance_scale, 0.0, 1.0)


def reward_function(previous_state, state) -> jax.Array:
    reward = 0.0

    life_loss = jnp.maximum(previous_state.lives - state.lives, 0)
    level_advanced = state.level.id > previous_state.level.id

    previous_power_available = jnp.sum(
        jnp.asarray(previous_state.level.power_pellets, dtype=jnp.int32)
    )
    current_power_available = jnp.sum(
        jnp.asarray(state.level.power_pellets, dtype=jnp.int32)
    )
    power_collected = jnp.maximum(previous_power_available - current_power_available, 0)

    collected_delta = jnp.maximum(
        state.level.collected_pellets - previous_state.level.collected_pellets, 0
    )
    regular_collected = jnp.maximum(collected_delta - power_collected, 0)

    ghosts_eaten = jnp.maximum(
        state.player.eaten_ghosts - previous_state.player.eaten_ghosts, 0
    )

    fruit_disappeared = jnp.logical_and(
        previous_state.fruit.spawned, ~state.fruit.spawned
    )
    fruit_distance = jnp.sqrt(
        jnp.sum(
            (_player_position(state) - _to_float32(previous_state.fruit.position)) ** 2
        )
    )
    fruit_collected = jnp.logical_and(
        fruit_disappeared,
        fruit_distance <= FRUIT_COLLECTION_DISTANCE,
    )

    reward += PELLET_REWARD * _to_float32(regular_collected)
    reward += POWER_PELLET_REWARD * _to_float32(power_collected)
    reward += GHOST_EAT_REWARD * _to_float32(ghosts_eaten)
    reward += jnp.where(fruit_collected, FRUIT_REWARD, 0.0)
    reward += jnp.where(level_advanced, LEVEL_CLEAR_REWARD, 0.0)
    reward += LIFE_LOSS_PENALTY * _to_float32(life_loss)

    stable_step = jnp.logical_and(
        state.level.id == previous_state.level.id, life_loss == 0
    )

    previous_nearest_pellet, previous_has_pellet = _nearest_pellet_distance(
        previous_state
    )
    current_nearest_pellet, current_has_pellet = _nearest_pellet_distance(state)
    pellet_shaping = _approach_signal(
        previous_nearest_pellet,
        current_nearest_pellet,
        PELLET_APPROACH_MAX_REWARD,
        PELLET_APPROACH_DISTANCE_SCALE,
    )
    reward += jnp.where(
        jnp.logical_and(
            stable_step, jnp.logical_and(previous_has_pellet, current_has_pellet)
        ),
        pellet_shaping,
        0.0,
    )

    previous_nearest_power, previous_has_power = _nearest_power_pellet_distance(
        previous_state
    )
    current_nearest_power, current_has_power = _nearest_power_pellet_distance(state)
    power_pellet_shaping = _approach_signal(
        previous_nearest_power,
        current_nearest_power,
        POWER_PELLET_APPROACH_MAX_REWARD,
        POWER_PELLET_APPROACH_DISTANCE_SCALE,
    )
    reward += jnp.where(
        jnp.logical_and(
            stable_step, jnp.logical_and(previous_has_power, current_has_power)
        ),
        power_pellet_shaping,
        0.0,
    )

    previous_fruit_distance = jnp.sqrt(
        jnp.sum(
            (
                _player_position(previous_state)
                - _to_float32(previous_state.fruit.position)
            )
            ** 2
        )
    )
    current_fruit_distance = jnp.sqrt(
        jnp.sum((_player_position(state) - _to_float32(state.fruit.position)) ** 2)
    )
    fruit_is_nearby = (
        jnp.minimum(previous_fruit_distance, current_fruit_distance)
        <= FRUIT_NEARBY_DISTANCE
    )
    fruit_shaping = _approach_signal(
        previous_fruit_distance,
        current_fruit_distance,
        FRUIT_APPROACH_MAX_REWARD,
        FRUIT_APPROACH_DISTANCE_SCALE,
    )
    reward += jnp.where(
        jnp.logical_and(
            stable_step,
            jnp.logical_and(
                previous_state.fruit.spawned,
                jnp.logical_and(state.fruit.spawned, fruit_is_nearby),
            ),
        ),
        fruit_shaping,
        0.0,
    )

    previous_ghost_positions = _to_float32(previous_state.ghosts.positions)
    current_ghost_positions = _to_float32(state.ghosts.positions)

    previous_modes = jnp.asarray(previous_state.ghosts.modes)
    current_modes = jnp.asarray(state.ghosts.modes)

    previous_active_ghost_mask = previous_modes <= GHOST_MODE_SCATTER
    current_active_ghost_mask = current_modes <= GHOST_MODE_SCATTER

    previous_active_ghost_distance, previous_has_active_ghost = (
        _nearest_masked_distance(
            previous_ghost_positions,
            previous_active_ghost_mask,
            _player_position(previous_state),
        )
    )
    current_active_ghost_distance, current_has_active_ghost = _nearest_masked_distance(
        current_ghost_positions,
        current_active_ghost_mask,
        _player_position(state),
    )

    active_ghost_penalty = _approach_signal(
        previous_active_ghost_distance,
        current_active_ghost_distance,
        ACTIVE_GHOST_APPROACH_MAX_PENALTY,
        ACTIVE_GHOST_APPROACH_DISTANCE_SCALE,
    )
    reward -= jnp.where(
        jnp.logical_and(
            stable_step,
            jnp.logical_and(previous_has_active_ghost, current_has_active_ghost),
        ),
        active_ghost_penalty,
        0.0,
    )

    previous_frightened_mask = jnp.logical_or(
        previous_modes == GHOST_MODE_FRIGHTENED,
        previous_modes == GHOST_MODE_BLINKING,
    )
    current_frightened_mask = jnp.logical_or(
        current_modes == GHOST_MODE_FRIGHTENED,
        current_modes == GHOST_MODE_BLINKING,
    )

    previous_frightened_distance, previous_has_frightened = _nearest_masked_distance(
        previous_ghost_positions,
        previous_frightened_mask,
        _player_position(previous_state),
    )
    current_frightened_distance, current_has_frightened = _nearest_masked_distance(
        current_ghost_positions,
        current_frightened_mask,
        _player_position(state),
    )

    frightened_approach_reward = _approach_signal(
        previous_frightened_distance,
        current_frightened_distance,
        FRIGHTENED_GHOST_APPROACH_MAX_REWARD,
        FRIGHTENED_GHOST_APPROACH_DISTANCE_SCALE,
    )
    reward += jnp.where(
        jnp.logical_and(
            stable_step,
            jnp.logical_and(previous_has_frightened, current_has_frightened),
        ),
        frightened_approach_reward,
        0.0,
    )

    return jnp.asarray(reward, dtype=jnp.float32)
