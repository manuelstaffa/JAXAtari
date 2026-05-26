import chex
import jax
import jax.numpy as jnp

SURFACE_RESCUE_REWARD = 100.0
SURFACE_LOW_OXYGEN_REWARD = 10.0
SURFACE_HIGH_OXYGEN_PENALTY = -50.0
SURFACE_EMPTY_PENALTY = -100.0
DEATH_PENALTY = -100.0

ENEMY_PROXIMITY_MAX_PENALTY = 10.0
ENEMY_PROXIMITY_HORIZONTAL_RANGE = 52.0
ENEMY_PROXIMITY_VERTICAL_RANGE = 18.0
ENEMY_PROXIMITY_SHARPNESS = 4.5

ENEMY_FIRING_REWARD = 1.0
ENEMY_FIRING_Y_TOLERANCE = 8.0
ENEMY_FIRING_X_RANGE = 64.0

DIVER_APPROACH_MAX_REWARD = 1.0
DIVER_APPROACH_MAX_DISTANCE = 72.0
DIVER_COLLECT_REWARD = 5.0
ENEMY_DESTROY_REWARD = 5.0


def _player_position(state):
    return jnp.asarray([state.player_x, state.player_y], dtype=jnp.float32)


def _normalize_entities(positions):
    return jnp.reshape(jnp.asarray(positions, dtype=jnp.float32), (-1, 3))


def _active_mask(positions):
    return _normalize_entities(positions)[:, 2] != 0


def _active_entities(*groups):
    return jnp.concatenate([_normalize_entities(group) for group in groups], axis=0)


def _entity_distance(player_position, entity_position):
    delta = jnp.asarray(entity_position[:2], dtype=jnp.float32) - player_position
    return jnp.sqrt(jnp.sum(delta * delta))


def _enemy_proximity_penalty(player_position, enemy_position):
    dx = jnp.abs(jnp.asarray(enemy_position[0], dtype=jnp.float32) - player_position[0])
    dy = jnp.abs(jnp.asarray(enemy_position[1], dtype=jnp.float32) - player_position[1])
    ellipse_distance = (dx / ENEMY_PROXIMITY_HORIZONTAL_RANGE) ** 2 + (
        dy / ENEMY_PROXIMITY_VERTICAL_RANGE
    ) ** 2
    proximity = jnp.clip(1.0 - ellipse_distance, 0.0, 1.0)
    shaped = jnp.where(
        proximity > 0.0,
        (1.0 - jnp.exp(-ENEMY_PROXIMITY_SHARPNESS * proximity))
        / (1.0 - jnp.exp(-ENEMY_PROXIMITY_SHARPNESS)),
        0.0,
    )
    return -ENEMY_PROXIMITY_MAX_PENALTY * shaped


def _closest_enemy_penalty(state):
    player_position = _player_position(state)
    enemy_positions = _active_entities(
        state.shark_positions, state.sub_positions, state.surface_sub_position
    )
    active = _active_mask(enemy_positions)
    penalties = jax.vmap(
        lambda enemy_position, is_active: jnp.where(
            is_active,
            _enemy_proximity_penalty(player_position, enemy_position),
            0.0,
        )
    )(enemy_positions, active)
    return jnp.min(penalties)


def _firing_alignment_reward(state):
    player_position = _player_position(state)
    enemy_positions = _active_entities(
        state.shark_positions, state.sub_positions, state.surface_sub_position
    )
    active = _active_mask(enemy_positions)
    missile_active = jnp.logical_and(
        state.player_missile_position[2] != 0,
        state.player_missile_position[2] == state.player_direction,
    )
    dx = jnp.asarray(enemy_positions[:, 0], dtype=jnp.float32) - player_position[0]
    dy = jnp.abs(
        jnp.asarray(enemy_positions[:, 1], dtype=jnp.float32) - player_position[1]
    )
    facing_right = state.player_direction == 1
    facing_enemy = jnp.where(facing_right, dx > 0, dx < 0)
    within_range = jnp.abs(dx) <= ENEMY_FIRING_X_RANGE
    aligned = jnp.logical_and(active, missile_active)
    aligned = jnp.logical_and(aligned, dy <= ENEMY_FIRING_Y_TOLERANCE)
    aligned = jnp.logical_and(aligned, facing_enemy)
    aligned = jnp.logical_and(aligned, within_range)
    return jnp.where(jnp.any(aligned), ENEMY_FIRING_REWARD, 0.0)


def _diver_approach_reward(previous_state, state):
    current_player_position = _player_position(state)
    previous_player_position = _player_position(previous_state)
    current_distances = jax.vmap(
        lambda position: _entity_distance(current_player_position, position)
    )(_normalize_entities(state.diver_positions))
    previous_distances = jax.vmap(
        lambda position: _entity_distance(previous_player_position, position)
    )(_normalize_entities(previous_state.diver_positions))
    current_distances = jnp.where(
        _active_mask(state.diver_positions), current_distances, jnp.inf
    )
    previous_distances = jnp.where(
        _active_mask(previous_state.diver_positions), previous_distances, jnp.inf
    )
    has_current = jnp.any(_active_mask(state.diver_positions))
    has_previous = jnp.any(_active_mask(previous_state.diver_positions))
    current_nearest = jnp.min(current_distances)
    previous_nearest = jnp.min(previous_distances)
    improvement = jnp.where(
        jnp.logical_and(has_current, has_previous),
        jnp.maximum(previous_nearest - current_nearest, 0.0),
        0.0,
    )
    return DIVER_APPROACH_MAX_REWARD * jnp.clip(
        improvement / DIVER_APPROACH_MAX_DISTANCE, 0.0, 1.0
    )


def reward_function(previous_state, state) -> chex.Array:
    reward = 0.0

    rescue_delta = jnp.maximum(
        state.successful_rescues - previous_state.successful_rescues, 0
    )
    reward += SURFACE_RESCUE_REWARD * rescue_delta

    just_surfaced = jnp.logical_and(state.just_surfaced == 1, rescue_delta == 0)
    reward += jnp.where(
        just_surfaced,
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

    death_event = state.death_counter == 90
    reward += jnp.where(death_event, DEATH_PENALTY, 0.0)

    reward += jnp.where(
        jnp.logical_and(state.score > previous_state.score, state.death_counter == 0),
        ENEMY_DESTROY_REWARD,
        0.0,
    )
    reward += DIVER_COLLECT_REWARD * jnp.maximum(
        state.divers_collected - previous_state.divers_collected, 0
    )
    reward += _closest_enemy_penalty(state)
    reward += _firing_alignment_reward(state)
    reward += _diver_approach_reward(previous_state, state)

    return jnp.asarray(reward, dtype=jnp.float32)
