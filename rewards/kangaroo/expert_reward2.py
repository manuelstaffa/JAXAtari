import jax
import jax.numpy as jnp

CHILD_REACH_REWARD = 100.0
LIFE_LOSS_PENALTY = 100.0
FRUIT_REWARD = 5.0
LAST_FRUIT_REWARD = 10.0
BELL_REWARD = 7.0
MONKEY_KILL_REWARD = 5.0

COCONUT_MAX_PENALTY = 1.0
COCONUT_SHARPNESS = 5.0
COCONUT_VERTICAL_RADIUS = 92.0
COCONUT_HORIZONTAL_RADIUS = 72.0
COCONUT_RADIUS = 16.0

CHILD_APPROACH_MAX_REWARD = 0.5
CHILD_APPROACH_MAX_DISTANCE = 72.0
CHILD_SIMILAR_Y_THRESHOLD = 20.0
CHILD_X_WEIGHT = 0.35

LADDER_APPROACH_MAX_REWARD = 0.2
LADDER_APPROACH_MAX_DISTANCE = 120.0
LADDER_Y_WEIGHT = 0.35
LADDER_VERTICAL_TOLERANCE = 4.0

MOVE_UP_REWARD = 0.1
MOVE_DOWN_PENALTY = 0.1
MOVE_UP_ON_LADDER_REWARD = 0.1
MOVE_DOWN_ON_LADDER_PENALTY = 0.2


def _float32(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _player_center(state):
    return jnp.asarray(
        [state.player.x + 4.0, state.player.y + state.player.height * 0.5],
        dtype=jnp.float32,
    )


def _rect_center(position, size):
    position = jnp.asarray(position, dtype=jnp.float32)
    size = jnp.asarray(size, dtype=jnp.float32)
    return position + size * 0.5


def _ladder_centers(state):
    return (
        state.level.ladder_positions.astype(jnp.float32)
        + state.level.ladder_sizes.astype(jnp.float32) * 0.5
    )


def _child_center(state):
    return _rect_center(
        state.level.child_position, jnp.asarray([8.0, 15.0], dtype=jnp.float32)
    )


def _coconut_active_mask(state):
    thrown_active = state.level.coco_states != 0
    falling_active = jnp.logical_or(
        state.level.falling_coco_position[0] != 13,
        state.level.falling_coco_position[1] != -1,
    )
    return jnp.concatenate([thrown_active, jnp.asarray([falling_active])], axis=0)


def _coconut_positions(state):
    falling = state.level.falling_coco_position[jnp.newaxis, :]
    return jnp.concatenate([state.level.coco_positions, falling], axis=0).astype(
        jnp.float32
    )


def _asymmetric_ellipse_proximity(
    player_center, hazard_center, right_radius, up_radius
):
    dx = hazard_center[0] - player_center[0]
    dy = hazard_center[1] - player_center[1]
    x_radius = jnp.where(dx >= 0.0, right_radius, COCONUT_RADIUS)
    y_radius = jnp.where(dy <= 0.0, up_radius, COCONUT_RADIUS)
    ellipse_distance = (dx / x_radius) * (dx / x_radius) + (dy / y_radius) * (
        dy / y_radius
    )
    proximity = jnp.clip(1.0 - ellipse_distance, 0.0, 1.0)
    return jnp.where(
        proximity > 0.0,
        (1.0 - jnp.exp(-COCONUT_SHARPNESS * proximity))
        / (1.0 - jnp.exp(-COCONUT_SHARPNESS)),
        0.0,
    )


def _coconut_penalty(player_center, coconut_center):
    vertical_proximity = _asymmetric_ellipse_proximity(
        player_center,
        coconut_center,
        COCONUT_RADIUS,
        COCONUT_VERTICAL_RADIUS,
    )
    horizontal_proximity = _asymmetric_ellipse_proximity(
        player_center,
        coconut_center,
        COCONUT_HORIZONTAL_RADIUS,
        COCONUT_RADIUS,
    )
    return -COCONUT_MAX_PENALTY * jnp.maximum(vertical_proximity, horizontal_proximity)


def _approach_reward(previous_distance, current_distance, max_reward, max_distance):
    progress = jnp.maximum(previous_distance - current_distance, 0.0)
    return max_reward * jnp.clip(progress / max_distance, 0.0, 1.0)


def _child_distance(player_center, child_center):
    dx = jnp.abs(child_center[0] - player_center[0])
    dy = jnp.abs(child_center[1] - player_center[1])
    return jnp.where(dy <= CHILD_SIMILAR_Y_THRESHOLD, dy + CHILD_X_WEIGHT * dx, dy)


def _nearest_ladder_target(state):
    player_center = _player_center(state)
    ladder_centers = _ladder_centers(state)
    player_bottom = state.player.y.astype(jnp.float32) + state.player.height.astype(
        jnp.float32
    )
    ladder_bottoms = state.level.ladder_positions[:, 1].astype(
        jnp.float32
    ) + state.level.ladder_sizes[:, 1].astype(jnp.float32)
    valid_ladders = jnp.logical_and(
        jnp.all(state.level.ladder_positions >= 0, axis=1),
        jnp.all(state.level.ladder_sizes > 0, axis=1),
    )
    vertical_distance = jnp.where(
        jnp.logical_and(valid_ladders, ladder_bottoms <= player_bottom),
        player_bottom - ladder_bottoms,
        jnp.inf,
    )
    min_vertical_distance = jnp.min(vertical_distance)
    same_tier = (
        jnp.abs(vertical_distance - min_vertical_distance) <= LADDER_VERTICAL_TOLERANCE
    )
    horizontal_distance = jnp.abs(ladder_centers[:, 0] - player_center[0])
    distances = jnp.where(same_tier, horizontal_distance, jnp.inf)
    target_index = jnp.argmin(distances)
    target_center = ladder_centers[target_index]
    target_valid = jnp.isfinite(vertical_distance[target_index])
    return target_center, target_valid


def _ladder_distance(player_center, ladder_center):
    dx = jnp.abs(ladder_center[0] - player_center[0])
    dy = jnp.abs(ladder_center[1] - player_center[1])
    return dx + LADDER_Y_WEIGHT * dy


def reward_function(previous_state, state) -> jax.Array:
    reward = 0.0

    reward += jnp.where(state.levelup, CHILD_REACH_REWARD, 0.0)

    life_loss = jnp.maximum(previous_state.lives - state.lives, 0)
    reward -= LIFE_LOSS_PENALTY * _float32(life_loss)

    previous_fruit_count = jnp.sum(previous_state.level.fruit_actives.astype(jnp.int32))
    current_fruit_count = jnp.sum(state.level.fruit_actives.astype(jnp.int32))
    fruit_collected = jnp.maximum(previous_fruit_count - current_fruit_count, 0)
    last_fruit_collected = jnp.where(
        jnp.logical_and(previous_fruit_count > 0, current_fruit_count == 0),
        jnp.minimum(fruit_collected, 1),
        0,
    )
    reward += FRUIT_REWARD * _float32(fruit_collected)
    reward += (LAST_FRUIT_REWARD - FRUIT_REWARD) * _float32(last_fruit_collected)

    bell_triggered = jnp.logical_and(
        previous_state.level.bell_timer == 0, state.level.bell_timer > 0
    )
    reward += jnp.where(bell_triggered, BELL_REWARD, 0.0)

    player_punching = jnp.logical_or(state.player.punch_left, state.player.punch_right)
    monkey_killed = jnp.logical_and(
        player_punching,
        jnp.any(
            jnp.logical_and(
                previous_state.level.monkey_states != 0, state.level.monkey_states == 0
            )
        ),
    )
    reward += jnp.where(monkey_killed, MONKEY_KILL_REWARD, 0.0)

    same_level = jnp.logical_and(
        previous_state.levelup_timer == 0, state.levelup_timer == 0
    )

    previous_player_center = _player_center(previous_state)
    current_player_center = _player_center(state)

    child_target = _child_center(state)
    previous_child_distance = _child_distance(previous_player_center, child_target)
    current_child_distance = _child_distance(current_player_center, child_target)
    reward += jnp.where(
        same_level,
        _approach_reward(
            previous_child_distance,
            current_child_distance,
            CHILD_APPROACH_MAX_REWARD,
            CHILD_APPROACH_MAX_DISTANCE,
        ),
        0.0,
    )

    ladder_target, ladder_target_valid = _nearest_ladder_target(state)
    current_ladder_distance = _ladder_distance(current_player_center, ladder_target)
    reward += jnp.where(
        jnp.logical_and(same_level, ladder_target_valid),
        LADDER_APPROACH_MAX_REWARD
        * jnp.clip(
            1.0 - current_ladder_distance / LADDER_APPROACH_MAX_DISTANCE, 0.0, 1.0
        ),
        0.0,
    )

    reward += jnp.where(
        jnp.logical_and(
            state.player.is_climbing,
            current_player_center[1] < previous_player_center[1],
        ),
        MOVE_UP_ON_LADDER_REWARD,
        0.0,
    )
    reward -= jnp.where(
        jnp.logical_and(
            state.player.is_climbing,
            current_player_center[1] > previous_player_center[1],
        ),
        MOVE_DOWN_ON_LADDER_PENALTY,
        0.0,
    )
    reward += jnp.where(
        current_player_center[1] < previous_player_center[1], MOVE_UP_REWARD, 0.0
    )
    reward -= jnp.where(
        current_player_center[1] > previous_player_center[1], MOVE_DOWN_PENALTY, 0.0
    )

    coconut_positions = _coconut_positions(state)
    coconut_active = _coconut_active_mask(state)
    coconut_penalties = jax.vmap(
        lambda coconut_position, active: jnp.where(
            active,
            _coconut_penalty(current_player_center, coconut_position),
            0.0,
        )
    )(coconut_positions, coconut_active)
    reward += jnp.min(coconut_penalties)

    return jnp.asarray(reward, dtype=jnp.float32)
