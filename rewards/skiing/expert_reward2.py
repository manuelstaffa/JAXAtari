import jax
import jax.numpy as jnp

PASS_GATE_REWARD = 20.0
MISS_GATE_PENALTY = -10.0
TREE_OR_FLAG_COLLISION_PENALTY = -25.0
MOGUL_COLLISION_PENALTY = -10.0
FALL_PENALTY = -100.0
GAME_OVER_PENALTY = -200.0

ALIGNMENT_MAX_REWARD = 1.0
FORWARD_PROGRESS_MAX_REWARD = 0.5
ZIGZAG_MAX_PENALTY = -0.5
HAZARD_MAX_PENALTY = -1.0
JUMP_MAX_REWARD = 0.5

SKIER_X_WIDTH = 10.0
SKIER_Y = 46.0
SKIER_Y_HEIGHT = 18.0
FLAG_CENTER_OFFSET = 16.0
TREE_CENTER_OFFSET_X = 8.0
TREE_CENTER_OFFSET_Y = 15.0
MOGUL_CENTER_OFFSET_X = 8.0
MOGUL_CENTER_OFFSET_Y = 3.5

ALIGNMENT_RADIUS = 16.0
FORWARD_SPEED_SCALE = 1.2
ZIGZAG_SCALE = 4.0
TREE_HAZARD_X_RADIUS = 28.0
TREE_HAZARD_FORWARD_Y_RADIUS = 128.0
TREE_HAZARD_BACKWARD_Y_RADIUS = 18.0
MOGUL_HAZARD_X_RADIUS = 22.0
MOGUL_HAZARD_FORWARD_Y_RADIUS = 88.0
MOGUL_HAZARD_BACKWARD_Y_RADIUS = 14.0
JUMP_MOGUL_X_RADIUS = 24.0
JUMP_MOGUL_FORWARD_Y_RADIUS = 92.0
JUMP_MOGUL_BACKWARD_Y_RADIUS = 16.0


def _float32(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _center_x(x, width):
    return _float32(x) + _float32(width) * 0.5


def _center_y(y, height):
    return _float32(y) + _float32(height) * 0.5


def _directional_proximity(dx, dy, x_radius, forward_y_radius, backward_y_radius):
    y_radius = jnp.where(
        dy >= 0.0, _float32(forward_y_radius), _float32(backward_y_radius)
    )
    ellipse_distance = (dx / _float32(x_radius)) ** 2 + (dy / y_radius) ** 2
    return jnp.clip(1.0 - ellipse_distance, 0.0, 1.0)


def _nearest_gate_alignment(state):
    skier_center_x = _center_x(state.skier_x, SKIER_X_WIDTH)
    gate_centers_x = _float32(state.flags[:, 0]) + _float32(FLAG_CENTER_OFFSET)
    gate_y = _float32(state.flags[:, 1])
    active_mask = gate_y >= _float32(-15.0)
    upcoming_mask = jnp.logical_and(
        active_mask,
        jnp.logical_and(
            gate_y >= _float32(SKIER_Y), jnp.logical_not(state.flags_passed)
        ),
    )
    use_upcoming = jnp.any(upcoming_mask)
    target_mask = jnp.where(use_upcoming, upcoming_mask, active_mask)
    gate_distances = jnp.where(
        target_mask,
        jnp.where(
            upcoming_mask,
            gate_y - _float32(SKIER_Y),
            jnp.abs(gate_y - _float32(SKIER_Y)),
        ),
        jnp.inf,
    )
    target_index = jnp.argmin(gate_distances)
    target_center_x = gate_centers_x[target_index]
    alignment = 1.0 - jnp.clip(
        jnp.abs(skier_center_x - target_center_x) / _float32(ALIGNMENT_RADIUS), 0.0, 1.0
    )
    return jnp.where(jnp.any(target_mask), alignment, 0.0)


def _nearest_hazard_penalty(state):
    skier_center_x = _center_x(state.skier_x, SKIER_X_WIDTH)
    skier_center_y = _center_y(SKIER_Y, SKIER_Y_HEIGHT)

    tree_centers_x = _float32(state.trees[:, 0]) + _float32(TREE_CENTER_OFFSET_X)
    tree_centers_y = _float32(state.trees[:, 1]) + _float32(TREE_CENTER_OFFSET_Y)
    tree_dx = tree_centers_x - skier_center_x
    tree_dy = tree_centers_y - skier_center_y
    tree_proximity = _directional_proximity(
        tree_dx,
        tree_dy,
        TREE_HAZARD_X_RADIUS,
        TREE_HAZARD_FORWARD_Y_RADIUS,
        TREE_HAZARD_BACKWARD_Y_RADIUS,
    )

    mogul_centers_x = _float32(state.moguls[:, 0]) + _float32(MOGUL_CENTER_OFFSET_X)
    mogul_centers_y = _float32(state.moguls[:, 1]) + _float32(MOGUL_CENTER_OFFSET_Y)
    mogul_dx = mogul_centers_x - skier_center_x
    mogul_dy = mogul_centers_y - skier_center_y
    mogul_proximity = _directional_proximity(
        mogul_dx,
        mogul_dy,
        MOGUL_HAZARD_X_RADIUS,
        MOGUL_HAZARD_FORWARD_Y_RADIUS,
        MOGUL_HAZARD_BACKWARD_Y_RADIUS,
    )

    hazard_score = jnp.clip(
        jnp.max(tree_proximity) + jnp.max(mogul_proximity), 0.0, 1.0
    )
    return HAZARD_MAX_PENALTY * hazard_score


def _jump_over_mogul_reward(state):
    skier_center_x = _center_x(state.skier_x, SKIER_X_WIDTH)
    skier_center_y = _center_y(SKIER_Y, SKIER_Y_HEIGHT)
    mogul_centers_x = _float32(state.moguls[:, 0]) + _float32(MOGUL_CENTER_OFFSET_X)
    mogul_centers_y = _float32(state.moguls[:, 1]) + _float32(MOGUL_CENTER_OFFSET_Y)
    mogul_dx = mogul_centers_x - skier_center_x
    mogul_dy = mogul_centers_y - skier_center_y
    mogul_proximity = _directional_proximity(
        mogul_dx,
        mogul_dy,
        JUMP_MOGUL_X_RADIUS,
        JUMP_MOGUL_FORWARD_Y_RADIUS,
        JUMP_MOGUL_BACKWARD_Y_RADIUS,
    )
    nearby_mogul = jnp.max(mogul_proximity)
    clear_jump = jnp.logical_and(
        state.is_jumping, jnp.logical_not(state.collision_type == 2)
    )
    return JUMP_MAX_REWARD * nearby_mogul * _float32(clear_jump)


def reward_function(previous_state, state) -> jax.Array:
    reward = 0.0

    gate_successes = jnp.maximum(
        previous_state.successful_gates - state.successful_gates, 0.0
    )
    gates_seen_delta = jnp.maximum(state.gates_seen - previous_state.gates_seen, 0.0)
    missed_gates = jnp.maximum(gates_seen_delta - gate_successes, 0.0)
    reward += PASS_GATE_REWARD * gate_successes
    reward += MISS_GATE_PENALTY * missed_gates

    tree_or_flag_collision = jnp.logical_and(
        previous_state.collision_type == 0,
        jnp.logical_or(state.collision_type == 1, state.collision_type == 3),
    )
    mogul_collision = jnp.logical_and(
        previous_state.collision_type == 0,
        jnp.logical_and(state.collision_type == 2, jnp.logical_not(state.is_jumping)),
    )
    reward += jnp.where(tree_or_flag_collision, TREE_OR_FLAG_COLLISION_PENALTY, 0.0)
    reward += jnp.where(mogul_collision, MOGUL_COLLISION_PENALTY, 0.0)

    fell = jnp.logical_and(previous_state.skier_fell == 0, state.skier_fell > 0)
    reward += jnp.where(fell, FALL_PENALTY, 0.0)
    reward += jnp.where(
        jnp.logical_and(jnp.logical_not(previous_state.game_over), state.game_over),
        GAME_OVER_PENALTY,
        0.0,
    )

    reward += ALIGNMENT_MAX_REWARD * _nearest_gate_alignment(state)

    forward_progress = jnp.clip(
        _float32(state.skier_y_speed) / _float32(FORWARD_SPEED_SCALE), 0.0, 1.0
    )
    reward += FORWARD_PROGRESS_MAX_REWARD * forward_progress

    direction_delta = jnp.maximum(
        _float32(
            state.direction_change_counter - previous_state.direction_change_counter
        ),
        0.0,
    )
    reward += ZIGZAG_MAX_PENALTY * jnp.clip(
        direction_delta / _float32(ZIGZAG_SCALE), 0.0, 1.0
    )

    reward += _nearest_hazard_penalty(state)
    reward += _jump_over_mogul_reward(state)

    return jnp.asarray(reward, dtype=jnp.float32)
