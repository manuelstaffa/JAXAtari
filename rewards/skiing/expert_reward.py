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

SKIER_Y = 46.0

ALIGNMENT_X_RADIUS = 16.0
ALIGNMENT_Y_RADIUS = 60.0
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

# replace above with this?
HAZARD_UP_RADIUS = 80.0
HAZARD_DOWN_RADIUS = 20.0
HAZARD_HORIZONTAL_RADIUS = 35.0


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

    skier_center = jnp.array([state.skier_x, jnp.float32(SKIER_Y)], dtype=jnp.float32)

    gate_x = jnp.asarray(state.flags[:, 0], dtype=jnp.float32)
    gate_y = jnp.asarray(state.flags[:, 1], dtype=jnp.float32)
    active_mask = gate_y >= jnp.float32(-15.0)
    upcoming_mask = jnp.logical_and(
        active_mask,
        jnp.logical_and(
            gate_y >= jnp.float32(SKIER_Y), jnp.logical_not(state.flags_passed)
        ),
    )
    use_upcoming = jnp.any(upcoming_mask)
    target_mask = jnp.where(use_upcoming, upcoming_mask, active_mask)
    gate_distances = jnp.where(
        target_mask,
        jnp.where(
            upcoming_mask,
            gate_y - jnp.float32(SKIER_Y),
            jnp.abs(gate_y - jnp.float32(SKIER_Y)),
        ),
        jnp.inf,
    )
    target_index = jnp.argmin(gate_distances)
    target_gate = jnp.array(
        [gate_x[target_index], gate_y[target_index]], dtype=jnp.float32
    )
    alignment = _asymmetric_ellipse_exponential_distance(
        skier_center,
        target_gate,
        ALIGNMENT_X_RADIUS,
        ALIGNMENT_X_RADIUS,
        ALIGNMENT_Y_RADIUS,
        ALIGNMENT_Y_RADIUS,
    )
    reward += jnp.where(jnp.any(target_mask), ALIGNMENT_MAX_REWARD * alignment, 0.0)

    forward_progress = jnp.clip(
        jnp.asarray(state.skier_y_speed, dtype=jnp.float32)
        / jnp.float32(FORWARD_SPEED_SCALE),
        0.0,
        1.0,
    )
    reward += FORWARD_PROGRESS_MAX_REWARD * forward_progress

    direction_delta = jnp.maximum(
        jnp.asarray(
            state.direction_change_counter - previous_state.direction_change_counter,
            dtype=jnp.float32,
        ),
        0.0,
    )
    reward += ZIGZAG_MAX_PENALTY * jnp.clip(
        direction_delta / jnp.float32(ZIGZAG_SCALE), 0.0, 1.0
    )

    tree_centers = jnp.stack(
        [
            jnp.asarray(state.trees[:, 0], dtype=jnp.float32),
            jnp.asarray(state.trees[:, 1], dtype=jnp.float32),
        ],
        axis=1,
    )
    tree_proximity = jax.vmap(
        lambda tree: _asymmetric_ellipse_exponential_distance(
            skier_center,
            tree,
            TREE_HAZARD_X_RADIUS,
            TREE_HAZARD_X_RADIUS,
            TREE_HAZARD_FORWARD_Y_RADIUS,
            TREE_HAZARD_BACKWARD_Y_RADIUS,
        )
    )(tree_centers)

    mogul_centers = jnp.stack(
        [
            jnp.asarray(state.moguls[:, 0], dtype=jnp.float32),
            jnp.asarray(state.moguls[:, 1], dtype=jnp.float32),
        ],
        axis=1,
    )
    mogul_proximity = jax.vmap(
        lambda mogul: _asymmetric_ellipse_exponential_distance(
            skier_center,
            mogul,
            MOGUL_HAZARD_X_RADIUS,
            MOGUL_HAZARD_X_RADIUS,
            MOGUL_HAZARD_FORWARD_Y_RADIUS,
            MOGUL_HAZARD_BACKWARD_Y_RADIUS,
        )
    )(mogul_centers)

    hazard_score = jnp.clip(
        jnp.max(tree_proximity) + jnp.max(mogul_proximity), 0.0, 1.0
    )
    reward += HAZARD_MAX_PENALTY * hazard_score

    nearby_mogul = jnp.max(
        jax.vmap(
            lambda mogul: _asymmetric_ellipse_exponential_distance(
                skier_center,
                mogul,
                JUMP_MOGUL_X_RADIUS,
                JUMP_MOGUL_X_RADIUS,
                JUMP_MOGUL_FORWARD_Y_RADIUS,
                JUMP_MOGUL_BACKWARD_Y_RADIUS,
            )
        )(mogul_centers)
    )
    clear_jump = jnp.logical_and(
        state.is_jumping, jnp.logical_not(state.collision_type == 2)
    )
    reward += (
        JUMP_MAX_REWARD * nearby_mogul * jnp.asarray(clear_jump, dtype=jnp.float32)
    )

    return jnp.asarray(reward, dtype=jnp.float32)
