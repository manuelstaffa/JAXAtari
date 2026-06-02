import jax
import jax.numpy as jnp

SKIER_Y = 46.0

PASS_GATE_REWARD = 20.0
MISS_GATE_PENALTY = -10.0
TREE_FLAG_COLLISION_PENALTY = -25.0
MOGUL_COLLISION_PENALTY = -10.0
FALL_PENALTY = -100.0
GAME_OVER_PENALTY = -200.0

FLAG_MAX_REWARD = 1.0
PROGRESS_MAX_REWARD = 0.5
ZIGZAG_MAX_PENALTY = -0.5
HAZARD_MAX_PENALTY = -1.0
JUMP_MAX_REWARD = 0.5

FLAG_HORIZONTAL_RADIUS = 16.0
FLAG_Y_RADIUS = 60.0
HAZARD_UP_RADIUS = 40.0
HAZARD_DOWN_RADIUS = 100.0
HAZARD_HORIZONTAL_RADIUS = 40.0
JUMP_UP_RADIUS = 40.0
JUMP_DOWN_RADIUS = 80.0
JUMP_HORIZONTAL_RADIUS = 24.0


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

    # Gate rewards and penalties
    gate_successes = jnp.maximum(
        previous_state.successful_gates - state.successful_gates, 0.0
    )
    gates_seen_delta = jnp.maximum(state.gates_seen - previous_state.gates_seen, 0.0)
    missed_gates = jnp.maximum(gates_seen_delta - gate_successes, 0.0)
    reward += PASS_GATE_REWARD * gate_successes
    reward += MISS_GATE_PENALTY * missed_gates

    # Collisions penalties
    tree_or_flag_collision = jnp.logical_and(
        previous_state.collision_type == 0,
        jnp.logical_or(state.collision_type == 1, state.collision_type == 3),
    )
    collision = jnp.logical_and(
        previous_state.collision_type == 0,
        jnp.logical_and(state.collision_type == 2, jnp.logical_not(state.is_jumping)),
    )
    reward += jnp.where(tree_or_flag_collision, TREE_FLAG_COLLISION_PENALTY, 0.0)
    reward += jnp.where(collision, MOGUL_COLLISION_PENALTY, 0.0)

    # Falling and game over penalties
    fell = jnp.logical_and(previous_state.skier_fell == 0, state.skier_fell > 0)
    reward += jnp.where(fell, FALL_PENALTY, 0.0)
    reward += jnp.where(
        jnp.logical_and(jnp.logical_not(previous_state.game_over), state.game_over),
        GAME_OVER_PENALTY,
        0.0,
    )

    skier_center = jnp.array([state.skier_x, jnp.float32(SKIER_Y)], dtype=jnp.float32)

    # Flag approach reward
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
    FLAG = _asymmetric_ellipse_exponential_distance(
        skier_center,
        target_gate,
        FLAG_HORIZONTAL_RADIUS,
        FLAG_HORIZONTAL_RADIUS,
        FLAG_Y_RADIUS,
        FLAG_Y_RADIUS,
    )
    reward += jnp.where(jnp.any(target_mask), FLAG_MAX_REWARD * FLAG, 0.0)

    # Forward progress reward
    forward_progress = jnp.clip(
        jnp.asarray(state.skier_y_speed, dtype=jnp.float32),
        0.0,
        1.0,
    )
    reward += PROGRESS_MAX_REWARD * forward_progress

    # Zig-zag penalty
    direction_delta = jnp.maximum(
        jnp.asarray(
            state.direction_change_counter - previous_state.direction_change_counter,
            dtype=jnp.float32,
        ),
        0.0,
    )
    reward += ZIGZAG_MAX_PENALTY * jnp.clip(direction_delta, 0.0, 1.0)

    # Hazard proximity penalty
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
            HAZARD_HORIZONTAL_RADIUS,
            HAZARD_HORIZONTAL_RADIUS,
            HAZARD_UP_RADIUS,
            HAZARD_DOWN_RADIUS,
        )
    )(tree_centers)

    centers = jnp.stack(
        [
            jnp.asarray(state.moguls[:, 0], dtype=jnp.float32),
            jnp.asarray(state.moguls[:, 1], dtype=jnp.float32),
        ],
        axis=1,
    )
    proximity = jax.vmap(
        lambda mogul: _asymmetric_ellipse_exponential_distance(
            skier_center,
            mogul,
            HAZARD_HORIZONTAL_RADIUS,
            HAZARD_HORIZONTAL_RADIUS,
            HAZARD_UP_RADIUS,
            HAZARD_DOWN_RADIUS,
        )
    )(centers)

    hazard_score = jnp.clip(jnp.max(tree_proximity) + jnp.max(proximity), 0.0, 1.0)
    reward += HAZARD_MAX_PENALTY * hazard_score

    # Jump reward
    nearby_mogul = jnp.max(
        jax.vmap(
            lambda mogul: _asymmetric_ellipse_exponential_distance(
                skier_center,
                mogul,
                JUMP_HORIZONTAL_RADIUS,
                JUMP_HORIZONTAL_RADIUS,
                JUMP_UP_RADIUS,
                JUMP_DOWN_RADIUS,
            )
        )(centers)
    )
    clear_jump = jnp.logical_and(
        state.is_jumping, jnp.logical_not(state.collision_type == 2)
    )
    reward += (
        JUMP_MAX_REWARD * nearby_mogul * jnp.asarray(clear_jump, dtype=jnp.float32)
    )

    return jnp.asarray(reward, dtype=jnp.float32)
