import jax
import jax.numpy as jnp

GOAL_REWARD = 100.0
LOSE_LIFE_PENALTY = -100.0

COCONUT_MAX_PENALTY = -1.0
COCONUT_LONG_RADIUS = 110.0
COCONUT_SHORT_RADIUS = 20.0

ENEMY_REWARD = 5.0
FRUIT_REWARD = 6.0
LAST_FRUIT_REWARD = 10.0
BELL_REWARD = 7.0

CHILD_MAX_REWARD = 0.5
CHILD_HORIZONTAL_RADIUS = 100.0
CHILD_VERTICAL_RADIUS = 30.0

LADDER_MAX_REWARD = 0.2
LADDER_UP_REWARD = 0.1
LADDER_DOWN_PENALTY = -0.15

MOVE_UP_REWARD = 0.1
MOVE_DOWN_PENALTY = -0.15


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
        return jnp.asarray(0.0, dtype=jnp.float32)


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
        return jnp.asarray(0.0, dtype=jnp.float32)


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

    player_center = jnp.array(
        [
            state.player.x + 4.0,
            state.player.y
            + (jnp.asarray(state.player.height, dtype=jnp.float32) * 0.5),
        ],
        dtype=jnp.float32,
    )

    # Level completion reward
    reward += jnp.where(
        state.level_finished & ~previous_state.level_finished, GOAL_REWARD, 0.0
    )

    # Life loss penalty
    reward += jnp.where(state.lives < previous_state.lives, LOSE_LIFE_PENALTY, 0.0)

    # Coconut penalty
    thrown_cocos = jnp.asarray(state.level.coco_positions, dtype=jnp.float32)
    coco_penalty_strength = jax.vmap(
        lambda coco: _asymmetric_ellipse_exponential_distance(
            player_center,
            coco,
            COCONUT_SHORT_RADIUS,
            COCONUT_LONG_RADIUS,
            COCONUT_SHORT_RADIUS,
            COCONUT_SHORT_RADIUS,
        )
    )(thrown_cocos)
    reward += jnp.max(coco_penalty_strength) * COCONUT_MAX_PENALTY

    falling_cocos = jnp.asarray(state.level.falling_coco_position, dtype=jnp.float32)
    coco_penalty_strength = jax.vmap(
        lambda coco: _asymmetric_ellipse_exponential_distance(
            player_center,
            coco,
            COCONUT_SHORT_RADIUS,
            COCONUT_LONG_RADIUS,
            COCONUT_SHORT_RADIUS,
            COCONUT_SHORT_RADIUS,
        )
    )(falling_cocos)
    reward += jnp.max(coco_penalty_strength) * COCONUT_MAX_PENALTY

    # Enemy kill reward
    monkey_kills = jnp.sum(
        (previous_state.level.monkey_states != 0) & (state.level.monkey_states == 0)
    )
    reward += monkey_kills * ENEMY_REWARD

    # Fruit and bell rewards
    fruit_collected = previous_state.level.fruit_actives & ~state.level.fruit_actives
    fruit_reward = jnp.where(
        previous_state.level.fruit_stages == 2, LAST_FRUIT_REWARD, FRUIT_REWARD
    )
    reward += jnp.sum(jnp.where(fruit_collected, fruit_reward, 0.0))

    reward += jnp.where(
        (previous_state.level.bell_timer == 0) & (state.level.bell_timer > 0),
        BELL_REWARD,
        0.0,
    )

    # Child approach reward
    child_vertical_reward_strength = _linear_distance(
        player_center[1],
        state.level.child_position[1],
    )
    reward += child_vertical_reward_strength * CHILD_MAX_REWARD

    child_reward_strength = jax.vmap(
        lambda coco: _asymmetric_ellipse_exponential_distance(
            player_center,
            coco,
            CHILD_HORIZONTAL_RADIUS,
            CHILD_HORIZONTAL_RADIUS,
            CHILD_VERTICAL_RADIUS,
            CHILD_VERTICAL_RADIUS,
        )
    )(state.level.child_position)
    reward += jnp.max(child_reward_strength) * CHILD_MAX_REWARD

    # Ladder climbing reward
    ladder_positions = jnp.asarray(state.level.ladder_positions, dtype=jnp.float32)
    print("ladder_positions:", ladder_positions)
    ladder_sizes = jnp.asarray(state.level.ladder_sizes, dtype=jnp.float32)
    print("ladder_sizes:", ladder_sizes)
    player_bottom = state.player.y + jnp.asarray(state.player.height, dtype=jnp.float32)
    target_idx = jnp.argmin(
        jnp.where(
            ladder_positions[:, 1] >= player_bottom,
            ladder_positions[:, 1],
            jnp.inf,
        )
    )
    """target_center = ladder_positions[target_idx] + jnp.array(
        [0.0, ladder_sizes[target_idx] * 0.5],
        dtype=jnp.float32,
    )
    reward += LADDER_MAX_REWARD * _linear_distance(
        player_center,
        target_center,
    )"""

    reward += jnp.where(
        state.player.is_climbing & (state.player.y < previous_state.player.y),
        LADDER_UP_REWARD,
        0.0,
    )
    reward += jnp.where(
        state.player.is_climbing & (state.player.y > previous_state.player.y),
        LADDER_DOWN_PENALTY,
        0.0,
    )

    # Horizontal movement reward
    reward += jnp.where(state.player.y < previous_state.player.y, MOVE_UP_REWARD, 0.0)
    reward += jnp.where(
        state.player.y > previous_state.player.y, MOVE_DOWN_PENALTY, 0.0
    )

    return jnp.asarray(reward, dtype=jnp.float32)
