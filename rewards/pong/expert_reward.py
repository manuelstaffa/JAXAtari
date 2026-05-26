import jax.numpy as jnp
import jax

SCREEN_WIDTH = 160.0
SCORE_REWARD = 10.0
BALL_ON_ENEMY_HALF_REWARD = 0.1
BALL_ON_PLAYER_HALF_PENALTY = -0.1
PLAYER_BALL_PROXIMITY_RADIUS = 54.0
PLAYER_BALL_PROXIMITY_SHARPNESS = 4.0
PLAYER_BALL_PROXIMITY_MAX_REWARD = 1.0


def _center_x(x, width):
    return (
        jnp.asarray(x, dtype=jnp.float32) + jnp.asarray(width, dtype=jnp.float32) * 0.5
    )


def _center_y(y, height):
    return (
        jnp.asarray(y, dtype=jnp.float32) + jnp.asarray(height, dtype=jnp.float32) * 0.5
    )


def _player_ball_distance(state):
    player_x = _center_x(140.0, 4.0)
    player_y = _center_y(state.player_y, 16.0)
    ball_x = _center_x(state.ball_x, 2.0)
    ball_y = _center_y(state.ball_y, 4.0)
    delta_x = ball_x - player_x
    delta_y = ball_y - player_y
    return jnp.sqrt(delta_x * delta_x + delta_y * delta_y)


def _proximity_reward(distance):
    scaled_distance = jnp.clip(distance / PLAYER_BALL_PROXIMITY_RADIUS, 0.0, 1.0)
    shaped = (
        1.0 - jnp.exp(-PLAYER_BALL_PROXIMITY_SHARPNESS * (1.0 - scaled_distance))
    ) / (1.0 - jnp.exp(-PLAYER_BALL_PROXIMITY_SHARPNESS))
    return PLAYER_BALL_PROXIMITY_MAX_REWARD * shaped


def reward_function(previous_state, state) -> jax.Array:
    reward = 0.0

    player_score_delta = jnp.asarray(
        state.player_score - previous_state.player_score, dtype=jnp.float32
    )
    enemy_score_delta = jnp.asarray(
        state.enemy_score - previous_state.enemy_score, dtype=jnp.float32
    )
    reward += SCORE_REWARD * player_score_delta
    reward -= SCORE_REWARD * enemy_score_delta

    ball_on_enemy_half = (
        jnp.asarray(state.ball_x, dtype=jnp.float32) < SCREEN_WIDTH * 0.5
    )
    reward += jnp.where(ball_on_enemy_half, BALL_ON_ENEMY_HALF_REWARD, 0.0)

    ball_on_player_half = jnp.logical_not(ball_on_enemy_half)
    player_ball_distance = _player_ball_distance(state)
    close_to_ball = player_ball_distance <= PLAYER_BALL_PROXIMITY_RADIUS
    reward += jnp.where(
        ball_on_player_half,
        jnp.where(
            close_to_ball,
            _proximity_reward(player_ball_distance),
            BALL_ON_PLAYER_HALF_PENALTY,
        ),
        0.0,
    )

    return jnp.asarray(reward, dtype=jnp.float32)
