import jax
import jax.numpy as jnp

SCORE_REWARD = 100.0
MOVE_UP_REWARD = 0.1
MOVE_DOWN_PENALTY = 0.2
PROXIMITY_MAX_PENALTY = 1.0
PROXIMITY_HORIZONTAL_RADIUS = 40.0
PROXIMITY_FRONT_RADIUS = 16.0
PROXIMITY_BACK_RADIUS = 8.0
PROXIMITY_SHARPNESS = 5.0
PLAYER_X = 44.0
PLAYER_WIDTH = 6.0
PLAYER_HEIGHT = 8.0
CAR_WIDTH = 8.0
CAR_HEIGHT = 10.0


def _player_center_x():
    return jnp.asarray(PLAYER_X + PLAYER_WIDTH * 0.5, dtype=jnp.float32)


def _player_center_y(chicken_y):
    return jnp.asarray(chicken_y, dtype=jnp.float32) - PLAYER_HEIGHT * 0.5


def _car_center_x(car_x):
    return jnp.asarray(car_x, dtype=jnp.float32) + CAR_WIDTH * 0.5


def _car_center_y(car_y):
    return jnp.asarray(car_y, dtype=jnp.float32) - CAR_HEIGHT * 0.5


def _car_penalty(chicken_y, car):
    dx = jnp.abs(_car_center_x(car[0]) - _player_center_x())
    dy = _car_center_y(car[1]) - _player_center_y(chicken_y)
    horizontal_term = dx / PROXIMITY_HORIZONTAL_RADIUS
    vertical_radius = jnp.where(dy < 0, PROXIMITY_FRONT_RADIUS, PROXIMITY_BACK_RADIUS)
    vertical_term = jnp.abs(dy) / vertical_radius
    ellipse_distance = horizontal_term * horizontal_term + vertical_term * vertical_term
    proximity = jnp.clip(1.0 - ellipse_distance, 0.0, 1.0)
    shaped = jnp.where(
        proximity > 0,
        (1.0 - jnp.exp(-PROXIMITY_SHARPNESS * proximity))
        / (1.0 - jnp.exp(-PROXIMITY_SHARPNESS)),
        0.0,
    )
    return -PROXIMITY_MAX_PENALTY * shaped


def reward_function(previous_state, state) -> jax.Array:
    reward = 0.0
    score_delta = jnp.asarray(state.score - previous_state.score, dtype=jnp.float32)
    reward += SCORE_REWARD * score_delta
    reward += jnp.where(score_delta > 0, MOVE_UP_REWARD, 0.0)
    reward += jnp.where(
        jnp.logical_and(score_delta <= 0, state.chicken_y < previous_state.chicken_y),
        MOVE_UP_REWARD,
        0.0,
    )
    reward -= jnp.where(
        jnp.logical_and(score_delta <= 0, state.chicken_y > previous_state.chicken_y),
        MOVE_DOWN_PENALTY,
        0.0,
    )
    penalties = jax.vmap(lambda car: _car_penalty(state.chicken_y, car))(state.cars)
    reward += jnp.min(penalties)

    return jnp.asarray(reward, dtype=jnp.float32)
