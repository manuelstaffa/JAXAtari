import chex
from flax import struct
from typing import NamedTuple
import jax.numpy as jnp


class LevelState(NamedTuple):
    id: chex.Array  # Int - Number of the current level, starts at 1
    collected_pellets: chex.Array  # Int - Number of collected pellets
    pellets: chex.Array  # Bool[x][y] - 2D grid of 0 (empty) or 1 (pellet)
    power_pellets: (
        chex.Array
    )  # Bool[4] - Indicates wheter the power pellet is available
    loaded: chex.Array  # Int - 0: Not loaded, 1: loading, 2: loaded


class GhostsState(NamedTuple):
    positions: chex.Array  # Tuple - (x, y)
    types: chex.Array  # Enum - 0: BLINKY, 1: PINKY, 2: INKY, 3: SUE
    actions: chex.Array  # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    modes: (
        chex.Array
    )  # Enum - 0: RANDOM, 1: CHASE, 2: SCTATTER, 3: FRIGHTENED, 4: BLINKING, 5: RETURNING, 6: ENJAILED
    timers: (
        chex.Array
    )  # Int - Triggers mode change when reaching 0, decrements every step


class PlayerState(NamedTuple):
    position: chex.Array  # Tuple - (x, y)
    action: chex.Array  # Enum - 0: NOOP, 1: FURE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    has_pellet: chex.Array  # Bool - Indicates if pacman just collected a pellet
    eaten_ghosts: (
        chex.Array
    )  # Int - Indicates the number of ghosts eaten since the last power pellet
    last_horiz_dir: chex.Array = jnp.array(
        2, dtype=jnp.int32
    )  # Default LEFT (2 in act_to_dir)


class FruitState(NamedTuple):
    position: chex.Array  # Tuple - (x, y)
    exit: chex.Array  # Tuple - (x, y) Position of the tunnel through which it will exit
    type: (
        chex.Array
    )  # Enum - 0: CHERRY, 1: STRAWBERRY, 2: ORANGE, 3: PRETZEL, 4: APPLE, 5: PEAR, 6: BANANA, 7: NONE
    action: chex.Array  # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    spawn: (
        chex.Array
    )  # Bool - Indicates wether a fruit should spawn into the maze as soon as possible
    spawned: (
        chex.Array
    )  # Bool - Indicates wether a fruit is currently present within the maze
    timer: (
        chex.Array
    )  # Int - Time until leaving through the exit tunnel, decrements every step


@struct.dataclass
class PacmanState:
    level: LevelState  # LevelState
    player: PlayerState  # PlayerState
    ghosts: GhostsState  # GhostStates
    fruit: FruitState  # FruitState
    lives: chex.Array  # Int - Number of lives left
    score: chex.Array  # Int - Total score reached
    score_changed: (
        chex.Array
    )  # Bool[] - Indicates which score digit changed since the last step
    freeze_timer: chex.Array  # Int - Time until game is unfrozen, decrements every step
    step_count: chex.Array  # Int - Number of steps made in the current level
    key: chex.PRNGKey  # PRNGKey for RNG during step
