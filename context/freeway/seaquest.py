import chex
from flax import struct


@struct.dataclass
class SpawnState:
    difficulty: chex.Array  # Current difficulty level (0-7)
    lane_dependent_pattern: chex.Array  # Track waves independently per lane [4 lanes]
    to_be_spawned: (
        chex.Array
    )  # tracks which enemies are still in the spawning cycle [4 lanes * 3 slots] -> necessary due to the spaced out spawning of multiple enemies
    survived: (
        chex.Array
    )  # track if last enemy survived [4 lanes * 3 slots] -> 1 if survived whilst going right, 0 if not, -1 if survived whilst going left
    prev_sub: chex.Array  # Track previous entity type for each lane [4 lanes]
    spawn_timers: chex.Array  # Individual spawn timers per lane [4 lanes]
    diver_array: (
        chex.Array
    )  # Track which divers are still in the spawning cycle [4 lanes]
    lane_directions: (
        chex.Array
    )  # Track lane directions for each wave [4 lanes] -> 0 = right, 1 = left


# Game state container
@struct.dataclass
class SeaquestState:
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0 for right, 1 for left
    oxygen: chex.Array
    divers_collected: chex.Array
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    diver_positions: chex.Array  # (4, 3) array for divers
    shark_positions: (
        chex.Array
    )  # (12, 3) array for sharks - separated into 4 lanes, 3 slots per lane [left to right]
    sub_positions: (
        chex.Array
    )  # (12, 3) array for enemy subs - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_missile_positions: (
        chex.Array
    )  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_sub_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: (
        chex.Array
    )  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment
    successful_rescues: (
        chex.Array
    )  # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation
    rng_key: chex.PRNGKey
