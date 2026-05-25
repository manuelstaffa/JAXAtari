import chex
from flax import struct


@struct.dataclass
class SkiingState:
    """Represents the current state of the game"""

    skier_x: chex.Array
    skier_pos: (
        chex.Array
    )  # --> --_  \  |  |   | |  /  _-- <-- States are doubles in ALE (9 total)
    skier_fell: chex.Array
    skier_x_speed: chex.Array
    skier_y_speed: chex.Array
    flags: chex.Array
    trees: chex.Array
    moguls: chex.Array
    successful_gates: chex.Array
    step_count: chex.Array
    direction_change_counter: chex.Array
    game_over: chex.Array
    key: chex.Array
    collision_type: chex.Array  # 0 = none, 1 = tree, 2 = mogul, 3 = flag
    flags_passed: chex.Array
    collision_cooldown: (
        chex.Array
    )  # Frames where collisions are ignored (Debounce after recovery)
    skier_just_respawned: (
        chex.Array
    )  # Boolean indicating if skier is in post-recovery immunity
    jump_timer: chex.Array  # Timer for jump duration and cooldown
    is_jumping: chex.Array  # Boolean indicating if the skier is currently jumping
    gates_seen: chex.Array  # Number of already processed gates (despawned)
