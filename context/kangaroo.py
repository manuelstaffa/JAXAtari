import chex
from flax import struct


@struct.dataclass
class PlayerState:
    # Player position
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    orientation: chex.Array
    height: chex.Array
    # crouching
    is_crouching: chex.Array
    # jumping
    is_jumping: chex.Array
    jump_base_y: chex.Array
    jump_counter: chex.Array
    jump_orientation: chex.Array
    landing_base_y: chex.Array
    # climbing
    is_climbing: chex.Array
    climb_base_y: chex.Array
    climb_counter: chex.Array
    cooldown_counter: chex.Array
    # other
    is_crashing: chex.Array
    chrash_timer: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array
    last_stood_on_platform_y: chex.Array
    walk_animation: chex.Array
    punch_counter: chex.Array  # New field to track consecutive punches
    needs_release: chex.Array  # New field to track if spacebar needs to be released


@struct.dataclass
class LevelState:
    """All level related state variables."""

    timer: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    fruit_positions: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    bell_position: chex.Array
    bell_timer: chex.Array
    child_position: chex.Array
    child_velocity: chex.Array
    child_timer: chex.Array
    falling_coco_position: chex.Array
    falling_coco_dropping: chex.Array
    falling_coco_counter: chex.Array
    falling_coco_skip_update: chex.Array
    step_counter: chex.Array
    monkey_states: chex.Array
    """
    - 0: non-existent
    - 1: moving down
    - 2: moving left
    - 3: throwing
    - 4: moving right
    - 5: moving up
    """
    monkey_positions: chex.Array
    """2D array: [monkey_index, [x, y]]"""
    monkey_throw_timers: chex.Array
    spawn_protection: chex.Array
    coco_positions: chex.Array
    coco_states: chex.Array
    """
    - 0: non existent
    - 1: charging
    - 2: throwing
    """
    spawn_position: chex.Array
    """
    - 0: foot
    - 1: head
    """
    bell_animation: chex.Array


@struct.dataclass
class KangarooState:
    player: PlayerState
    level: LevelState
    score: chex.Array
    current_level: chex.Array
    level_finished: chex.Array
    levelup_timer: chex.Array
    reset_coords: chex.Array
    levelup: chex.Array
    lives: chex.Array
