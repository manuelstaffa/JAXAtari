import chex
from flax import struct


@struct.dataclass
class FreewayState:
    """Represents the current state of the game"""

    chicken_y: chex.Array
    cars: (
        chex.Array
    )  # Shape: (num_lanes, 2) for x,y positions (ints for render/collide)
    # Per-lane cadence counters (frames), advance independently to sync movement patterns per lane
    lane_time: chex.Array
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    walking_frames: chex.Array
    game_over: chex.Array
