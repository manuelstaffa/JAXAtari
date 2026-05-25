def reward_function(previous_state, state) -> float:
    """Return the Freeway player Y position as the reward.

    Freeway exposes the player position as `chicken_y` on the game state.
    """
    reward = 0.0

    reward += state.chicken_y
    reward += (state.score - previous_state.score) * 1000

    return reward
