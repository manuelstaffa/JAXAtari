def reward_function(previous_state, state) -> float:
    reward = 0.0

    reward += state.chicken_y
    reward += (state.score - previous_state.score) * 1000

    return reward
