"""Run a JAXAtari environment from the command line.

The CLI follows the repository README flow:
base env -> optional reward override -> human control or random autoplay.
"""

from __future__ import annotations

import functools
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pygame

from jaxatari.core import make as jaxatari_make
from jaxatari.wrappers import JaxatariWrapper

from scripts.utils import get_human_action, update_pygame

UPSCALE_FACTOR = 8


def _normalize_items(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None

    normalized: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                normalized.append(part)
    return normalized or None


def _resolve_reward_function(spec: str | None) -> Callable | None:
    if spec is None:
        return None

    reward_path = Path(spec).expanduser().resolve()
    if reward_path.suffix != ".py":
        raise ValueError(
            f"Reward function must be a Python file path ending in .py, got {spec!r}"
        )
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward function file not found: {reward_path}")

    module_name = f"jaxatari_reward_{reward_path.stem}_{abs(hash(str(reward_path)))}"
    module_spec = importlib.util.spec_from_file_location(module_name, reward_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load reward function module from {reward_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    for function_name in ("reward_function", "reward", "compute_reward", "get_reward"):
        reward_function = getattr(module, function_name, None)
        if callable(reward_function):
            return reward_function

    raise AttributeError(
        f"No callable reward function found in {reward_path}. "
        "Expected one of: reward_function, reward, compute_reward, get_reward."
    )


def _reward_value(reward) -> float:
    if hasattr(reward, "item"):
        return float(reward.item())
    return float(reward)


def _print_reward_update(previous_reward: float | None, count: int) -> None:
    if previous_reward is None or count <= 0:
        return
    print(f"reward={previous_reward} [{count} time(s)]")


def _update_reward_print_state(
    reward_value: float,
    mode: str,
    previous_reward: float | None,
    count: int,
) -> tuple[float | None, int]:
    if mode == "all":
        print(f"reward={reward_value}")
        return previous_reward, count

    if mode != "update":
        return previous_reward, count

    if previous_reward is None:
        return reward_value, 1

    if reward_value == previous_reward:
        return previous_reward, count + 1

    _print_reward_update(previous_reward, count)
    return reward_value, 1


def _flush_reward_print_state(
    mode: str, previous_reward: float | None, count: int
) -> None:
    if mode == "update":
        _print_reward_update(previous_reward, count)


class RewardOverrideWrapper(JaxatariWrapper):
    """Replace the base environment reward with a custom callable."""

    def __init__(self, env, reward_function: Callable):
        super().__init__(env)
        self._reward_function = reward_function

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, new_state, reward, done, info = self._env.step(state, action)
        custom_reward = self._reward_function(state, new_state)

        if hasattr(info, "_asdict"):
            info = info._asdict()
        elif hasattr(info, "__dict__"):
            info = vars(info)

        if isinstance(info, dict):
            info["base_reward"] = reward
            info["custom_reward"] = custom_reward

        return obs, new_state, custom_reward, done, info


def _apply_reward_function(env, reward_function: Callable | None):
    if reward_function is None:
        return env
    return RewardOverrideWrapper(env, reward_function)


def _map_action_to_index(env, action_constant):
    if hasattr(env, "ACTION_SET"):
        action_set = np.array(env.ACTION_SET)
        action_int = int(action_constant)
        matches = np.where(action_set == action_int)[0]
        if len(matches) > 0:
            return jnp.array(int(matches[0]), dtype=jnp.int32)
        return jnp.array(0, dtype=jnp.int32)
    return jnp.array(action_constant, dtype=jnp.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JAXAtari game.")
    parser.add_argument("-g", "--game", required=True, type=str, help="Game name.")
    parser.add_argument(
        "-rf",
        "--reward_function",
        type=str,
        default=None,
        help=(
            "Path to a Python file defining a reward callable. Expected function names: "
            "reward_function, reward, compute_reward, or get_reward."
        ),
    )
    parser.add_argument(
        "-hu",
        "--human_playable",
        action="store_true",
        help="Enable keyboard-controlled play with pygame rendering.",
    )
    parser.add_argument(
        "-pr",
        "--print_reward",
        choices=("all", "update", "none"),
        default="none",
        help=(
            "Reward printing mode: all prints every step, update prints only when the reward changes, "
            "and none prints nothing. Default: none."
        ),
    )
    parser.add_argument(
        "-m",
        "--modifs",
        nargs="+",
        type=str,
        default=None,
        help="Space-separated or comma-separated environment modifications.",
    )
    parser.add_argument(
        "--allow_conflicts",
        action="store_true",
        help="Allow conflicting mods.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Run JAX on CPU.")
    parser.add_argument("--fps", type=int, default=30, help="Render frame rate.")
    args = parser.parse_args()

    mods = _normalize_items(args.modifs) or []
    reward_function = _resolve_reward_function(args.reward_function)

    env = jaxatari_make(
        game_name=args.game,
        mods=mods,
        allow_conflicts=args.allow_conflicts,
    )
    env = _apply_reward_function(env, reward_function)

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    master_key = jrandom.PRNGKey(args.seed)
    reset_key = jrandom.fold_in(master_key, 0)
    obs, state = jitted_reset(reset_key)
    info = {}
    reward_print_state: tuple[float | None, int] = (None, 0)

    if args.human_playable:
        pygame.init()
        pygame.display.set_caption(f"JAXAtari - {args.game}")
        image = jitted_render(state)
        window = pygame.display.set_mode(
            (int(image.shape[1] * UPSCALE_FACTOR), int(image.shape[0] * UPSCALE_FACTOR))
        )
        clock = pygame.time.Clock()

        running = True
        total_reward = 0.0
        episode_index = 0

        while running:
            try:
                action_constant = get_human_action()
            except SystemExit:
                break

            action = _map_action_to_index(env, action_constant)
            obs, state, reward, done, info = jitted_step(state, action)
            done = bool(np.asarray(done))
            total_reward += float(reward)

            if reward_function is not None and args.print_reward != "none":
                reward_print_state = _update_reward_print_state(
                    _reward_value(reward),
                    args.print_reward,
                    reward_print_state[0],
                    reward_print_state[1],
                )

            image = jitted_render(state)
            update_pygame(
                window, image, UPSCALE_FACTOR, int(image.shape[1]), int(image.shape[0])
            )

            if done:
                if isinstance(info, dict) and "custom_reward" in info:
                    print(f"custom_reward={np.asarray(info['custom_reward']).item()}")
                if reward_function is not None and args.print_reward == "update":
                    _flush_reward_print_state(
                        args.print_reward, reward_print_state[0], reward_print_state[1]
                    )
                    reward_print_state = (None, 0)
                print(f"Episode finished. Total reward: {total_reward:.3f}")
                total_reward = 0.0
                episode_index += 1
                reset_key = jrandom.fold_in(master_key, episode_index)
                obs, state = jitted_reset(reset_key)

            clock.tick(args.fps)

        pygame.quit()
        return

    action_space = env.action_space()
    action_key = jrandom.fold_in(master_key, 1_000_000)
    done = False
    total_reward = 0.0

    while not done:
        action = action_space.sample(action_key)
        action_key, _ = jrandom.split(action_key)
        obs, state, reward, done, info = jitted_step(state, action)
        done = bool(np.asarray(done))
        total_reward += float(reward)

        if reward_function is not None and args.print_reward != "none":
            reward_print_state = _update_reward_print_state(
                _reward_value(reward),
                args.print_reward,
                reward_print_state[0],
                reward_print_state[1],
            )

    if reward_function is not None and args.print_reward == "update":
        _flush_reward_print_state(
            args.print_reward, reward_print_state[0], reward_print_state[1]
        )

    if isinstance(info, dict) and "custom_reward" in info:
        print(f"custom_reward={np.asarray(info['custom_reward']).item()}")
    print(f"Episode finished. Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
