# `scripts/run.py`

Run a JAXAtari environment in either random-agent mode (default) or keyboard-controlled mode, with optional custom reward override and optional step-by-step execution.

## Modes (short overview)

- **Default (no `--human_playable`)**: samples random actions from the environment action space and runs one episode.
- **Human playable (`--human_playable`)**: opens a pygame window and lets you control actions from the keyboard.
- **Step-by-step (`--step`)**:
	- In **human playable mode**: step manually with one-shot or held controls.
	- In **non-human mode**: terminal prompt advances one step at a time.

## Controls

- **Human playable + step (`-hu -s`)**:
	- `n`: next step (one-shot)
	- `b`: back one step (one-shot)
	- hold `j`: continuous forward stepping
	- hold `h`: continuous backward stepping
- **Non-human + step (`-s`)** at terminal prompt:
	- `Enter`: next step
	- `q`: quit

## Parameters

| Parameter | Default | Type / Options | What it does |
|---|---:|---|---|
| `-g`, `--game` | **required** | `str` | Game name passed to `jaxatari.core.make(...)`. |
| `-rf`, `--reward_function` | `None` | `str` (path to `.py`) | Loads a reward function from a Python file and overrides env reward via wrapper. Accepted callable names: `reward_function`, `reward`, `compute_reward`, `get_reward`. |
| `-hu`, `--human_playable` | `False` | flag | Enables pygame window rendering and keyboard control. |
| `-s`, `--step` | `False` | flag | Enables step-by-step execution with history navigation (forward/back). |
| `-pr`, `--print_reward` | `none` | `all` \| `update` \| `none` | Reward printing mode for custom reward output: every step, only on value change (with run-length count), or no printing. |
| `-m`, `--modifs` | `None` | one or more `str` (`nargs='+'`, comma-separated also supported) | Environment modifications. Values are normalized from space- and comma-separated inputs. |
| `--allow_conflicts` | `False` | flag | Allows conflicting mods when creating the environment. |
| `--seed` | `0` | `int` | Master PRNG seed used for reset/action keys. |
| `--cpu` | `False` | flag | Requests CPU backend by setting JAX-related env vars before JAX import. |
| `--fps` | `30` | `int` | Target render/update rate used in pygame loop (`clock.tick(fps)`). |
