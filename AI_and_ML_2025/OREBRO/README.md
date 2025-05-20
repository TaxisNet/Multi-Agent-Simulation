# Human Swarm Simulation

This repository contains a simulation of the human swarm exercise, as described in [assignment.txt](assignment.txt). The simulation models agents moving in a 2D world, each following local rules to reproduce observed swarm behaviors.

## Features

- **Multiple agent strategies:**
  - **Rule 1:** Each agent positions itself between two randomly chosen agents.
  - **Rule 2:** Each agent positions itself so that one agent is "hidden" behind another.
  - **Random movement:** Agents move in random directions.
- **Collision avoidance:** Agents avoid overlapping using repulsion forces.
- **Parameterizable simulation:** Easily adjust number of agents, world size, agent speed, perception radius, and more.
- **Visualization:** Real-time visualization using matplotlib, with optional video saving.
- **Experimentation:** Analyze convergence and sensitivity to different parameters.

## File Structure

- [`agents.py`](agents.py): Main simulation logic and agent definitions.
- [`example.py`](example.py): Example configuration and entry point.
- [`assignment.txt`](assignment.txt): Assignment description and requirements.
- `.gitignore`: Ignores generated video files and cache.

## Getting Started

### Requirements

- Python 3.x
- numpy
- matplotlib

Install dependencies with:

```sh
pip install numpy matplotlib
```

### Running the Simulation

To run the simulation with the default configuration:

```sh
python example.py
```

You can modify parameters in [`example.py`](example.py) or directly in [`agents.py`](agents.py) under the `CONFIG` dictionary.

### Saving Videos

Set `'save_video': True` in the configuration to save a video of the simulation.

## Customization

- **Agent strategies:** Change `'agent_strategy'` in the config to `'rule_1'`, `'rule_2'`, or `'random_movement'`.
- **Agent parameters:** Adjust speed, repulsion, perception radius, and more in the `'agent_params'` dictionary.
- **World size and agent count:** Set `'world_size'` and `'num_agents'` as desired.

## Assignment

See [assignment.txt](assignment.txt) for the full assignment description and experiment scenarios.
