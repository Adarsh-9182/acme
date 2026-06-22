# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wraps a Gymnasium environment to be used as a dm_env environment.

This wrapper supports the Gymnasium API (successor to OpenAI Gym), which has
breaking changes from legacy Gym:
  - reset() returns (observation, info)
  - step() returns (observation, reward, terminated, truncated, info)
  - Seeding is done via reset(seed=...)
"""

from typing import Any, Dict, Optional

from acme import specs
from acme import types

import dm_env
import numpy as np
import tree

try:
  import gymnasium
  from gymnasium import spaces
except ImportError:
  gymnasium = None
  spaces = None


class GymnasiumWrapper(dm_env.Environment):
  """Environment wrapper for Gymnasium environments.

  This wrapper converts a Gymnasium environment to the dm_env interface used
  by Acme. It handles the updated Gymnasium API semantics:
    - reset() returns (observation, info)
    - step() returns (observation, reward, terminated, truncated, info)
    - Proper handling of terminated vs truncated episodes
  """

  def __init__(self, environment: 'gymnasium.Env'):
    if gymnasium is None:
      raise ImportError(
          'gymnasium is not installed. Install it with: pip install gymnasium')

    self._environment = environment
    self._reset_next_step = True
    self._last_info: Optional[Dict[str, Any]] = None

    # Convert action and observation specs.
    obs_space = self._environment.observation_space
    act_space = self._environment.action_space
    self._observation_spec = _convert_to_spec(obs_space, name='observation')
    self._action_spec = _convert_to_spec(act_space, name='action')

  def reset(self, seed: Optional[int] = None) -> dm_env.TimeStep:
    """Resets the episode.

    Args:
      seed: Optional seed for reproducibility. Passed to the underlying
        Gymnasium environment's reset method.

    Returns:
      A dm_env TimeStep with step_type FIRST.
    """
    self._reset_next_step = False
    reset_kwargs = {}
    if seed is not None:
      reset_kwargs['seed'] = seed
    observation, info = self._environment.reset(**reset_kwargs)
    self._last_info = info
    return dm_env.restart(observation)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment.

    Args:
      action: Action to take in the environment.

    Returns:
      A dm_env TimeStep representing the result of the action.
    """
    if self._reset_next_step:
      return self.reset()

    observation, reward, terminated, truncated, info = (
        self._environment.step(action))
    self._last_info = info
    self._reset_next_step = terminated or truncated

    # Convert the type of the reward based on the spec, respecting the scalar
    # or array property.
    reward = tree.map_structure(
        lambda x, t: (  # pylint: disable=g-long-lambda
            t.dtype.type(x)
            if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
        reward,
        self.reward_spec())

    if terminated:
      return dm_env.termination(reward, observation)
    if truncated:
      return dm_env.truncation(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def reward_spec(self):
    return specs.Array(shape=(), dtype=float, name='reward')

  def get_info(self) -> Optional[Dict[str, Any]]:
    """Returns the last info returned from env.step(action) or env.reset().

    Returns:
      info: dictionary of diagnostic information from the last environment
        step or reset.
    """
    return self._last_info

  @property
  def environment(self) -> 'gymnasium.Env':
    """Returns the wrapped Gymnasium environment."""
    return self._environment

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name))
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _convert_to_spec(space: 'gymnasium.Space',
                     name: Optional[str] = None) -> types.NestedSpec:
  """Converts a Gymnasium space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gymnasium spaces are converted to
  BoundedArray specs. Discrete spaces are converted to DiscreteArray specs.
  Tuple and Dict spaces are recursively converted to tuples and dictionaries
  of specs.

  Args:
    space: The Gymnasium space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    if space.start != 0:
      return specs.BoundedArray(
          shape=(),
          dtype=space.dtype,
          minimum=space.start,
          maximum=space.start + int(space.n) - 1,
          name=name)
    return specs.DiscreteArray(
        num_values=int(space.n), dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=(space.n,) if isinstance(space.n, int) else space.n,
        dtype=np.int8,
        minimum=0,
        maximum=1,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _convert_to_spec(value, key)
        for key, value in space.spaces.items()
    }

  elif isinstance(space, spaces.Text):
    return specs.Array(shape=(), dtype=str, name=name)

  else:
    raise ValueError('Unexpected gymnasium space: {}'.format(space))
