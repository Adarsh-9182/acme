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

"""Tests for gymnasium_wrapper."""

import unittest

from dm_env import specs
import numpy as np

from absl.testing import absltest

SKIP_GYMNASIUM_TESTS = False
SKIP_GYMNASIUM_MESSAGE = 'gymnasium not installed.'

try:
  # pylint: disable=g-import-not-at-top
  from acme.wrappers import gymnasium_wrapper
  import gymnasium
  # pylint: enable=g-import-not-at-top
except (ModuleNotFoundError, ImportError):
  SKIP_GYMNASIUM_TESTS = True


@unittest.skipIf(SKIP_GYMNASIUM_TESTS, SKIP_GYMNASIUM_MESSAGE)
class GymnasiumWrapperTest(absltest.TestCase):

  def test_gymnasium_cartpole(self):
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))

    # Test converted observation spec.
    observation_spec: specs.BoundedArray = env.observation_spec()
    self.assertEqual(type(observation_spec), specs.BoundedArray)
    self.assertEqual(observation_spec.shape, (4,))
    self.assertEqual(observation_spec.minimum.shape, (4,))
    self.assertEqual(observation_spec.maximum.shape, (4,))
    self.assertEqual(observation_spec.dtype, np.dtype('float32'))

    # Test converted action spec.
    action_spec: specs.DiscreteArray = env.action_spec()
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 1)
    self.assertEqual(action_spec.num_values, 2)
    self.assertEqual(action_spec.dtype, np.dtype('int64'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    timestep = env.step(1)
    self.assertEqual(timestep.reward, 1.0)
    self.assertTrue(np.isscalar(timestep.reward))
    self.assertEqual(timestep.observation.shape, (4,))
    env.close()

  def test_reset_with_seed(self):
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))

    # Test that seeded reset produces reproducible results.
    ts1 = env.reset(seed=42)
    obs1 = ts1.observation.copy()

    env2 = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))
    ts2 = env2.reset(seed=42)
    obs2 = ts2.observation.copy()

    np.testing.assert_array_equal(obs1, obs2)
    env.close()
    env2.close()

  def test_truncation(self):
    # Pendulum truncates at the time limit (200 steps).
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('Pendulum-v1'))
    ts = env.reset()
    while not ts.last():
      ts = env.step(env.action_spec().generate_value())
    # Pendulum only truncates (never terminates early).
    self.assertEqual(ts.discount, 1.0)
    self.assertTrue(np.isscalar(ts.reward))
    env.close()

  def test_termination(self):
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))
    ts = env.reset()
    while not ts.last():
      # Always push right to unbalance the pole.
      ts = env.step(1)
    # CartPole terminates (discount=0) when the pole falls.
    # Unless it hits the time limit (truncation with discount=1).
    self.assertIn(ts.discount, [0.0, 1.0])
    env.close()

  def test_get_info(self):
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))
    # Info is available after reset.
    env.reset()
    info = env.get_info()
    self.assertIsNotNone(info)
    self.assertIsInstance(info, dict)
    env.close()

  def test_multi_discrete(self):
    space = gymnasium.spaces.MultiDiscrete([2, 3])
    spec = gymnasium_wrapper._convert_to_spec(space)

    spec.validate(np.array([0, 0]))
    spec.validate(np.array([1, 2]))

    self.assertRaises(ValueError, spec.validate, np.array([2, 2]))
    self.assertRaises(ValueError, spec.validate, np.array([1, 3]))

  def test_dict_space(self):
    space = gymnasium.spaces.Dict({
        'position': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        'velocity': gymnasium.spaces.Box(low=-5.0, high=5.0, shape=(3,)),
    })
    spec = gymnasium_wrapper._convert_to_spec(space)

    self.assertIsInstance(spec, dict)
    self.assertIn('position', spec)
    self.assertIn('velocity', spec)
    self.assertEqual(spec['position'].shape, (3,))
    self.assertEqual(spec['velocity'].shape, (3,))

  def test_tuple_space(self):
    space = gymnasium.spaces.Tuple((
        gymnasium.spaces.Discrete(5),
        gymnasium.spaces.Box(low=0, high=1, shape=(2,)),
    ))
    spec = gymnasium_wrapper._convert_to_spec(space)

    self.assertIsInstance(spec, tuple)
    self.assertEqual(len(spec), 2)
    self.assertIsInstance(spec[0], specs.DiscreteArray)
    self.assertIsInstance(spec[1], specs.BoundedArray)

  def test_discrete_space_with_start_offset(self):
    space = gymnasium.spaces.Discrete(5, start=2)
    spec = gymnasium_wrapper._convert_to_spec(space)

    # With start != 0, should produce BoundedArray not DiscreteArray.
    self.assertIsInstance(spec, specs.BoundedArray)
    self.assertEqual(spec.minimum, 2)
    self.assertEqual(spec.maximum, 6)
    spec.validate(np.int64(2))
    spec.validate(np.int64(6))
    self.assertRaises(ValueError, spec.validate, np.int64(1))
    self.assertRaises(ValueError, spec.validate, np.int64(7))

  def test_discrete_space_zero_start(self):
    space = gymnasium.spaces.Discrete(3)
    spec = gymnasium_wrapper._convert_to_spec(space)

    # With start == 0 (default), should produce DiscreteArray.
    self.assertIsInstance(spec, specs.DiscreteArray)
    self.assertEqual(spec.num_values, 3)

  def test_auto_reset_on_step_after_done(self):
    env = gymnasium_wrapper.GymnasiumWrapper(
        gymnasium.make('CartPole-v1'))
    ts = env.reset()
    while not ts.last():
      ts = env.step(1)
    # After done, next step should trigger auto-reset.
    ts = env.step(0)
    self.assertTrue(ts.first())
    env.close()


if __name__ == '__main__':
  absltest.main()
