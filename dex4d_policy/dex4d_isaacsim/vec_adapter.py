"""Adapter presenting the IsaacLab env with the interface Dex4D's PPO expects.

SimToolReal never needed this, it ran rl_games on both sides. Dex4D's PPO is its
own and reads a `VecTaskPython` shaped object plus a `.task` with attributes on
it, so this reproduces that surface exactly rather than editing the PPO. The RL
stack is 2,418 lines with zero isaacgym imports and it should stay untouched.

What `algorithms/rl/ppo/ppo.py` actually reads, traced line by line.

  vec_env.observation_space / state_space / action_space   gym Spaces
  vec_env.num_envs
  vec_env.reset() -> obs                                    :148, :231
  vec_env.get_state() -> states                             :149, :232
  vec_env.step(actions, id) -> obs, rew, done, extras       :169, :237
  vec_env.task.num_keypoints                                :82
  vec_env.task.kp_start                                     :84, defaults 197
  vec_env.task.iteration = it                               :225
  vec_env.task.update_curriculum()                          :226
  vec_env.task.max_episode_length                           :160
  vec_env.task.successes / current_successes /
                consecutive_successes                       :172-174
  vec_env.task.sim_params.dt                                :155
  vec_env.task.cfg["env"]["controlFrequencyInv"]            :155
  vec_env.task.cfg["vis_env_id"]                            :157

Clipping matches `VecTaskPython`, obs and states to +-clip_obs, actions to
+-clip_actions, both applied before the env sees them.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from gym import spaces

import termination_utils as T


class _TaskView:
    """The `.task` attribute PPO reaches through. Backed by the lab env."""

    def __init__(self, env):
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        env = self._env
        d4 = env.d4
        self.num_keypoints = d4.obs.num_keypoints
        self.kp_start = env.lay["kp_start"]
        self.max_episode_length = d4.termination.episode_length
        self.num_envs = env.num_envs
        self.iteration = 0
        self.sim_params = SimpleNamespace(dt=d4.control.sim_dt)
        self.cfg = {
            "env": {"controlFrequencyInv": d4.control.control_frequency_inv},
            "vis_env_id": 0,
            "test": False,
        }

    # Three DISTINCT tensors in gym, tasks:204-206, read separately at
    # ppo.py:172-174. All three used to return `consecutive_successes`, which
    # reported the held-goal count under all three labels.
    @property
    def successes(self):
        return self._env.successes

    @property
    def current_successes(self):
        return self._env.current_successes

    @property
    def consecutive_successes(self):
        return self._env.consecutive_successes

    @property
    def random_time(self):
        return self._env.random_time

    @random_time.setter
    def random_time(self, v):
        # ppo.py:147 writes this in the test branch. On a plain object the write
        # would silently create a dead attribute instead of reaching the env.
        self._env.random_time = v

    @property
    def states_buf(self):
        return self._env.states_buf

    def update_curriculum(self):
        """`update_curriculum` at tasks:1604, applied to the lab env's live
        values. It overwrites three things and it fires once, at 15000."""
        env = self._env
        out = T.update_curriculum(self.iteration, env.d4.curriculum,
                                  env.d4.stage2_start_iteration)
        if out is None:
            return
        env.too_far = out["too_far_reset_threshold"]
        env.goal_stable_ratio = out["goal_reset_stable_ratio"]
        env.dof_speed_scale = out["robot_dof_speed_scale"]


class Dex4DVecAdapter:
    """`VecTaskPython` over an IsaacLab `DirectRLEnv`."""

    def __init__(self, env, rl_device="cuda:0", clip_observations=5.0,
                 clip_actions=1.0):
        self.env = env                 # may be the viewer wrapper
        self.inner = env.unwrapped if hasattr(env, "unwrapped") else env
        self.rl_device = rl_device
        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.task = _TaskView(env)
        self._extras = {}

        n_obs = self.inner.lay["num_obs"]
        n_states = self.inner.lay["num_states"]
        n_act = self.inner.n_dof
        inf = np.inf
        self.observation_space = spaces.Box(-inf, inf, (n_obs,), dtype=np.float32)
        self.state_space = spaces.Box(-inf, inf, (n_states,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32)

    @property
    def num_envs(self):
        return self.inner.num_envs

    @property
    def num_obs(self):
        return self.observation_space.shape[0]

    @property
    def num_acts(self):
        return self.action_space.shape[0]

    def get_number_of_agents(self):
        return 1

    def _clip_obs(self, x):
        return torch.clamp(x, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def get_state(self):
        return self._clip_obs(self.inner.states_buf)

    def reset(self):
        obs, _ = self.env.reset()
        return self._clip_obs(obs["policy"])

    def step(self, actions, id=0):
        """`id` is the gym side's frame counter for the viewer. It is accepted
        and ignored, the lab env keeps its own episode counter."""
        a = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs, rew, term, trunc, extras = self.env.step(a.to(self.inner.device))
        done = (term | trunc).to(torch.long)
        # Dex4D's PPO does `ep_info[key].to(device)` on every extras value at
        # ppo.py:304, so only tensors may cross. IsaacLab's own DirectRLEnv puts
        # nested dicts in extras (its "log" group), and SimToolReal's rl_games
        # observer takes those happily while this PPO cannot. Flatten one level
        # so the same content survives, and drop anything still not a tensor.
        # gym allocates `self.extras = {}` ONCE at base_task.py:82 and hands
        # back that same object every step, so ppo.py:246 appends one dict eight
        # times and the concatenation at ppo.py:301-305 averages the last step
        # with itself. Reusing one dict here reproduces that, which matters
        # because ppo.py:282 promotes the best checkpoint off the same values.
        self._extras.clear()
        if isinstance(extras, dict):
            for k, v in extras.items():
                if isinstance(v, torch.Tensor):
                    self._extras[k] = v
                elif isinstance(v, dict):
                    for sub, sv in v.items():
                        if not isinstance(sv, torch.Tensor):
                            raise TypeError(
                                f"extras[{k!r}][{sub!r}] is {type(sv).__name__}, "
                                f"and ppo.py:304 calls .to(device) on every value")
                        self._extras[f"{k}_{sub}"] = sv
                else:
                    raise TypeError(
                        f"extras[{k!r}] is {type(v).__name__}, and ppo.py:304 "
                        f"calls .to(device) on every value")
        flat = self._extras
        return (self._clip_obs(obs["policy"]),
                rew.to(self.rl_device),
                done.to(self.rl_device),
                flat)
