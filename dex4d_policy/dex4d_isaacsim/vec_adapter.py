"""VecTaskPython surface over an IsaacLab DirectRLEnv, for Dex4D's own PPO."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from gym import spaces

import termination_utils as T


class _TaskView:
    """The `.task` attribute PPO reaches through, backed by the lab env."""

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
        self._env.random_time = v

    @property
    def states_buf(self):
        return self._env.states_buf

    def update_curriculum(self):
        """Apply the stage 2 overrides to the env's live values."""
        env = self._env
        out = T.update_curriculum(self.iteration, env.d4.curriculum,
                                  env.d4.stage2_start_iteration)
        if out is None:
            return
        env.too_far = out["too_far_reset_threshold"]
        env.goal_stable_ratio = out["goal_reset_stable_ratio"]
        env.dof_speed_scale = out["robot_dof_speed_scale"]


class Dex4DVecAdapter:

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
        """`id` is the caller's frame counter, accepted and ignored."""
        a = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs, rew, term, trunc, extras = self.env.step(a.to(self.inner.device))
        done = (term | trunc).to(torch.long)
        # PPO calls .to(device) on every extras value, so only tensors may cross
        # and nested groups are flattened one level. The dict is reused rather
        # than rebuilt, so the logged values are the last step's.
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
                                f"every extras value must be a tensor")
                        self._extras[f"{k}_{sub}"] = sv
                else:
                    raise TypeError(
                        f"extras[{k!r}] is {type(v).__name__}, "
                        f"every extras value must be a tensor")
        flat = self._extras
        return (self._clip_obs(obs["policy"]),
                rew.to(self.rl_device),
                done.to(self.rl_device),
                flat)
