from datetime import datetime
from importlib.resources import path
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time

from matplotlib.patches import FancyArrow
from gym import spaces

from gym.spaces import Space

import numpy as np
import statistics
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from algorithms.rl.dagger import RolloutStorage
from collections import defaultdict

from tqdm import tqdm
from utils.result_recorder import VideoRecorder, MetricWriter

from utils.util import format_time



class DAGGER:

    def __init__(self,
                 vec_env,
                 actor_class,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 buffer_size,
                 init_noise_std=1.0,
                 learning_rate=1e-3,
                 schedule="fixed",
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 expert_chkpt_path = "",
                 is_vision = False
                 ):

        self.expert_chkpt_path = expert_chkpt_path
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.schedule = schedule
        self.step_size = learning_rate

        # add num_keypoints to model_cfg
        model_cfg['num_keypoints'] = vec_env.task.num_keypoints
        model_cfg['kp_downsample_ratio'] = vec_env.task.kp_downsample_ratio
        # student obs prefix (qpos+qvel+action) is embodiment-dependent (22 stock, 19 FR3)
        model_cfg['num_robot_dofs'] = getattr(vec_env.task, 'num_robot_dofs', 22)
        self.future_loss_weights = model_cfg.get("future_loss_weights", None)
        self.teacher_forcing_iterations = model_cfg.get("teacher_forcing_iterations", -1)

        # DAGGER components
        self.buffer_size = buffer_size
        self.vec_env = vec_env
        self.actor = actor_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric, use_pc = is_vision)
        self.is_vision_expert = False
        if self.is_vision_expert:
            self.expert_observation_space = self.observation_space
        else:
            # NOTE: the expert is always full-state, otherwise there's no need for dagger
            num_states = self.state_space.shape[0]
            self.expert_observation_space = spaces.Box(np.ones(num_states) * -np.Inf, np.ones(num_states) * np.Inf)

        self.actor_expert = actor_critic_class(self.expert_observation_space.shape, self.state_space.shape, self.action_space.shape, init_noise_std, model_cfg, asymmetric=False, use_pc = False)
        self.actor.to(self.device)
        self.actor_expert.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, self.buffer_size, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # DAGGER parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        if not is_testing:
            wandb.init(
                project="Dex4D-DAGGER",
                name=os.path.basename(self.log_dir),
                dir=os.path.dirname(self.log_dir),
            )
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.model_dir = path
        self.actor.load_state_dict(torch.load(path,map_location=self.device))
        model_iter = path.split('model_')[-1].split('.')[0]
        self.current_learning_iteration = int(model_iter.split('_')[0])
        self.actor.eval()
       
    def load(self, path):
        self.actor.load_state_dict(torch.load(path))
        model_iter = path.split('model_')[-1].split('.')[0]
        self.current_learning_iteration = int(model_iter.split('_')[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        id = -1
        if self.is_testing:
            self.vec_env.task.random_time = False

        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            is_bench = getattr(self.vec_env.task, 'is_bench', False)
            path = os.path.join(os.path.dirname(self.model_dir), 'exported')
            model_name = self.model_dir.split('/')[-2]
            model_iter = os.path.basename(self.model_dir).split('model_')[-1].split('.')[0]
            filename_pt = f'{model_name}_iter_{model_iter}_jit.pt'
            self.actor.export_policy_as_jit(path, filename=filename_pt)
            self.vec_env.task.iteration = self.current_learning_iteration
            self.vec_env.task.update_curriculum()
            fps = round(
                1 / (self.vec_env.task.sim_params.dt * self.vec_env.task.cfg["env"].get("controlFrequencyInv", 1))
            )

            # video_recorder = VideoRecorder(self.vec_env.task, model_dir=self.model_dir, fps=60, env_id=self.vec_env.task.cfg['vis_env_id'])
            # self.vec_env.task.video_recorder = video_recorder
            # self.vec_env.task.self_record_video = True
            video_recorder = VideoRecorder(self.vec_env.task, model_dir=self.model_dir, fps=fps, env_id=self.vec_env.task.cfg['vis_env_id'])

            metric_writer = MetricWriter(model_dir=self.model_dir)
            N_rollouts = 1
            try:
                for i in tqdm(range(self.vec_env.task.max_episode_length * N_rollouts)):
                    video_recorder.record_frame()

                    with torch.no_grad():
                        # Compute the action
                        id = (id+1)%self.vec_env.task.max_episode_length
                        actions = self.actor.act_inference(current_obs)
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                        current_obs.copy_(next_obs)
                    if i % self.vec_env.task.max_episode_length == self.vec_env.task.max_episode_length - 2:
                        success_rate = self.vec_env.task.successes.sum() / self.vec_env.num_envs
                        if is_bench:
                            final_successes = self.vec_env.task.get_final_successes()
                            self.vec_env.task.consecutive_successes[final_successes] = self.vec_env.task.waypoint_num
                        consecutive_successes_avg = self.vec_env.task.consecutive_successes.sum() / self.vec_env.num_envs
                        metrics_dict = {
                            'is_bench': is_bench,
                            'env_name': self.vec_env.task.__class__.__name__,
                            'method': self.vec_env.task.cfg['baseline'] if self.vec_env.task.cfg['baseline'] is not None else 'Ours',
                            'rollout': i//self.vec_env.task.max_episode_length,
                            'num_envs': self.vec_env.num_envs,
                            'success_rate': success_rate.item(),
                            'consecutive_successes_avg': consecutive_successes_avg.item()
                        }
                        print("================= Test Rollout Results =================")
                        print(f"Test Rollout {i//self.vec_env.task.max_episode_length}:")
                        print(f"Env: {metrics_dict['env_name']}")
                        print(f"Method: {metrics_dict['method']}")
                        print(f"Num Envs: {metrics_dict['num_envs']}")
                        print(f"Success Rate: {success_rate.item()*100:.2f}%")
                        print(f"Consecutive Successes Avg: {consecutive_successes_avg.item():.4f}")
                        if is_bench:
                            final_success_rate = final_successes.sum() / self.vec_env.num_envs
                            task_progress = consecutive_successes_avg / self.vec_env.task.waypoint_num
                            metrics_dict['final_success_rate'] = final_success_rate.item()
                            metrics_dict['task_progress'] = task_progress.item()
                            print(f"Final Success Rate: {final_success_rate.item()*100:.2f}%")
                            print(f"Task Progress: {task_progress.item()*100:.2f}%")
                            metric_writer.write_metrics_csv(metrics_dict)
                        metric_writer.write_metrics_txt(metrics_dict)
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                model_name = self.model_dir.split('/')[-2]
                model_iter = os.path.basename(self.model_dir).split('model_')[-1].split('.')[0]
                # save the action trajectory
                if hasattr(self.vec_env.task, 'action_trajectory'):
                    action_trajectory = self.vec_env.task.action_trajectory # this is a list of dicts, save as npy
                    save_path = os.path.join(os.path.dirname(self.model_dir), f'{model_name}_iter_{model_iter}_action_trajectory.npy')
                    np.save(save_path, np.array(action_trajectory, dtype=object))
                    print(f"===> Saved action trajectory to {save_path}")
                if hasattr(self.vec_env.task, 'object_trajectory'):
                    object_trajectory = self.vec_env.task.object_trajectory # this is a list of dicts, save as npy
                    save_path = os.path.join(os.path.dirname(self.model_dir), f'{model_name}_iter_{model_iter}_object_trajectory.npy')
                    np.save(save_path, np.array(object_trajectory, dtype=object))
                    print(f"===> Saved object trajectory to {save_path}")
                if hasattr(self.vec_env.task, 'object_code_and_scale_str_for_envs'):
                    object_code_and_scale_str_for_envs = self.vec_env.task.object_code_and_scale_str_for_envs # this is a list of dicts, save as npy
                    save_path = os.path.join(os.path.dirname(self.model_dir), f'{model_name}_iter_{model_iter}_object_code_and_scale_str.npy')
                    np.save(save_path, np.array(object_code_and_scale_str_for_envs, dtype=object))
                    print(f"===> Saved object code and scale str to {save_path}")
                    
                # To load it back, use:
                # loaded_trajectory = np.load(save_path, allow_pickle=True)
                # loaded_trajectory = loaded_trajectory.tolist()
                
                exit()
            
        else:
            #expert
            self.actor_expert.load_state_dict(torch.load(self.expert_chkpt_path, map_location = self.device))
            self.actor_expert.eval()

            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            best_current_successes = 0.0

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                self.vec_env.task.iteration = it
                self.vec_env.task.update_curriculum()

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    id = (id+1)%self.vec_env.task.max_episode_length
                    
                    actions = self.actor.act_inference(current_obs)
                    actions_expert = self.actor_expert.act_inference(current_states)
                    if it < self.teacher_forcing_iterations:
                        step_actions = actions_expert
                    else:
                        step_actions = actions
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(step_actions, id)
                    next_states = self.vec_env.get_state()
                    # Record the transition (use next_obs as the "future" target)
                    self.storage.add_transitions(current_obs, actions_expert, rews, dones, future_observations=next_obs)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _ = self.actor.act_inference(current_obs)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                update_stats = self.update()
                mean_policy_loss = update_stats["policy"]
                future_loss_log = update_stats.get("future", {})
                total_loss = update_stats.get("total", mean_policy_loss)
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in ep_infos:
                        infotensor = torch.cat((infotensor, ep_info['current_successes'].to(self.device)))
                    current_successes = torch.mean(infotensor)
                    if current_successes > best_current_successes:
                        best_current_successes = current_successes
                        self.save(os.path.join(self.log_dir, 'model_{}_best.pt'.format(it)))
                    else:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}_final.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "successes" in key:
                    wandb_dict['Episode/' + key] = value
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                elif "rewards" in key:
                    wandb_dict['Rewards/' + key] = value
                elif "metrics" in key:
                    wandb_dict['Metrics/' + key] = value
                else:
                    wandb_dict['Others/' + key] = value

        wandb_dict['Loss/policy'] = locs['mean_policy_loss']
        if 'future_loss_log' in locs:
            for key, val in locs['future_loss_log'].items():
                wandb_dict[f"Loss/future_{key}"] = val
        if 'total_loss' in locs:
            wandb_dict['Loss/total'] = locs['total_loss']

        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

        wandb_dict['Train2/mean_reward/step'] = locs['mean_reward']
        wandb_dict['Train2/mean_episode_length/episode'] = locs['mean_trajectory_length']

        wandb.log(wandb_dict, step=locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Total loss:':>{pad}} {locs.get('total_loss', locs['mean_policy_loss']):.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Total loss:':>{pad}} {locs.get('total_loss', locs['mean_policy_loss']):.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        if 'future_loss_log' in locs and locs['future_loss_log']:
            for key, val in locs['future_loss_log'].items():
                log_string += f"""{'Future loss ' + key + ':':>{pad}} {val:.4f}\n"""

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Task & device:':>{pad}} {os.path.basename(self.log_dir)} <{self.device}>\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {format_time(self.tot_time)}\n"""
                       f"""{'ETA:':>{pad}} {format_time(self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']))}\n""")
        print(log_string)

    def extract_future_targets(self, future_obs_batch):
        """
        Slice next-step observations into targets aligned with actor.future_state_keys.
        """
        if not hasattr(self.actor, "obs_dim_dict"):
            raise RuntimeError("Actor does not expose obs_dim_dict for future-state supervision.")

        obs_ptr = 0
        targets = {}
        for key, dim in self.actor.obs_dim_dict.items():
            if key not in getattr(self.actor, "future_state_keys", []):
                obs_ptr += dim
                continue

            if "kp" not in key:
                targets[key] = future_obs_batch[:, obs_ptr:obs_ptr+dim]
                obs_ptr += dim
            else:
                kp_raw = future_obs_batch[:, obs_ptr:obs_ptr+dim].reshape(-1, dim // 3, 3)
                obs_ptr += dim
                object_kp = kp_raw[:, :self.actor.num_keypoints, :]
                goal_kp = kp_raw[:, self.actor.num_keypoints:, :]
                concatenated_kp = torch.cat([object_kp, goal_kp], dim=-1)
                targets[key] = concatenated_kp.reshape(future_obs_batch.shape[0], -1)
        return targets

    def update(self):
        mean_policy_loss = 0
        mean_future_losses = defaultdict(float)
        mean_total_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                future_obs_batch = self.storage.future_observations.view(-1, *self.storage.future_observations.size()[2:])[indices]

                actions_expert_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]

                if getattr(self.actor, "predict_future_states", False):
                    actions_batch, future_pred_batch = self.actor.act(obs_batch, return_future_states=True)
                else:
                    actions_batch, _ = self.actor.act(obs_batch)
                    future_pred_batch = None

                # Policy loss
                bc_loss = F.l1_loss(actions_batch, actions_expert_batch)
                
                # World model loss
                wm_loss = 0.0
                # Auxiliary future-state loss (supervise with next observations)
                if future_pred_batch is not None:
                    future_targets = self.extract_future_targets(future_obs_batch)
                    for key in self.actor.future_state_keys:
                        pred = future_pred_batch[f"future_{key}"]
                        target = future_targets[key]
                        future_loss = F.l1_loss(pred, target)
                        if isinstance(self.future_loss_weights, dict):
                            weight = self.future_loss_weights.get(key, 0.1)
                        elif isinstance(self.future_loss_weights, (float, int)):
                            weight = float(self.future_loss_weights)
                        else:
                            weight = 0.1
                        wm_loss = wm_loss + weight * future_loss
                        mean_future_losses[key] += future_loss.item()
                
                loss = bc_loss + wm_loss
                
                mean_policy_loss += bc_loss.item()
                mean_total_loss += loss.item()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        mean_total_loss /= num_updates
        mean_future_losses = {k: v / num_updates for k, v in mean_future_losses.items()}
        
        return {"policy": mean_policy_loss, "future": mean_future_losses, "total": mean_total_loss}
