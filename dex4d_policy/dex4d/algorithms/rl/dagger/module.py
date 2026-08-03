import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
# import pytorch_lightning as pl
from typing import List, Optional, Tuple
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
from algorithms.utils.util import get_activation
from algorithms.rl.ppo.module import PointNetBackbone, SimplePointNetBackbone, Simple6DPointNetBackbone
from algo.pn_utils.transformer import Transformer


class Actor(nn.Module): # student

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(Actor, self).__init__()

        self.asymmetric = asymmetric
        
        self.use_pc = use_pc
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_obs = obs_shape[0]

        actor_layers = []

        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print("student")
        print(self.actor)


        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        actions = self.actor(observations[:,:self.num_obs])
        return actions

    def act_inference(self, observations):
        actions = self.actor(observations[:,:self.num_obs])
        return actions.detach()


class ActorPointNet(nn.Module): # student

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(ActorPointNet, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_keypoints = model_cfg['num_keypoints']
        self.kp_downsample_ratio = model_cfg['kp_downsample_ratio']
        self.num_keypoints = self.num_keypoints // self.kp_downsample_ratio
        self.num_robot_dofs = model_cfg.get('num_robot_dofs', 22)
        self.kp_feature_dim = 128
        self.num_obs = obs_shape[0] # partial
        self.num_obs = self.num_obs - self.num_keypoints * 6 + self.kp_feature_dim # remove kp, add kp feature

        self.kp_backbone = PointNetBackbone(pc_dim=6, feature_dim=self.kp_feature_dim)

        actor_layers = []

        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print("student")
        print(self.actor)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError
    
    def compute_obs_with_pn(self, observations):
        kp_start = 3 * self.num_robot_dofs  # qpos + qvel + last action
        object_kp = observations[:, kp_start:kp_start+self.num_keypoints*3].reshape(-1, self.num_keypoints, 3) # [B, self.num_keypoints, 3]
        goal_kp = observations[:, kp_start+self.num_keypoints*3:kp_start+self.num_keypoints*6].reshape(-1, self.num_keypoints, 3) # [B, self.num_keypoints, 3]
        concatenated_kp = torch.cat([object_kp, goal_kp], dim=-1)  # [B, N, 6]
        concatenated_kp_feature = self.kp_backbone(concatenated_kp)  # [B, kp_feature_dim]
        observations = torch.cat([observations[:, :kp_start], concatenated_kp_feature], dim=1)
        return observations

    def act(self, observations):
        observations = self.compute_obs_with_pn(observations)
        actions = self.actor(observations)
        return actions

    def act_inference(self, observations):
        observations = self.compute_obs_with_pn(observations)
        actions = self.actor(observations)
        return actions.detach()
    


class ActorPointNetTransformer(nn.Module): # student

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(ActorPointNetTransformer, self).__init__()

        self.asymmetric = asymmetric

        self.num_keypoints = model_cfg['num_keypoints']
        self.kp_downsample_ratio = model_cfg['kp_downsample_ratio']
        self.num_keypoints = self.num_keypoints // self.kp_downsample_ratio
        self.num_robot_dofs = model_cfg.get('num_robot_dofs', 22)
        self.token_dim = 128
        self.num_obs = obs_shape[0] # partial
        self.predict_future_states = model_cfg.get('predict_future_states', False)
        if self.predict_future_states:
            future_loss_weights_cfg = model_cfg.get('future_loss_weights', None)
            if future_loss_weights_cfg and isinstance(future_loss_weights_cfg, dict):
                self.future_state_keys = list(future_loss_weights_cfg.keys())

        self.build_tokenizers()

        self.transformer = Transformer(input_dim=self.token_dim, output_dim=actions_shape[0], output_token_idx=self.output_token_idx, latent_dim=self.token_dim, num_layer=4, use_input_layer=False, use_positional_encoding=False)
        self.transformer.output_layer[1].weight.data *= 0.01 # init last layer to be 100x smaller
        
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim)) # learnable [CLS] token

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        if self.predict_future_states:
            self.build_future_state_decoders()
        
    def build_tokenizers(self):
        self.obs_dim_dict = {
            'robot_qpos': self.num_robot_dofs,
            'robot_qvel': self.num_robot_dofs,
            'action': self.num_robot_dofs,
            'object_and_goal_kp': self.num_keypoints * 6,
        }
        self.token_keys = list(self.obs_dim_dict.keys())
        self.token_indices = {k: idx for idx, k in enumerate(self.token_keys)}
        assert 'action' in self.token_indices, "action token missing from obs_dim_dict"
        self.output_token_idx = self.token_indices['action'] # use action token
        assert sum(self.obs_dim_dict.values()) == self.num_obs, "full state dim does not match!"
        self.tokenizers = nn.ModuleDict()
        for key, dim in self.obs_dim_dict.items():
            if "kp" not in key:
                self.tokenizers[key] = nn.Linear(dim, self.token_dim)
            else:
                self.tokenizers[key] = Simple6DPointNetBackbone(pc_dim=6, feature_dim=self.token_dim)
        
    def tokenize(self, observations):
        obs_ptr = 0
        # cls_token = self.cls_token.expand(observations.shape[0], -1, -1) # [B, 1, token_dim]
        # tokens = [cls_token]
        tokens = []
        for i, (key, value) in enumerate(self.obs_dim_dict.items()):
            if "kp" not in key:
                obs = observations[:, obs_ptr:obs_ptr+value]
                obs_ptr += value
                token = self.tokenizers[key](obs) # [B, token_dim]
                token = token.unsqueeze(1) # [B, 1, token_dim]
                tokens.append(token)
            else:
                obs = observations[:, obs_ptr:obs_ptr+value].reshape(-1, value//3, 3) # [B, self.num_keypoints*2, 3]
                obs_ptr += value
                object_kp = obs[:, :self.num_keypoints, :] # [B, self.num_keypoints, 3]
                goal_kp = obs[:, self.num_keypoints:, :] # [B, self.num_keypoints, 3]
                concatenated_kp = torch.cat([object_kp, goal_kp], dim=-1) # [B, self.num_keypoints, 6]
                token = self.tokenizers[key](concatenated_kp) # [B, token_dim]
                token = token.unsqueeze(1) # [B, 1, token_dim]
                tokens.append(token)

        tokens = torch.cat(tokens, dim=1) # [B, N_tokens, token_dim]
        
        return tokens

    def build_future_state_decoders(self):
        missing_keys = [k for k in self.future_state_keys if k not in self.token_indices]
        if len(missing_keys) > 0:
            raise ValueError(f"Future state keys {missing_keys} are not present in obs_dim_dict.")

        self.future_state_decoders = nn.ModuleDict()
        for key in self.future_state_keys:
            decoder = nn.Linear(self.token_dim, self.obs_dim_dict[key])
            decoder.weight.data *= 0.01  # keep auxiliary head small at init
            self.future_state_decoders[f"future_{key}"] = decoder

    def decode_future_states(self, token_latents):
        if not self.predict_future_states:
            raise RuntimeError("Future state decoding requested but predict_future_states is disabled.")
        if not hasattr(self, "future_state_decoders"):
            raise RuntimeError("Future state decoders are not initialized.")

        future_states = {}
        for key in self.future_state_keys:
            token_idx = self.token_indices[key]
            token_feat = token_latents[:, token_idx, :]
            future_states[f"future_{key}"] = self.future_state_decoders[f"future_{key}"](token_feat)
        return future_states

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        # NOTE: THIS SHOULD BE CHANGED WHENEVER THE ARCHITECTURE CHANGES
        obs_ptr = 0
        robot_qpos_token = self.tokenizers['robot_qpos'](x[:, obs_ptr:obs_ptr+22]).unsqueeze(1)
        obs_ptr += 22
        robot_qvel_token = self.tokenizers['robot_qvel'](x[:, obs_ptr:obs_ptr+22]).unsqueeze(1)
        obs_ptr += 22
        action_token = self.tokenizers['action'](x[:, obs_ptr:obs_ptr+22]).unsqueeze(1)
        obs_ptr += 22
        object_and_goal_kp = x[:, obs_ptr:obs_ptr+self.num_keypoints*6].reshape(-1, self.num_keypoints*2, 3)
        object_kp = object_and_goal_kp[:, :self.num_keypoints, :]
        goal_kp = object_and_goal_kp[:, self.num_keypoints:, :]
        concatenated_kp = torch.cat([object_kp, goal_kp], dim=-1)
        kp_token = self.tokenizers['object_and_goal_kp'](concatenated_kp).unsqueeze(1)
        obs_ptr += self.num_keypoints*6
        tokens = torch.cat([robot_qpos_token, robot_qvel_token, action_token, kp_token], dim=1)
        actions, token_latents = self.transformer(tokens)
        # decode future robot_qpos
        future_robot_qpos = self.future_state_decoders['future_robot_qpos'](token_latents[:, self.token_indices['robot_qpos'], :])
        actions_and_future_robot_qpos = torch.cat([actions, future_robot_qpos], dim=1) # [B, action_dim + robot_qpos_dim]
        
        return actions_and_future_robot_qpos

    def act(self, observations, return_future_states=False):
        action, future_states = self._act_impl(observations, return_future_states, detach=False)
        return action, future_states

    def act_inference(self, observations, return_future_states=False):
        actions, _ = self._act_impl(observations, return_future_states, detach=True)
        return actions

    def _act_impl(self, observations, return_future_states: bool, detach: bool):
        tokens = self.tokenize(observations)
        assert not (return_future_states and not self.predict_future_states), "Requested future states but predict_future_states is False."
        actions, token_latents = self.transformer(tokens)

        actions = actions.detach() if detach else actions
        if return_future_states:
            future_states = self.decode_future_states(token_latents)
            if detach:
                future_states = {k: v.detach() for k, v in future_states.items()}
            return actions, future_states
        else:
            return actions, None
    
    def export_policy_as_jit(self, path, filename="actor_pointnet_transformer_policy.pt"):
        import os, copy
        
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        model = copy.deepcopy(self).to('cpu')
        model.eval()
        scripted_model = torch.jit.script(model)
        scripted_model.save(path)
        print(f"Saved JIT scripted policy to {path}")
    
###############################

