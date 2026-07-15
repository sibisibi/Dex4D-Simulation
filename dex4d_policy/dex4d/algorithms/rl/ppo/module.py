import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
# import pytorch_lightning as pl
from typing import List, Optional, Tuple
from algorithms.utils.util import get_activation
from algo.pn_utils.dgcnn import DGCNN
from algo.pn_utils.transformer import Transformer


class SimplePointNetBackbone(nn.Module):
    """
    Lightweight point feature extractor for scripting.
    Aggregates per-point features with a small MLP and mean pooling.
    """

    def __init__(self, pc_dim: int, feature_dim: int):
        super().__init__()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(pc_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[1]
        flat_points = point_cloud.reshape(batch_size * num_points, self.pc_dim)
        point_features = self.mlp(flat_points)
        point_features = point_features.reshape(batch_size, num_points, self.feature_dim)
        return torch.mean(point_features, dim=1)


class Simple6DPointNetBackbone(nn.Module):
    """
    TorchScript-friendly PointNet-style encoder for 6D point clouds (current + goal xyz).
    
    Mirrors PointNetBackbone's signature while using only primitive torch ops.
    
    Implements: subtract-mean trick, shared Conv1d MLP, max/mean mix aggregation, and a global MLP.
    """

    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        assert pc_dim == 6, "Currently only supports 6D point clouds (current + goal xyz)."
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.stack_frame = 1  # keep same default as getPointNet

        conv_in_dim = pc_dim * 2
        self.conv_mlp = nn.Sequential(
            nn.Conv1d(conv_in_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(256 * self.stack_frame, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("Missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("Unexpected_keys:", unexpected_keys)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        # point_cloud: [B, N, pc_dim]
        batch_size, num_points, _ = point_cloud.shape

        # subtract mean coordinates and append mean xyz (matches getPointNet defaults)
        curr_xyz = point_cloud[:, :, :3]
        mean_curr_xyz = curr_xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
        centered_curr_xyz = curr_xyz - mean_curr_xyz  # [B, N, 3]
        goal_xyz = point_cloud[:, :, 3:6]
        mean_goal_xyz = goal_xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
        centered_goal_xyz = goal_xyz - mean_goal_xyz  # [B, N, 3]
        features = torch.cat([mean_curr_xyz.expand(-1, num_points, -1), centered_curr_xyz, mean_goal_xyz.expand(-1, num_points, -1), centered_goal_xyz], dim=-1)

        # shared Conv1d MLP over points
        point_features = self.conv_mlp(features.transpose(1, 2)).transpose(1, 2)  # [B, N, 256]

        # optional frame grouping kept for parity, stack_frame=1 here
        point_features = point_features.view(batch_size, self.stack_frame, num_points // self.stack_frame, point_features.shape[-1])

        # max/mean mix aggregation
        sep = point_features.shape[-1] // 2
        max_feature = point_features[..., :sep].amax(dim=2)  # [B, K, CF/2]
        mean_feature = point_features[..., sep:].mean(dim=2)  # [B, K, CF/2]
        global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]

        global_feature = global_feature.reshape(batch_size, -1)  # flatten frames

        return self.global_mlp(global_feature)


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        # others = {}
        return self.backbone(input_pc)



class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        state_dim: int = 191 + 29,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = 191 + 29
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        others = {}
        input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim = -1)
        return self.transpn(input_pc), others


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_obs = obs_shape[0] # partial or full
        self.num_states = states_shape[0] # full

        actor_layers = []
        critic_layers = []
            
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers.append(nn.Linear(self.num_states if self.asymmetric else self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations[:,:self.num_obs])
        # actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        observations = observations[:,:self.num_obs]

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations[:,:self.num_obs])
        return actions_mean.detach()
        
    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations[:,:self.num_obs])

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        observations = observations[:,:self.num_obs]

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
    
    
class ActorCriticPointNet(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(ActorCriticPointNet, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_keypoints = model_cfg.get('num_keypoints', 256)
        # offset of the object/goal keypoints inside the obs vector; depends on the
        # embodiment's DOF/fingertip counts (197 for xArm6+LEAP, 204 for FR3+XHand)
        self.kp_start = model_cfg.get('kp_start', 197)
        self.kp_feature_dim = 128
        self.num_obs = obs_shape[0] # partial or full
        self.num_states = states_shape[0] # full
        self.num_obs = self.num_obs - self.num_keypoints * 6 + self.kp_feature_dim # remove kp, add kp feature

        self.kp_backbone = SimplePointNetBackbone(pc_dim=6, feature_dim=self.kp_feature_dim)

        actor_layers = []
        critic_layers = []
            
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers.append(nn.Linear(self.num_states if self.asymmetric else self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def compute_obs_with_pn(self, observations):
        kp_start = self.kp_start
        kp_end = kp_start + self.num_keypoints * 6
        object_kp = observations[:, kp_start:kp_start+self.num_keypoints*3].reshape(-1, self.num_keypoints, 3)
        goal_kp = observations[:, kp_start+self.num_keypoints*3:kp_start+self.num_keypoints*6].reshape(-1, self.num_keypoints, 3)
        concatenated_kp = torch.cat([object_kp, goal_kp], dim=-1)  # [B, N, 6]
        concatenated_kp_feature = self.kp_backbone(concatenated_kp)  # [B, kp_feature_dim]
        observations = torch.cat([observations[:, :kp_start], concatenated_kp_feature, observations[:, kp_end:]], dim=1)
        return observations

    def act(self, observations, states):

        observations = self.compute_obs_with_pn(observations)
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        observations = self.compute_obs_with_pn(observations)
        actions_mean = self.actor(observations)
        return actions_mean.detach()
        
    def evaluate(self, observations, states, actions):
        observations = self.compute_obs_with_pn(observations)
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

