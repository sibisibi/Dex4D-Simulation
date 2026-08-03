# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import os.path as osp
import random

from pyparsing import And
import torch

from utils.torch_jit_utils import *
from utils.data_info import plane2euler
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from utils.util import visualize_point_isaacgym
from copy import deepcopy

from utils.util import sample_position, sample_rotation, compute_keypoints, extract_mesh_keypoints, keypoint_local_to_world, mask_keypoints_oneside

from tasks.fr3_xhand_ap2ap import FR3XHandAP2AP, orientation_error, control_osc, control_ik


class FR3XHandAP2APVision(FR3XHandAP2AP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5, 6]], [[0, 1, 2, 3, 4, 5, 6]]], is_multi_agent=False,
                 enable_camera_sensors=False):

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index, is_multi_agent, enable_camera_sensors)

    def _create_envs(self, num_envs, spacing, num_per_row):
        return super()._create_envs(num_envs, spacing, num_per_row)

    def compute_full_state(self, visualize=True):
        super().compute_full_state(visualize=False)

        if self.cfg['test'] and visualize:
            # points = torch.concat([self.right_hand_pos.unsqueeze(1), self.right_hand_ff_pos.unsqueeze(1), self.right_hand_mf_pos.unsqueeze(1), self.right_hand_rf_pos.unsqueeze(1), self.right_hand_th_pos.unsqueeze(1)], dim=1)
            self._visualize_point_flow(self.object_masked_keypoints, self.goal_masked_keypoints, num_timesteps=5)

    def reset(self, env_ids):
        super().reset(env_ids)
    

#####################################################################
###=========================jit functions=========================###
#####################################################################
