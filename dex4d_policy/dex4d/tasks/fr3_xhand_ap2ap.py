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
import time

from pyparsing import And
import torch

from utils.torch_jit_utils import *
from utils.data_info import plane2euler, MergeLoader
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from utils.util import visualize_point_isaacgym, visualize_axes_isaacgym
from copy import deepcopy
import yaml

from utils.robot_gains import (ARM_ARMATURE, ARM_FRICTION, ARM_GAIN_ROWS,
                               HAND_ARMATURE, HAND_DAMPING, HAND_STIFFNESS)
from utils.util import sample_position, sample_rotation, compute_keypoints, extract_mesh_keypoints, keypoint_local_to_world, mask_keypoints_oneside, mask_keypoints_test_time, mask_object_goal_keypoints_test_time, compute_fingertip_to_object_vecs, mask_object_goal_keypoints_random_height_test_time


class FR3XHandAP2AP(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5, 6]], [[0, 1, 2, 3, 4, 5, 6]]], is_multi_agent=False,
                 enable_camera_sensors=False):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = False if self.cfg['test'] else self.cfg["task"].get("randomize", False)
        self.curriculum = self.cfg["task"].get("curriculum", False)
        # stage 1 -> 2 flip iteration, CLI-overridable for the elongated-curriculum runs
        self.stage2_start_iteration = self.cfg["task"].get("stage2_start_iteration", 15000)
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.robot_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.use_osc_control = self.cfg["env"].get("useOSCControl", True)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.reward_scales = self.cfg["reward_scales"]
        self._prepare_reward_function()

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        # TODO: may change state and obs dimension
        self.num_keypoints = self.cfg["env"]["numKeypoints"]
        self.num_kp_flatten = self.num_keypoints * 6
        self.kp_downsample_ratio = self.cfg["env"]["kpDownsampleRatio"]
        self.num_masked_kp_flatten = 6 * (self.num_keypoints // self.kp_downsample_ratio)
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        # 283 = 3*19 dof + 13*5 fingertip states + 6*5 fingertip F/T + 13 hand + 19 actions + 20 obj/goal + 64 visual feat + 3*5 fingertip-obj vecs
        num_states = 283 + self.num_kp_flatten # always full state, for Critic or Dagger expert
        num_obs = 57 + self.num_masked_kp_flatten if self.asymmetric_obs else num_states # partial state, for Actor (when asymmetric) or Dagger student
        self.num_obs_dict = {
            "full_state": num_obs,
        }

        self.up_axis = 'z'
        # XHand fingertip bodies (collision-bearing distal links, simtoolreal fr3-xhand-adapter order)
        self.fingertips = ["index_rota_link2", "mid_link2", "ring_link2", "thumb_rota_link2", "pinky_link2"]  # 5 fingers
        self.hand_center = ["palm_center"]
        self.num_fingertips = len(self.fingertips) 
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 19  # FR3 (7 DOF) + XHand (12 DOF)
        # keypoint offset in the full-state obs vector, consumed by ActorCriticPointNet
        # (3*19 dof + 13*5 ft states + 6*5 ft F/T + 13 hand + 19 actions + 20 obj/goal = 204)
        self.kp_start = 204
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.segmentation_id = {
            "table": 1,
            "robot": 2,
            "object": 3,
            "goal_object": 4,
        }
        self.table_dims = gymapi.Vec3(1.2, 1.2, 0.6)

        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 0.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if "full_state" in self.obs_type or self.asymmetric_obs:
            force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                    self.num_robot_dofs + self.num_object_dofs)
            self.dof_force_tensor = self.dof_force_tensor[:, :self.num_robot_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices; FR3 arm home pose hovers the palm over the table center
        if self.cfg["env"].get("robot_default_dof_pos", None) is not None:
            self.robot_default_dof_pos = to_torch(self.cfg["env"]["robot_default_dof_pos"], dtype=torch.float, device=self.device)
        else:
            self.robot_default_dof_pos = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)
            self.robot_default_dof_pos[:7] = to_torch([0.0, 0.2, 0.0, -2.2, 0.0, 2.1, -0.785], dtype=torch.float, device=self.device)
        self.robot_default_dof_vel = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]
        self.last_robot_dof_vel = self.robot_dof_vel.clone()
        # self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        
        idx = self.hand_body_idx_dict['palm_center']
        self.right_hand_state = self.rigid_body_states[:, idx, 0:13]
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_vel = self.rigid_body_states[:, idx, 7:]
        
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.goal_pose = self.goal_state[:, 0:7]
        self.goal_pos = self.goal_state[:, 0:3]
        self.goal_rot = self.goal_state[:, 3:7]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs # total dofs in one env, including articulated objects
        self.prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device) # [num_envs, 22]
        self.cur_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device) # [num_envs, 22]
        self.effort_action = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device) # [num_envs, 22]
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.achieved_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0

        self.common_step_counter = 0
        self.push_objects = False if self.cfg['test'] else self.cfg["task"].get("push_objects", False)
        push_interval_s = self.cfg["task"].get("push_interval_s", None)
        self.push_interval = int(self.cfg["task"].get("push_interval", 0))
        if push_interval_s is not None:
            action_dt = self.dt * self.control_freq_inv
            self.push_interval = max(1, int(np.ceil(push_interval_s / action_dt)))
        self.max_push_linvel_xy = self.cfg["task"].get("max_push_linvel_xy", 0.0)
        self.max_push_angvel = self.cfg["task"].get("max_push_angvel", 0.0)
        self.rand_push_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.rand_push_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.achieved_last_time = 30
        self.too_far_reset_threshold = self.cfg['env'].get('too_far_reset_threshold', 0.3)
        self.success_threshold = 0.05

        # Nominal finger curled config: open hand (all zeros), matching the stock setup.
        # XHand has 12 hand DOFs (thumb 3, index 3, middle 2, ring 2, pinky 2).
        self.curled_q = torch.zeros(12, device=self.device)
        self.curled_q = self.curled_q.repeat(self.num_envs, 1).contiguous()

        self.object_init_stable_ratio = self.cfg['env'].get('object_init_stable_ratio', 0.2)
        self.goal_reset_stable_ratio = self.cfg['env'].get('goal_reset_stable_ratio', 0.2)
        # self.arm_action_clip = self.cfg['env'].get('arm_action_clip', 1.0)
        # self.hand_action_clip = self.cfg['env'].get('hand_action_clip', 1.0)

        self.delta_x_range = self.cfg["env"]["delta_x_range"]
        self.delta_y_range = self.cfg["env"]["delta_y_range"]


        self.update_curriculum()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_object_scale_dict(self):
        if "object_code_dict_file" in self.cfg["env"]:
            with open(self.cfg["env"]["object_code_dict_file"], 'r') as f:
                self.object_scale_dict = yaml.load(f, Loader=MergeLoader)['object_code_dict']
            print("Load object scale dict from file: ", self.cfg["env"]["object_code_dict_file"])
        else:
            self.object_scale_dict = self.cfg['env']['object_code_dict']

        object_cls_list = self.cfg["env"].get("object_cls_list", None)
        object_cls_exclude_list = self.cfg["env"].get("object_cls_exclude_list", None)
        object_num = self.cfg["env"].get("object_num", None)
        self._filter_objects(object_cls_list, object_cls_exclude_list, object_num)

    def _filter_objects(self, object_cls_list=None, object_cls_exclude_list=None, object_num=None):
        '''
        object_cls_list: List, e.g. ['mug', 'bottle'], all lower case
        object_cls_exclude_list: List, e.g. ['bottle'], all lower case
        object_num: int, e.g. 10
        Filter the object_scale_dict by class or number
        If object_cls_list is not None, filter by class first
        If object_cls_list is None, filter by number
        If object_cls_exclude_list is provided, remove the matching classes regardless of include filtering
        '''
        if self.cfg['test']:
            object_num = self.num_envs
        include_classes = None if object_cls_list is None else {cls.lower() for cls in object_cls_list}
        exclude_classes = set()
        if object_cls_exclude_list:
            exclude_classes = {cls.lower() for cls in object_cls_exclude_list}

        if include_classes is not None or exclude_classes:
            new_dict = {}
            for object_code in self.object_scale_dict.keys():
                object_class = object_code.split('/')[1].split('-')[0].lower()
                if include_classes is not None and object_class not in include_classes:
                    continue
                if object_class in exclude_classes:
                    continue
                new_dict[object_code] = self.object_scale_dict[object_code]
            self.object_scale_dict = new_dict

        if object_num is not None:
            self.object_scale_dict = {k: self.object_scale_dict[k][:1] for k in list(self.object_scale_dict.keys())[:object_num]}

        num_instances = sum([len(scales) for scales in self.object_scale_dict.values()])
        print(f"=> Finished filtering self.object_scale_dict based on include classes {object_cls_list}, exclude classes {object_cls_exclude_list}, and number {object_num}. We now have {num_instances} object instances.")

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_object_scale_dict()
        self.object_code_list = list(self.object_scale_dict.keys())
        all_scales = set()
        for object_scales in self.object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales):
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in self.object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)
        self.repose_z = self.cfg['env']['repose_z']

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 19), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        self.grasp_data = {}
        assets_path = '../assets'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')

        # datasetv4.1 grasp priors encode xArm6+LEAP (22-dim) qpos and do not apply to FR3+XHand
        assert not self.goal_cond, "goal_cond grasp priors (datasetv4.1) are xArm6+LEAP-specific"
        if self.goal_cond:
            for object_code in self.object_code_list:
                data_per_object = {}
                dataset_path = dataset_root_path + '/' + object_code
                data_num_list = os.listdir(dataset_path)
                for num in data_num_list:
                    data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                    qpos = data_dict['qpos'].item() #goal
                    scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
                    scale = round(1 / scale_inverse, 2)
                    assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                    target_qpos = torch.tensor(list(qpos.values())[:22], dtype=torch.float, device=self.device)
                    target_hand_rot_xyz = torch.tensor(list(qpos.values())[22:25], dtype=torch.float, device=self.device)  # 3
                    target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                    target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                    plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                    translation, euler = plane2euler(plane, axes='sxyz')  # object
                    object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                    object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)

                    if object_init_z > 0.05:
                        continue

                    if scale in data_per_object:
                        data_per_object[scale]['target_qpos'].append(target_qpos)
                        data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                        data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                        data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                        data_per_object[scale]['object_init_z'].append(object_init_z)
                    else:
                        data_per_object[scale] = {}
                        data_per_object[scale]['target_qpos'] = [target_qpos]
                        data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                        data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                        data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                        data_per_object[scale]['object_init_z'] = [object_init_z]
                self.grasp_data[object_code] = data_per_object


        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        robot_asset_file = "urdf/fr3_xhand_description/fr3_xhand_dex4d.urdf"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            robot_asset_file = self.cfg["env"]["asset"].get("assetFileName", robot_asset_file)

        # load fr3 + xhand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True  # FR3 has a fixed base, not flying
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        # PD gains from simtoolreal's arm gain rows, selected by config key.
        self.arm_gain_row = self.cfg["env"].get("arm_gain_row", "a3")
        fr3_kp, fr3_kd = ARM_GAIN_ROWS[self.arm_gain_row]
        print(f"arm gain row {self.arm_gain_row}: kp {fr3_kp} kd {fr3_kd}")
        fr3_dof_stiffness = to_torch(list(fr3_kp), dtype=torch.float, device=self.device)
        fr3_dof_damping = to_torch(list(fr3_kd), dtype=torch.float, device=self.device)
        xhand_dof_stiffness = to_torch(list(HAND_STIFFNESS), dtype=torch.float, device=self.device)
        xhand_dof_damping = to_torch(list(HAND_DAMPING), dtype=torch.float, device=self.device)
        fr3_xhand_dof_stiffness = torch.cat((fr3_dof_stiffness, xhand_dof_stiffness), 0)
        fr3_xhand_dof_damping = torch.cat((fr3_dof_damping, xhand_dof_damping), 0)

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_arm_dofs = 7  # fr3 has 7 dofs
        self.num_hand_dofs = self.num_robot_dofs - self.num_arm_dofs # xhand has 12 dofs

        print("self.num_robot_bodies: ", self.num_robot_bodies)
        print("self.num_robot_shapes: ", self.num_robot_shapes)
        print("self.num_robot_dofs: ", self.num_robot_dofs)
        print("self.num_arm_dofs: ", self.num_arm_dofs)
        print("self.num_hand_dofs: ", self.num_hand_dofs)
        # import pdb; pdb.set_trace()
        # (Pdb) self.gym.get_asset_rigid_body_names(robot_asset)
        # ['world', 'link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'xarm_center', 'palm_lower', 'mcp_joint', 'pip', 'dip', 'fingertip', 'index_tip_head', 'thumb_temp_base', 'thumb_pip', 'thumb_dip', 'thumb_fingertip', 'thumb_tip_head', 'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2', 'middle_tip_head', 'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3', 'ring_tip_head', 'palm_center']
        # (Pdb) self.gym.get_asset_dof_names(robot_asset)
        # ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', '1', '0', '2', '3', '12', '13', '14', '15', '5', '4', '6', '7', '9', '8', '10', '11']

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            if self.use_osc_control and i < self.num_arm_dofs:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i] = 0.0
                robot_dof_props['damping'][i] = 0.0
            else:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                if self.physics_engine == gymapi.SIM_PHYSX:
                    robot_dof_props['stiffness'][i] = fr3_xhand_dof_stiffness[i]
                    robot_dof_props['damping'][i] = fr3_xhand_dof_damping[i]
                    if i >= self.num_arm_dofs:
                        robot_dof_props['armature'][i] = HAND_ARMATURE[i - self.num_arm_dofs]
                    else:
                        # motor inertia times gear ratio squared, fr3v2/dynamics.yaml
                        robot_dof_props['armature'][i] = ARM_ARMATURE[i]
                        robot_dof_props['friction'][i] = ARM_FRICTION[i]
                else:
                    raise Exception("Currently only PhysX is supported.")

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)

        # visual feature
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }

        object_scale_idx_pairs = []
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64), device=self.device)
        self.object_keypoint_data = {}
        self.object_keypoint_buf = torch.zeros((self.num_envs, self.num_keypoints, 3), device=self.device)
        self.masked_keypoint_buf = torch.zeros((self.num_envs, self.num_keypoints // self.kp_downsample_ratio, 3), device=self.device)

        for object_id in range(len(self.object_code_list)):
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            self.object_keypoint_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                if not self.goal_cond or scale in self.grasp_data[object_code]:
                    object_scale_idx_pairs.append([object_id, scale_id])
                else:
                    print(f'prior not found: {object_code}/{scale}')
                file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                with open(file_dir, 'rb') as f:
                    feat = np.load(f)
                self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)        

        object_asset_dict = {}
        goal_asset_dict = {}

        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            goal_asset_options = deepcopy(object_asset_options)
            goal_asset_options.disable_gravity = True # floating in the air
            
            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file,
                                                              object_asset_options)
                    scaled_goal_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file,
                                                            goal_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                        goal_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset
                    goal_asset_dict[object_id][scale_id] = scaled_goal_asset

                    if object_asset is None:
                        object_asset = scaled_object_asset

                    mesh_file = osp.join(mesh_path, object_code, f"coacd/decomposed_{scale_str}.obj")
                    if osp.exists(mesh_file):
                        keypoints = extract_mesh_keypoints(mesh_file, self.num_keypoints, seed=object_id*10+scale_id)
                        keypoints = torch.tensor(keypoints, device=self.device)
                        self.object_keypoint_data[object_id][scale_id] = keypoints

            assert object_asset is not None
            # object_asset_options.disable_gravity = True    
            # goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)


        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(-0.48, 0.0, self.table_dims.z)  # base sits on the table, 0.48 from its centre
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Upright orientation

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        self.goal_displacement = gymapi.Vec3(0.0, 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size. NOTE: for larger set of training objects, increase the max_agg_bodies and max_agg_shapes (x50 or more)
        max_agg_bodies = self.num_robot_bodies * 1 + 50 * self.num_object_bodies + 1  ##
        max_agg_shapes = self.num_robot_shapes * 1 + 50 * self.num_object_shapes + 1  ##

        self.robots = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.robot_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(robot_asset, name) for name in self.fingertips]
        
        # fingertip-point links (fixed children of the *_link2 bodies, already at the tip)
        body_names = {
            'wrist': 'fr3_link7',
            'palm_lower': 'palm',
            'palm_center': 'palm_center',
            'thumb': 'thumb_rota_tip',
            'index': 'index_rota_tip',
            'middle': 'mid_tip',
            'ring': 'ring_tip',
            'pinky': 'pinky_tip',
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(robot_asset, body_name)
            
        # self.termination_contact_names = [
        #     'link2', 'link3', 'link4', 'link5',
        # ]
        self.termination_contact_names = self.gym.get_asset_rigid_body_names(robot_asset)[3:]
        self.termination_contact_indices = torch.zeros(len(self.termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, self.termination_contact_names[i])

        # create force sensors: fingertips (6x5=30)
        if "full_state" in self.obs_type or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(robot_asset, ft_handle, sensor_pose)

        self.object_scale_buf = {}

        self.object_code_and_scale_str_for_envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, -1, self.segmentation_id["robot"])
            self.hand_start_states.append(
                [robot_start_pose.p.x, robot_start_pose.p.y, robot_start_pose.p.z,
                 robot_start_pose.r.x, robot_start_pose.r.y, robot_start_pose.r.z,
                 robot_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            self.robot_indices.append(robot_idx)

            # # randomize colors and textures for rigid body
            # num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, robot_actor)
            # hand_color = [147/255, 215/255, 160/255]
            # hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            # for n in self.agent_index[0]:
            #     for m in n:
            #         for o in hand_rigid_body_index[m]:
            #             self.gym.set_rigid_body_color(env_ptr, robot_actor, o, gymapi.MESH_VISUAL,
            #                                     gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors
            if "full_state" in self.obs_type or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)


            id = int(i / self.num_envs * len(self.object_code_list))
            object_code = self.object_code_list[id]
            available_scale = []
            for scale_id in self.object_scale_id_list[id]:
                scale = self.id2scale[scale_id]
                if not self.goal_cond or scale in self.grasp_data[object_code]:
                    available_scale.append(scale)
                else:
                    print(f'prior not found: {object_code}/{scale}')
            scale = available_scale[i % len(available_scale)]
            scale_id = self.scale2id[scale]
            self.object_scale_buf[i] = scale
            self.object_id_buf[i] = id

            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]
            self.object_keypoint_buf[i] = self.object_keypoint_data[id][scale_id]
            self.masked_keypoint_buf[i] = mask_keypoints_oneside(self.object_keypoint_buf[i], self.kp_downsample_ratio)

            # record object code and scale str
            self.object_code_and_scale_str_for_envs.append({
                'object_code': object_code,
                'scale_str': scale2str[scale],
            })

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i, 0, self.segmentation_id["object"])
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset_dict[id][scale_id], goal_start_pose, "goal_object", i + self.num_envs, 0, self.segmentation_id["goal_object"])
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1,0,0))
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, self.segmentation_id["table"])
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            robot_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_actor)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            for j in range(self.num_robot_shapes):
                robot_shape_props[j].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_actor, robot_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

        if self.use_osc_control:
            self.configure_osc_controller()

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_state = self.goal_init_state.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def configure_osc_controller(self):
        # For control the xarm6
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        wrist_index = self.hand_body_idx_dict['wrist']
        self.j_eef = jacobian[:, wrist_index - 1, :, :self.num_arm_dofs]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = self.mm[:, :self.num_arm_dofs, :self.num_arm_dofs]
        self.kp = 1000
        self.kd = 20
        self.kp_null = 3
        self.kd_null = 0.5
        # self.kp = 150.
        # self.kd = 2.0 * np.sqrt(self.kp)
        # self.kp_null = 10.
        # self.kd_null = 2.0 * np.sqrt(self.kp_null)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        if self.use_osc_control:
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
        # print(">1:", self.robot_dof_pos[0])

        if "full_state" in self.obs_type or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]



        idx = self.hand_body_idx_dict['palm_center']
        self.right_hand_state = self.rigid_body_states[:, idx, 0:13]
        # palm_center and the *_tip links are dedicated frames already at the palm center /
        # fingertip points (URDF fixed links), so no quat_apply offsets are needed here.
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_vel = self.rigid_body_states[:, idx, 7:]

        # right hand fingers
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict['pinky']
        self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]

        self.goal_pose = self.goal_state[:, 0:7]
        self.goal_pos = self.goal_state[:, 0:3]
        self.goal_rot = self.goal_state[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        self.delta_qpos = self.robot_dof_pos - self.target_qpos

        self.compute_full_state()

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, visualize=True):

        self.get_unpose_quat()
        
        states_ptr = 0
        obs_ptr = 0

        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##


        ##### 66-dim: robot state, including dof_pos, dof_vel, dof_force (torque) #####
        self.states_buf[:, 0:self.num_robot_dofs] = unscale(self.robot_dof_pos,
                                                               self.robot_dof_lower_limits,
                                                               self.robot_dof_upper_limits) # 22 ##
        self.states_buf[:,self.num_robot_dofs:2 * self.num_robot_dofs] = self.vel_obs_scale * self.robot_dof_vel # 22 ##
        self.states_buf[:,2 * self.num_robot_dofs:3 * self.num_robot_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_robot_dofs] # 19 ##
        states_ptr += 3 * self.num_robot_dofs

        # For student, we only use dof_pos and dof_vel as observation
        self.obs_buf[:, obs_ptr:obs_ptr + 2 * self.num_robot_dofs] = self.states_buf[:, 0:2 * self.num_robot_dofs]
        obs_ptr += 2 * self.num_robot_dofs
        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + self.num_robot_dofs] = self.states_buf[:, 2 * self.num_robot_dofs:3 * self.num_robot_dofs]
            obs_ptr += self.num_robot_dofs


        ##### 76-dim: fingertip information, including 4 fingertip states (4 x 13), fingertips' force/torque (4 x 6, do not need repose) #####
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        # for i in range(4):
        #     aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        fingertip_states_start = states_ptr
        self.states_buf[:, fingertip_states_start:fingertip_states_start + num_ft_states] = aux
        self.states_buf[:, fingertip_states_start + num_ft_states:fingertip_states_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques]
        states_ptr += num_ft_states + num_ft_force_torques

        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + num_ft_states + num_ft_force_torques] = self.states_buf[:, fingertip_states_start:fingertip_states_start + num_ft_states + num_ft_force_torques]
            obs_ptr += num_ft_states + num_ft_force_torques


        ##### 13-dim: right hand state #####
        hand_pose_start = states_ptr
        self.states_buf[:, hand_pose_start:hand_pose_start + 10] = self.right_hand_state[:, 0:10]
        self.states_buf[:, hand_pose_start + 10:hand_pose_start + 13] = self.vel_obs_scale * self.right_hand_state[:, 10:13]
        states_ptr += 13

        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + 13] = self.states_buf[:, hand_pose_start:hand_pose_start + 13]
            obs_ptr += 13


        ##### num_robot_dofs-dim: action (22-dim) #####
        action_states_start = states_ptr
        aux = self.actions[:, :self.num_robot_dofs]
        self.states_buf[:, action_states_start:action_states_start + self.num_robot_dofs] = aux
        states_ptr += self.num_robot_dofs

        self.obs_buf[:, obs_ptr:obs_ptr + self.num_robot_dofs] = self.states_buf[:, action_states_start:action_states_start + self.num_robot_dofs]
        obs_ptr += self.num_robot_dofs


        ##### (20+self.num_kp_flatten)-dim: object & goal information, including object state (13-dim), goal state (7-dim), and object & goal keypoints (2 x 64 x 3 = 384-dim) #####
        obj_states_start = states_ptr
        self.states_buf[:, obj_states_start:obj_states_start + 3] = self.object_pose[:, 0:3]
        self.states_buf[:, obj_states_start + 3:obj_states_start + 7] = self.object_pose[:, 3:7]
        self.states_buf[:, obj_states_start + 7:obj_states_start + 10] = self.object_linvel
        self.states_buf[:, obj_states_start + 10:obj_states_start + 13] = self.vel_obs_scale * self.object_angvel
        self.states_buf[:, obj_states_start + 13:obj_states_start + 16] = self.goal_pos - self.object_pos
        self.states_buf[:, obj_states_start + 16:obj_states_start + 20] = quat_mul(self.goal_rot, quat_conjugate(self.object_rot))
        # object_keypoints = compute_keypoints(self.object_pos, self.object_rot) # [num_envs, 8, 3]
        # goal_keypoints = compute_keypoints(self.goal_pos, self.goal_rot) # [num_envs, 8, 3]
        self.object_keypoints = keypoint_local_to_world(self.object_keypoint_buf, self.object_pos, self.object_rot)
        self.goal_keypoints = keypoint_local_to_world(self.object_keypoint_buf, self.goal_pos, self.goal_rot)
        self.states_buf[:, obj_states_start + 20:obj_states_start + 20 + self.num_kp_flatten // 2] = self.object_keypoints.reshape(self.num_envs, -1)
        self.states_buf[:, obj_states_start + 20 + self.num_kp_flatten // 2:obj_states_start + 20 + self.num_kp_flatten] = self.goal_keypoints.reshape(self.num_envs, -1)
        states_ptr += 20 + self.num_kp_flatten

        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + 20 + self.num_kp_flatten] = self.states_buf[:, obj_states_start:obj_states_start + 20 + self.num_kp_flatten]
            obs_ptr += 20 + self.num_kp_flatten
        else:
            ##### self.num_masked_kp_flatten-dim: masked keypoints #####
            #### NOTE: THIS IS ONLY USED IN ASYMMETRIC SETTINGS FOR STUDENT, NOT TEACHER ####
            masked_keypoint_start = obs_ptr
            self.object_masked_keypoints = keypoint_local_to_world(self.masked_keypoint_buf, self.object_pos, self.object_rot)
            self.goal_masked_keypoints = keypoint_local_to_world(self.masked_keypoint_buf, self.goal_pos, self.goal_rot)

            # do test-time random masking
            # self.object_masked_keypoints = mask_keypoints_test_time(self.object_masked_keypoints, max_mask_prob=0.2)
            # self.goal_masked_keypoints = mask_keypoints_test_time(self.goal_masked_keypoints, max_mask_prob=0.2)
            if getattr(self, "use_cotracker", None) is not None:
                rand_object_gaussian_noise = torch.randn_like(self.object_masked_keypoints) * 0.005
                rand_goal_gaussian_noise = torch.randn_like(self.goal_masked_keypoints) * 0.005
                self.object_masked_keypoints += rand_object_gaussian_noise
                self.goal_masked_keypoints += rand_goal_gaussian_noise
                pass
            else:
                self.object_masked_keypoints, self.goal_masked_keypoints = mask_object_goal_keypoints_random_height_test_time(
                    self.object_masked_keypoints,
                    self.goal_masked_keypoints,
                    max_mask_height=0.8,
                    above_plane_mask_prob=0.9,
                    below_plane_mask_prob=0.05,
                    noise_std=0.005,
                )

            self.obs_buf[:, masked_keypoint_start:masked_keypoint_start + self.num_masked_kp_flatten // 2] = self.object_masked_keypoints.reshape(self.num_envs, -1)
            self.obs_buf[:, masked_keypoint_start + self.num_masked_kp_flatten // 2:masked_keypoint_start + self.num_masked_kp_flatten] = self.goal_masked_keypoints.reshape(self.num_envs, -1)
            obs_ptr += self.num_masked_kp_flatten


        ##### 29-dim: goal hand qpos #####
        # hand_goal_start = states_ptr
        # self.states_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        # self.states_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
        # self.states_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos
        # states_ptr += 29

        # if not self.asymmetric_obs:
        #     self.obs_buf[:, obs_ptr:obs_ptr + 29] = self.states_buf[:, hand_goal_start:hand_goal_start + 29]
        #     obs_ptr += 29


        ##### 64-dim: visual feature #####
        visual_feat_start = states_ptr
        self.states_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf
        states_ptr += 64

        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + 64] = self.states_buf[:, visual_feat_start:visual_feat_start + 64]
            obs_ptr += 64
            
            
        ##### Vectors between fingertips and object (keypoints) #####
        fingertip_object_vec_start = states_ptr
        self.fingertip_pos = torch.cat([self.right_hand_ff_pos.unsqueeze(1),
                                   self.right_hand_mf_pos.unsqueeze(1),
                                   self.right_hand_rf_pos.unsqueeze(1),
                                   self.right_hand_th_pos.unsqueeze(1),
                                   self.right_hand_lf_pos.unsqueeze(1)], dim=1)  # [num_envs, 5, 3]
        self.fingertip_to_object_vecs = compute_fingertip_to_object_vecs(self.fingertip_pos, self.object_keypoints)  # [num_envs, num_fingertips, 3], in our case [num_envs, 5, 3]
        self.states_buf[:, fingertip_object_vec_start:fingertip_object_vec_start + self.num_fingertips * 3] = self.fingertip_to_object_vecs.reshape(self.num_envs, -1)
        states_ptr += self.num_fingertips * 3
        
        if not self.asymmetric_obs:
            self.obs_buf[:, obs_ptr:obs_ptr + self.num_fingertips * 3] = self.states_buf[:, fingertip_object_vec_start:fingertip_object_vec_start + self.num_fingertips * 3]
            obs_ptr += self.num_fingertips * 3

        # Update history buffers
        self.obs_history_buf = torch.cat([
            self.obs_buf.unsqueeze(1),
            self.obs_history_buf[:, :-1, :]
        ], dim=1)
        self.states_history_buf = torch.cat([
            self.states_buf.unsqueeze(1),
            self.states_history_buf[:, :-1, :]
        ], dim=1)

        
        if self.cfg['test'] and visualize:
            # points = torch.concat([self.right_hand_pos.unsqueeze(1), self.right_hand_ff_pos.unsqueeze(1), self.right_hand_mf_pos.unsqueeze(1), self.right_hand_rf_pos.unsqueeze(1), self.right_hand_th_pos.unsqueeze(1)], dim=1)
            # visualize_point_isaacgym(self, points, radius=0.01, selected_ids=set([0, self.cfg['vis_env_id']]))
            self._visualize_point_flow(self.object_keypoints, self.goal_keypoints, num_timesteps=5)
    
    def reset_object_pose(self, env_ids):
        if self.goal_cond and self.random_prior:
            for env_id in env_ids:
                i = env_id.item()
                object_code = self.object_code_list[self.object_id_buf[i]]
                scale = self.object_scale_buf[i]

                data = self.grasp_data[object_code][scale] # data for one object one scale
                buf = data['object_euler_xy']
                prior_idx = random.randint(0, len(buf) - 1)
                # prior_idx = 0 ## use only one data

                self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
                self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
                self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
                self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
                self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]
        else:
            self.object_init_euler_xy[env_ids] = torch_rand_float(-3.14, 3.14, (len(env_ids), 2), device=self.device)
            self.object_init_z[env_ids] = torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device) + self.object_init_state[env_ids, 2].unsqueeze(-1)

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        delta_x = torch_rand_float(self.delta_x_range[0], self.delta_x_range[1], (len(env_ids), 1), device=self.device)
        delta_y = torch_rand_float(self.delta_y_range[0], self.delta_y_range[1], (len(env_ids), 1), device=self.device)
        delta_xy = torch.cat([delta_x, delta_y], dim=1) # NOTE: reset object initial position HERE
        
        # For 20% chance, we use 90 degrees (pi/2) for the roll angle for object initialization
        # pi/2 roll is a stable pose for most objects in UniDexGrasp dataset
        random_vals = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device)[:, 0]
        roll_angles = torch.where(random_vals < self.object_init_stable_ratio, torch.pi / 2.0 * torch.ones_like(theta), self.object_init_euler_xy[env_ids, 0])
        pitch_angles = torch.where(random_vals < self.object_init_stable_ratio, torch.zeros_like(theta), self.object_init_euler_xy[env_ids, 1])

        new_object_rot = quat_from_euler_xyz(roll_angles, pitch_angles, theta)
        new_object_pos = self.object_init_state[env_ids, 0:3].clone() + torch.cat([delta_xy, torch.zeros((len(env_ids), 1), device=self.device)], dim=-1)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z

        self.root_state_tensor[self.object_indices[env_ids], 0:3] = new_object_pos  # reset object position
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

    def reset_goal_pose(self, env_ids, init=False):
        if init:
            # self.goal_state[env_ids] = self.goal_init_state[env_ids].clone()
            self.goal_state[env_ids, :3] = self.root_state_tensor[self.object_indices[env_ids], 0:3].clone() + self.goal_displacement_tensor
            self.goal_state[env_ids, 3:7] = self.root_state_tensor[self.object_indices[env_ids], 3:7].clone()
        else:
            # For 20% chance, we set the goal to be a stable pose (pi/2 roll) on the table
            # Only happens when not init
            random_vals = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device)[:, 0]
            stable_pose_ids = (random_vals < self.goal_reset_stable_ratio).nonzero(as_tuple=False).squeeze(-1)
            random_sampled_pos = sample_position(self.object_pos[env_ids], dis_range=0.1)
            random_sampled_rot = sample_rotation(self.object_rot[env_ids], angle_range=0.5)

            theta = torch_rand_float(-3.14, 3.14, (len(stable_pose_ids), 1), device=self.device)[:, 0]
            random_sampled_pos[stable_pose_ids, 2] = self.object_init_state[env_ids[stable_pose_ids], 2]  # set z to be table height
            random_sampled_rot[stable_pose_ids] = quat_from_euler_xyz(torch.pi / 2.0 * torch.ones(len(stable_pose_ids), device=self.device), torch.zeros(len(stable_pose_ids), device=self.device), theta) # set to stable pose
            
            self.goal_state[env_ids, :3] = random_sampled_pos
            self.goal_state[env_ids, 3:7] = random_sampled_rot

        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_state[env_ids, 0:3]
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_state[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        
        self.reset_goal_buf[env_ids] = 0
        
    def check_termination(self):
        time_out_reset = (self.progress_buf >= self.max_episode_length).int()
        if self.cfg['test']:
            reset_flag = time_out_reset
        else:
            too_far_reset = (self.goal_obj_dist >= self.too_far_reset_threshold).int()
            # contact_reset = (torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)).int()
            # print(self.contact_forces[0, :, :])
            reset_flag = time_out_reset + too_far_reset
        
        self.reset_buf = torch.where(reset_flag >= 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset(self, env_ids):
            
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # ####### reset (init) goal pose ########
        # self.reset_goal_pose(env_ids, init=True)

        ####### reset robot ########
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_robot_dofs * 2 + 5), device=self.device)

        delta_max = self.robot_dof_upper_limits - self.robot_default_dof_pos
        delta_min = self.robot_dof_lower_limits - self.robot_default_dof_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_robot_dofs]

        pos = self.robot_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.robot_dof_pos[env_ids, :] = pos

        self.robot_dof_vel[env_ids, :] = self.robot_default_dof_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_robot_dofs:5 + self.num_robot_dofs * 2]

        self.last_robot_dof_vel[env_ids, :] = self.robot_dof_vel[env_ids, :].clone()

        self.prev_targets[env_ids, :self.num_robot_dofs] = pos
        self.cur_targets[env_ids, :self.num_robot_dofs] = pos

        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        all_robot_indices = torch.unique(torch.cat([robot_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_robot_indices), len(all_robot_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_robot_indices), len(all_robot_indices))

        ####### reset object and goal ########
        self.reset_object_pose(env_ids)

        self.reset_goal_pose(env_ids, init=True) # reset goal according to new object pose

        all_indices = torch.unique(torch.cat([all_robot_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        # NOTE: IMPORTANT! This must be the last `set_actor_root_state_tensor_indexed` call in the function
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        ####### reset internal buffers ########
        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        # self.current_successes[env_ids] = 0
        self.consecutive_successes[env_ids] = 0

        self.obs_history_buf[env_ids] = 0
        self.states_history_buf[env_ids] = 0

        for env_id in env_ids:
            self.masked_keypoint_buf[env_id] = mask_keypoints_oneside(self.object_keypoint_buf[env_id], self.kp_downsample_ratio)

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0:
            self.reset_goal_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids)

        self.actions = actions.clone().to(self.device)
        
        if self.use_osc_control:
            pos_cur = self.rigid_body_states[:, self.hand_body_idx_dict['palm_center'], 0:3]
            rot_cur = self.rigid_body_states[:, self.hand_body_idx_dict['palm_center'], 3:7]
            pos_des = pos_cur + self.actions[:, 0:3] * self.transition_scale
            delta_rot = quat_from_euler_xyz(self.actions[:, 3] * self.orientation_scale,
                                            self.actions[:, 4] * self.orientation_scale,
                                            self.actions[:, 5] * self.orientation_scale)
            rot_des = quat_mul(delta_rot, rot_cur)
            rot_des /= torch.norm(rot_des, dim=-1).unsqueeze(-1)

            if self.cfg['test']:
                visualize_axes_isaacgym(self, torch.cat((pos_des, rot_des), dim=-1), selected_ids=set([0, self.cfg['vis_env_id']]))

            pos_err = pos_des - pos_cur
            rot_err = orientation_error(rot_des, rot_cur)
            self.dpose = torch.cat((pos_err, rot_err), dim=-1).unsqueeze(-1)  # [num_envs, 6, 1]
            self.effort_action[:, :self.num_arm_dofs] = control_osc(self.dpose, self.kp, self.kd, self.kp_null, self.kd_null, self.robot_default_dof_pos, self.mm, self.j_eef, self.robot_dof_pos.unsqueeze(-1), self.robot_dof_vel.unsqueeze(-1), self.right_hand_vel, self.num_arm_dofs, self.device)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action)) # only apply to DOF_MODE_EFFORT, i.e. idx 0-5

            self.cur_targets[:] = scale(self.actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            self.cur_targets[:] = self.act_moving_average * self.cur_targets + (1.0 - self.act_moving_average) * self.prev_targets
            self.cur_targets[:] = tensor_clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets)) # only apply to DOF_MODE_POS, i.e. idx 6-21

        else:
            # # Clip the actions to make the motion slow
            # self.actions[:, :self.num_arm_dofs] = torch.clamp(self.actions[:, :self.num_arm_dofs], -self.arm_action_clip, self.arm_action_clip)
            # self.actions[:, self.num_arm_dofs:] = torch.clamp(self.actions[:, self.num_arm_dofs:], -self.hand_action_clip, self.hand_action_clip)
            self.cur_targets[:, :self.num_arm_dofs] = self.prev_targets[:, :self.num_arm_dofs] + self.robot_dof_speed_scale * self.dt * self.actions[:, :self.num_arm_dofs]
            # self.non_successes = self.successes == 0
            # print(self.non_successes.sum())
            # self.cur_targets[self.non_successes, self.num_arm_dofs:] = scale(self.actions[self.non_successes, self.num_arm_dofs:], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.cur_targets[:, self.num_arm_dofs:] = scale(self.actions[:, self.num_arm_dofs:], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.cur_targets[:] = self.act_moving_average * self.cur_targets + (1.0 - self.act_moving_average) * self.prev_targets
            self.cur_targets[:] = tensor_clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            # self.cur_targets.fill_(0)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # if self.cfg['test']: # record and visualize trajectory
        #     self.record_trajectory() # this MIGHT be before the update since haven't refreshed

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1

        if self.push_objects and self.push_interval > 0 and self.common_step_counter % self.push_interval == 0:
            self._push_objects()

        self.compute_observations()
        self.compute_reward()

        self.prev_targets[:] = self.cur_targets[:]
        self.last_robot_dof_vel[:] = self.robot_dof_vel[:]

        if self.viewer and self.debug_viz:
            # draw axes on goal object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])

    def _push_objects(self):
        push_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(push_env_ids) == 0:
            return

        lin_vel_xy = torch_rand_float(
            -self.max_push_linvel_xy, self.max_push_linvel_xy, (len(push_env_ids), 2), device=self.device)
        ang_vel = torch_rand_float(
            -self.max_push_angvel, self.max_push_angvel, (len(push_env_ids), 3), device=self.device)

        self.rand_push_linvel[push_env_ids, :2] = lin_vel_xy
        self.rand_push_angvel[push_env_ids] = ang_vel

        object_indices = self.object_indices[push_env_ids].to(torch.int32)
        self.root_state_tensor[object_indices, 7:9] = self.rand_push_linvel[push_env_ids, :2]
        self.root_state_tensor[object_indices, 10:13] = self.rand_push_angvel[push_env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(push_env_ids)
        )

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def _visualize_point_flow(self, object_keypoints, goal_keypoints, num_timesteps=10):
        if not hasattr(self, 'point_flow_buf'):
            self.point_flow_buf = torch.zeros((self.num_envs, num_timesteps, object_keypoints.shape[1], 3), device=self.device)
        # push object_keypoints to the buffer
        self.point_flow_buf[:, 1:, :, :] = self.point_flow_buf[:, :-1, :, :].clone()
        self.point_flow_buf[:, 0, :, :] = object_keypoints.clone()
        # flatten the buffer and goal_keypoints
        points = torch.cat([self.point_flow_buf.reshape(self.num_envs, -1, 3), goal_keypoints], dim=1)
        visualize_point_isaacgym(self, points, radius=0.001, selected_ids=set([0, self.cfg['vis_env_id']]))

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

    def compute_reward(self):
        # Distance between object and goal
        # goal_obj_dist = torch.norm(goal_pos - object_pos, p=2, dim=-1)
        # goal_keypoints = compute_keypoints(self.goal_pos, self.goal_rot) # [num_envs, 8, 3]
        # object_keypoints = compute_keypoints(self.object_pos, self.object_rot) # [num_envs, 8, 3]
        object_keypoints = keypoint_local_to_world(self.object_keypoint_buf, self.object_pos, self.object_rot)
        goal_keypoints = keypoint_local_to_world(self.object_keypoint_buf, self.goal_pos, self.goal_rot)
        self.goal_obj_dist = torch.mean(torch.norm(goal_keypoints - object_keypoints, p=2, dim=-1), dim=-1) # [num_envs,]
        
        # Distance between hand and goal
        self.goal_hand_dist = torch.norm(self.goal_pos - self.right_hand_pos, p=2, dim=-1)
        
        # Distance between hand and object
        self.obj_hand_dist = torch.norm(self.object_pos - self.right_hand_pos, p=2, dim=-1)
        self.obj_hand_dist = torch.where(self.obj_hand_dist >= 0.5, 0.5 + 0 * self.obj_hand_dist, self.obj_hand_dist)

        # Distance from the fingers to the object (5 fingers; caps/thresholds keep the stock per-finger values)
        self.obj_finger_dist = torch.norm(self.object_pos - self.right_hand_ff_pos, p=2, dim=-1) \
                        + torch.norm(self.object_pos - self.right_hand_mf_pos, p=2, dim=-1) \
                        + torch.norm(self.object_pos - self.right_hand_rf_pos, p=2, dim=-1) \
                        + torch.norm(self.object_pos - self.right_hand_th_pos, p=2, dim=-1) \
                        + torch.norm(self.object_pos - self.right_hand_lf_pos, p=2, dim=-1)
        # self.obj_finger_dist = torch.sum(self.fingertip_to_object_vecs.norm(p=2, dim=-1), dim=1)
        self.obj_finger_dist = torch.where(self.obj_finger_dist >= 3.75, 3.75 + 0 * self.obj_finger_dist, self.obj_finger_dist)

        delta_hand_pos_value = torch.norm(self.delta_target_hand_pos, p=1, dim=-1)
        delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(self.delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
        delta_qpos_value = torch.norm(self.delta_qpos, p=1, dim=-1)
        delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value 
        target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()

        quat_diff = quat_mul(self.goal_rot, quat_conjugate(self.object_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        lowest = self.object_pos[:, 2]
        lift_z = self.object_init_z[:, 0] + 0.6 + 0.003

        self.achieved_buf = torch.where(self.goal_obj_dist <= self.success_threshold, self.achieved_buf + 1, 0)

        if self.goal_cond:
            flag = (self.obj_finger_dist <= 0.75).int() + (self.obj_hand_dist <= 0.12).int()  + target_flag
            goal_hand_rew = torch.zeros_like(self.obj_finger_dist)
            goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * self.goal_obj_dist), goal_hand_rew)

            flag2 = (self.obj_finger_dist <= 0.75).int() + (self.obj_hand_dist <= 0.12).int()
            hand_up = torch.zeros_like(self.obj_finger_dist)
            hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * self.actions[:, 2], hand_up), hand_up)
            hand_up = torch.where(lowest >= 0.80, torch.where(flag2 == 2, 0.2 - self.goal_hand_dist * 0, hand_up), hand_up)

            bonus = torch.zeros_like(self.goal_obj_dist)
            bonus = torch.where(self.goal_obj_dist <= self.success_threshold, 1.0 / (1 + 10 * self.goal_obj_dist), bonus)

            reward = -0.5 * self.obj_finger_dist - 1.0 * self.obj_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5*delta_value

        else:
            self.flag = (self.obj_finger_dist <= 0.6).int() + (self.obj_hand_dist <= 0.12).int()
            
            reward = 0.0
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew_func_return = self.reward_functions[i]()
                if isinstance(rew_func_return, tuple):
                    unscaled_rew, metric = rew_func_return
                else:
                    unscaled_rew = rew_func_return
                    metric = None
                scaled_rew = self.reward_scales[name] * unscaled_rew
                reward += scaled_rew
                self.extras["rewards_" + name] = scaled_rew
                if metric is not None:
                    self.extras["metrics_" + name] = metric


        self.rew_buf = reward
        self.check_termination()
        self.reset_goal_buf = torch.where(self.achieved_buf >= self.achieved_last_time, torch.ones_like(self.reset_goal_buf), self.reset_buf)
        self.consecutive_successes = torch.where(self.achieved_buf >= self.achieved_last_time, self.consecutive_successes + 1, self.consecutive_successes)
        self.achieved_buf = torch.where(self.reset_goal_buf == 1, torch.zeros_like(self.achieved_buf), self.achieved_buf)
        
        self.successes = torch.where(self.goal_obj_dist <= self.success_threshold, torch.ones_like(self.successes), self.successes)
        self.current_successes = torch.where(self.reset_buf == 1, self.successes, self.current_successes)

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        ###### THIS IS FOR DEBUGGING ######
        # print(bonus.mean(), terminal_bonus.mean(), success_flag.sum(), goal_reset_flag.sum())
        # print(goal_obj_dist.mean())
        # print(self.achieved_buf.max())
        # env_idx = 3
        # print("self.robot_dof_pos:\n", self.robot_dof_pos[env_idx])
        # print("self.cur_targets:\n", self.cur_targets[env_idx])
        # print("self.actions:\n", self.actions[env_idx])
        # import pdb; pdb.set_trace()
                
    
    ############################# Reward Functions ############################
                
    def _reward_obj_finger(self):
        return -0.5 * self.obj_finger_dist, self.obj_finger_dist

    def _reward_obj_hand(self):
        return -0.5 * self.obj_hand_dist, self.obj_hand_dist

    def _reward_goal_obj(self):
        dist_rew = torch.zeros_like(self.obj_finger_dist)
        dist_rew = torch.where(self.flag == 2, 1 * (1.4 - 3 * self.goal_obj_dist), dist_rew)
        # dist_rew = torch.where(self.flag == 2, 1.5 * torch.exp(-3 * self.goal_obj_dist), dist_rew)
        return dist_rew, self.goal_obj_dist

    def _reward_success_bonus(self):
        bonus = torch.zeros_like(self.goal_obj_dist)
        bonus = torch.where(self.flag == 2, torch.where(self.goal_obj_dist <= self.success_threshold, 5.0 / (1 + 10 * self.goal_obj_dist), bonus), bonus)
        return bonus, None

    def _reward_terminal_bonus(self):
        terminal_bonus = torch.where(self.achieved_buf >= self.achieved_last_time, 10.0, 0.0)
        return terminal_bonus, None

    def _reward_finger_curl_reg(self):
        finger_curl_dist = (self.robot_dof_pos[:, self.num_arm_dofs:] - self.curled_q).norm(p=2, dim=-1)
        # finger_curl_reg = torch.where(flag < 2, 0.01 * finger_curl_dist ** 2, 0.0)
        finger_curl_reg = -0.001 * finger_curl_dist ** 2
        return finger_curl_reg, finger_curl_dist

    def _reward_table_collision_penalty(self):
        # add a term to discourage rigid bodies (excluding first few links) from colliding with the table
        height_tolerance = 0.62
        # rigid_body_height = self.rigid_body_states[:, self.termination_contact_indices, 2]  # [num_envs, num_bodies]
        fingertip_height = self.fingertip_pos[:, :, 2] # [num_envs, num_fingertips]
        right_hand_height = self.right_hand_pos[:, 2].unsqueeze(-1)  # [num_envs, 1]
        rigid_body_height = torch.cat([fingertip_height, right_hand_height], dim=1)  # [num_envs, num_fingertips + 1]
        rigid_body_min_height, _ = torch.min(rigid_body_height, dim=1)  # [num_envs,]
        table_collision_penalty = torch.where(rigid_body_min_height < height_tolerance, (rigid_body_min_height - height_tolerance) * 10, 0.0)

        return table_collision_penalty, rigid_body_min_height

    def _reward_fall_penalty(self):
        # add a fall penalty
        fall_flag = (self.goal_obj_dist >= self.too_far_reset_threshold).int()
        fall_penalty = torch.where(fall_flag == 1, -5.0, 0.0)
        return fall_penalty, None

    def _reward_hand_vel_penalty(self):
        # penalize right hand linear velocity
        right_hand_vel = self.right_hand_state[:, 7:10].clone()
        hand_vel_penalty = torch.zeros_like(self.goal_obj_dist)
        hand_vel_penalty = torch.where(self.flag == 2, torch.where(self.goal_obj_dist <= self.success_threshold, -0.1 * torch.sum(torch.square(right_hand_vel), dim=-1), hand_vel_penalty), hand_vel_penalty)
        return hand_vel_penalty, torch.sum(torch.square(right_hand_vel), dim=-1)
    
    def _reward_dof_vel(self):
        # penalize robot dof velocity
        dof_vel_penalty = -1e-3 * torch.sum(torch.square(self.robot_dof_vel), dim=-1)
        dof_vel_penalty = torch.clamp(dof_vel_penalty, min=-0.5, max=0.0)
        return dof_vel_penalty, torch.mean(torch.abs(self.robot_dof_vel), dim=-1)
    
    def _reward_dof_acc(self):
        # penalize robot dof acceleration
        dof_acc = (self.last_robot_dof_vel - self.robot_dof_vel) / self.dt
        dof_acc_penalty = -1e-8 * torch.sum(torch.square(dof_acc), dim=-1)
        dof_acc_penalty = torch.clamp(dof_acc_penalty, min=-0.5, max=0.0)
        return dof_acc_penalty, torch.mean(torch.abs(dof_acc), dim=-1)
    
    def _reward_action_penalty(self):
        # penalize action magnitude
        action_penalty = -0.01 * torch.sum(self.actions ** 2, dim=-1)
        action_penalty = torch.clamp(action_penalty, min=-0.5, max=0.0)
        return action_penalty, torch.mean(torch.abs(self.actions), dim=-1)

    ############################# Reward Functions End ############################
    
    def print_debug(self, foobar):
        if self.cfg['test']:
            # print time convert to human readable format, and then print foobar
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]", foobar)

    def record_trajectory(self):
        '''
        For OSC control:
        - Arm:
            - current joint positions
            - current end-effector pose
            - end-effector pose delta target
        - Hand:
            - current joint positions
            - joint position targets

        For non-OSC control:
        - Arm:
            - current joint positions
            - joint position targets
        - Hand:
            - current joint positions
            - joint position targets
        '''
        if not hasattr(self, 'action_trajectory'):
            self.action_trajectory = []
        
        control_signal = {}
        if self.use_osc_control:
            # Arm
            control_signal['arm_qpos'] = self.robot_dof_pos[:, :self.num_arm_dofs].cpu().numpy().copy()  # [num_envs, 6]
            control_signal['arm_ee_pose'] = self.rigid_body_states[:, self.hand_body_idx_dict['palm_center'], 0:7].cpu().numpy().copy()  # [num_envs, 7]
            control_signal['arm_ee_delta_target'] = self.dpose.squeeze(-1).cpu().numpy().copy()  # [num_envs, 6]
            # Hand
            control_signal['hand_qpos'] = self.robot_dof_pos[:, self.num_arm_dofs:].cpu().numpy().copy()  # [num_envs, 16]
            control_signal['hand_qpos_target'] = self.cur_targets[:, self.num_arm_dofs:].cpu().numpy().copy()  # [num_envs, 16]
            self.action_trajectory.append(control_signal)
        else:
            control_signal['arm_qpos'] = self.robot_dof_pos[:, :self.num_arm_dofs].cpu().numpy().copy()  # [num_envs, 6]
            control_signal['arm_qpos_target'] = self.cur_targets[:, :self.num_arm_dofs].cpu().numpy().copy()  # [num_envs, 6]
            control_signal['hand_qpos'] = self.robot_dof_pos[:, self.num_arm_dofs:].cpu().numpy().copy()  # [num_envs, 16]
            control_signal['hand_qpos_target'] = self.cur_targets[:, self.num_arm_dofs:].cpu().numpy().copy()  # [num_envs, 16]
            self.action_trajectory.append(control_signal)

        object_trajectory = {}
        object_trajectory['obj_pos'] = self.object_pos.cpu().numpy().copy()  # [num_envs, 3]
        object_trajectory['obj_rot'] = self.object_rot.cpu().numpy().copy()  # [num_envs, 4]
        object_trajectory['goal_pos'] = self.goal_pos.cpu().numpy().copy()  # [num_envs, 3]
        object_trajectory['goal_rot'] = self.goal_rot.cpu().numpy().copy()  # [num_envs, 4]
        object_trajectory['object_masked_keypoints'] = self.object_masked_keypoints.cpu().numpy().copy() if hasattr(self, 'object_masked_keypoints') else np.zeros_like(self.masked_keypoint_buf.cpu().numpy().copy())  # [num_envs, num_keypoints, 3]
        object_trajectory['goal_masked_keypoints'] = self.goal_masked_keypoints.cpu().numpy().copy() if hasattr(self, 'goal_masked_keypoints') else np.zeros_like(self.masked_keypoint_buf.cpu().numpy().copy())  # [num_envs, num_keypoints, 3]
        if not hasattr(self, 'object_trajectory'):
            self.object_trajectory = []
        self.object_trajectory.append(object_trajectory)

    def update_curriculum(self):
        if self.curriculum:
            if self.iteration >= self.stage2_start_iteration:
                self.too_far_reset_threshold = 1e6
                self.goal_reset_stable_ratio = 0.2
                # self.arm_action_clip = 0.3
                self.robot_dof_speed_scale = 1.5
            # if self.iteration > 25000:
            #     self.reward_scales['obj_finger'] = 5.0
            #     self.reward_scales['obj_hand'] = 5.0
            #     self.reward_scales['goal_obj'] = 3.0
            #     # import pdb; pdb.set_trace()
        else:
            return

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(dpose, j_eef, damping, num_arm_dofs, device):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    num_envs = j_eef.shape[0]
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, num_arm_dofs)
    return u


def control_osc(dpose, kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, dof_pos, dof_vel, hand_vel, num_arm_dofs, device):
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :num_arm_dofs]
    u_null = mm @ u_null
    u += (torch.eye(num_arm_dofs, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)
