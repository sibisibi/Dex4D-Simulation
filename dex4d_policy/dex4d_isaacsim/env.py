"""Dex4D on IsaacLab. Thin env, shaped on SimToolReal's `simtoolreal_env.py`.

All task maths lives in the util modules, each checked numerically against the
gym original. This file is wiring only.

Four boundary facts the port has to respect, all measured rather than assumed.

  joint order   IsaacLab's order is its own, so everything crossing into task
                maths goes through `lab_to_canon`
  quaternions   IsaacLab is wxyz, Dex4D is xyzw, converted once on the way in
  palm_center   merged away by `merge_fixed_joints`, so it is fr3_link7 plus
                0.16 m along that link's local z, which is what the URDF's
                palm_center_joint says
  fingertips    all five survive the merge, index_rota_link2, mid_link2,
                ring_link2, thumb_rota_link2, pinky_link2
"""
from __future__ import annotations

import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils

import action_utils as A
import obs_utils as O
import randomization as D
import reset_utils as R
import reward_utils as W
import scene_utils as S
import termination_utils as T
from robots import (ARM_JOINTS, FINGERTIP_TIP_OFFSETS, HAND_JOINTS,
                    RESET_ARM_POS, RESET_HAND_POS)

CANONICAL_JOINTS = ARM_JOINTS + HAND_JOINTS
FINGERTIP_BODIES = ["index_rota_link2", "mid_link2", "ring_link2",
                    "thumb_rota_link2", "pinky_link2"]
PALM_PARENT = "fr3_link7"
PALM_OFFSET = (0.0, 0.0, 0.16)      # urdf palm_center_joint
OBJ_FINGER_CAP = 3.75               # tasks:1394
OBJ_HAND_CAP = 0.5                  # tasks:1390


def quat_rotate(q_xyzw, v):
    x, y, z, w = q_xyzw[:, 0:1], q_xyzw[:, 1:2], q_xyzw[:, 2:3], q_xyzw[:, 3:4]
    t = 2.0 * torch.cross(torch.cat([x, y, z], dim=1), v, dim=1)
    return v + w * t + torch.cross(torch.cat([x, y, z], dim=1), t, dim=1)


class Dex4DEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.d4 = cfg.dex4d
        self.lay = O.layout(
            num_robot_dofs=self.d4.control.num_arm_dofs + self.d4.control.num_hand_dofs,
            num_fingertips=self.d4.obs.num_fingertips,
            num_keypoints=self.d4.obs.num_keypoints,
            kp_downsample_ratio=self.d4.obs.kp_downsample_ratio,
            asymmetric=self.d4.obs.asymmetric_observations)
        cfg.action_space = self.d4.control.num_arm_dofs + self.d4.control.num_hand_dofs
        cfg.observation_space = self.lay["num_obs"]
        cfg.state_space = self.lay["num_states"]
        super().__init__(cfg, render_mode, **kwargs)
        self._allocate()
        S.apply_physx_material_properties(self)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        d4 = self.d4
        self._usd_paths = S.load_usd_cache(d4.usd_cache, d4.object_cls)
        self._goal_usd_paths = S.load_goal_usd_cache(d4.usd_cache)
        self._assignment = S.assign_objects_to_envs(self._usd_paths, self.num_envs)

        self.robot = Articulation(S.build_robot_cfg(d4.robot_usd, d4))
        self.table = RigidObject(S.build_table_cfg(d4))
        self.object = RigidObject(S.build_object_cfg(self._usd_paths, d4, goal_viz=False))
        self.goal_viz = RigidObject(S.build_object_cfg(
            self._usd_paths, d4, goal_viz=True,
            goal_usd_paths=self._goal_usd_paths))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_viz"] = self.goal_viz
        # tasks:252-255. Gym's `PlaneParams()` defaults put the floor at z = 0
        # with friction 1.0, and the table bottom rests on it. IsaacLab's
        # GroundPlaneCfg defaults to friction 0.5, so both have to be said.
        ground = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=d4.scene.ground_friction,
                dynamic_friction=d4.scene.ground_friction))
        ground.func("/World/ground", ground,
                    translation=(0.0, 0.0, d4.scene.ground_z))
        light = sim_utils.DomeLightCfg(intensity=750.0)
        light.func("/World/Light", light)
        # Gym gives every actor collision group `i`, and IsaacGym guarantees
        # actors in different groups never collide. Without this the envs are
        # 3.0 m apart and their objects can still touch each other.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])


    def _allocate(self):
        d4, dev, n = self.d4, self.device, self.num_envs
        self.c2l, self.l2c = S.joint_permutation(
            list(self.robot.data.joint_names), CANONICAL_JOINTS)
        self.c2l, self.l2c = self.c2l.to(dev), self.l2c.to(dev)

        lim = self.robot.data.joint_limits[0]
        self.dof_lower = lim[:, 0][self.c2l].clone()
        self.dof_upper = lim[:, 1][self.c2l].clone()

        bodies = list(self.robot.data.body_names)
        missing = [b for b in FINGERTIP_BODIES + [PALM_PARENT] if b not in bodies]
        if missing:
            raise ValueError(f"bodies absent after the merge: {missing}")
        self.ft_idx = torch.tensor([bodies.index(b) for b in FINGERTIP_BODIES],
                                   device=dev)
        self.palm_idx = bodies.index(PALM_PARENT)
        self.palm_offset = torch.tensor(PALM_OFFSET, device=dev).expand(n, 3).contiguous()
        # The tip frames the reward, the table penalty and the fingertip-object
        # vectors all read, reconstructed because the merge deleted the links.
        self.tip_offset = torch.tensor(
            [FINGERTIP_TIP_OFFSETS[b] for b in FINGERTIP_BODIES],
            device=dev, dtype=torch.float32)

        kp = np.load(d4.keypoint_npz)
        keys = list(kp["keys"])
        table = {k: i for i, k in enumerate(keys)}
        cache_keys = S.load_usd_cache_keys(d4.usd_cache)
        # The pose viewer resolves each env's object urdf through these.
        self._cache_keys = cache_keys
        self._corpus_root = d4.corpus_root
        rows = []
        for e_i in self._assignment:
            key = cache_keys[e_i]
            if key not in table:
                raise KeyError(f"no keypoints for {key}")
            rows.append(table[key])
        self.kp_local = torch.tensor(kp["keypoints"][rows], device=dev,
                                     dtype=torch.float32)

        self.n_arm = d4.control.num_arm_dofs
        self.n_dof = self.n_arm + d4.control.num_hand_dofs
        self.prev_targets = torch.zeros(n, self.n_dof, device=dev)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        self.actions = torch.zeros(n, self.n_dof, device=dev)
        self.achieved_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.consecutive_successes = torch.zeros(n, device=dev)
        self.states_buf = torch.zeros(n, self.lay["num_states"], device=dev)
        self.goal_pos = torch.zeros(n, 3, device=dev)
        self.goal_rot = torch.zeros(n, 4, device=dev); self.goal_rot[:, 3] = 1.0
        self.object_init_state = torch.zeros(n, 3, device=dev)
        self.object_init_state[:, 2] = d4.scene.object_spawn[2]
        self.too_far = d4.termination.too_far_reset_threshold
        self.dof_speed_scale = d4.control.dof_speed_scale
        self.goal_stable_ratio = d4.goal.stable_ratio
        self.curled_q = torch.zeros(d4.control.num_hand_dofs, device=dev)
        self.visual_feat = S.load_visual_features(
            d4.visual_feat_root, cache_keys, self._assignment, dev)
        # tasks:1331. A global step counter drives the push, exactly as
        # `common_step_counter` does, and the interval is push_interval_s over
        # the policy period.
        self.push_step = 0
        self.push_interval = max(1, int(np.ceil(
            d4.push.interval_s
            / (d4.control.sim_dt * d4.control.control_frequency_inv))))
        # Default joint state to restore at reset, in canonical order.
        self.default_dof_pos = torch.tensor(
            RESET_ARM_POS + RESET_HAND_POS, device=dev,
            dtype=torch.float32).expand(n, self.n_dof).contiguous()
        self.goal_obj_dist = torch.zeros(n, device=dev)
        self.flag = torch.zeros(n, dtype=torch.long, device=dev)
        # tasks:204-206. successes latches once the goal is ever met in an
        # episode, current_successes samples it at reset, consecutive_successes
        # counts held goals. PPO reads all three off extras at :282.
        self.successes = torch.zeros(n, device=dev)
        self.current_successes = torch.zeros(n, device=dev)
        self.random_time = d4.random_time
        # tasks:169, seeded at reset, consumed only by `_reward_dof_acc`.
        self.last_dof_vel = torch.zeros(n, self.n_dof, device=dev)

        # tasks:1177-1178 calls apply_randomizations on EVERY reset, and
        # `task.randomize` is True in every shipped FR3 config.
        self.dr = d4.randomization
        self.sim_frame = 0
        self.randomize_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.dr_first = True
        self.obs_corr = torch.zeros(n, self.lay["num_obs"], device=dev)
        self.act_corr = torch.zeros(n, self.n_dof, device=dev)
        self.dof_lower_dr = self.dof_lower.clone().expand(n, self.n_dof).contiguous()
        self.dof_upper_dr = self.dof_upper.clone().expand(n, self.n_dof).contiguous()
        # Nominal values every scaling draw multiplies, captured once so the
        # randomization never compounds on itself across resets.
        self._kp0 = {k: a.stiffness.clone() for k, a in self.robot.actuators.items()}
        self._kd0 = {k: a.damping.clone() for k, a in self.robot.actuators.items()}
        self._mass0 = {"robot": self.robot.root_physx_view.get_masses().clone(),
                       "object": self.object.root_physx_view.get_masses().clone()}

    # -------------------------------------------------------------- reading
    def _origins(self):
        return self.scene.env_origins

    def _object_pose(self):
        return (self.object.data.root_pos_w - self._origins(),
                S.quat_wxyz_to_xyzw(self.object.data.root_quat_w))

    def _palm(self):
        p = self.robot.data.body_pos_w[:, self.palm_idx] - self._origins()
        q = S.quat_wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.palm_idx])
        return p + quat_rotate(q, self.palm_offset), q

    @property
    def palm_lin_vel(self):
        """`right_hand_state[:, 7:10]`, the palm point's own linear velocity,
        which is link7's plus `w x r` across the 0.16 m weld."""
        q = S.quat_wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.palm_idx])
        w = self.robot.data.body_ang_vel_w[:, self.palm_idx]
        return (self.robot.data.body_lin_vel_w[:, self.palm_idx]
                + torch.cross(w, quat_rotate(q, self.palm_offset), dim=1))

    def _fingertip_bodies(self):
        """The five `*_link2` body frames. This is what gym's `fingertip_state`
        observation block reads, via `self.fingertips` at :104."""
        return self.robot.data.body_pos_w[:, self.ft_idx] - self._origins()[:, None, :]

    def _fingertips(self):
        """The five tip points. Gym reads the `*_tip` links here, and the merge
        deleted them, so each is its body position plus the URDF offset rotated
        into world. Feeds `obj_finger_dist`, the table penalty and the
        fingertip-object vectors, which is every consumer except the observation
        block above."""
        n, k = self.num_envs, self.ft_idx.numel()
        pos = self._fingertip_bodies()
        q = S.quat_wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.ft_idx])
        off = self.tip_offset.unsqueeze(0).expand(n, k, 3)
        shifted = quat_rotate(q.reshape(-1, 4), off.reshape(-1, 3)).reshape(n, k, 3)
        return pos + shifted

    def _fingertip_quat_xyzw(self):
        return S.quat_wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.ft_idx])

    def _fingertip_wrench(self):
        """The five 6-axis fingertip force and torque, finger major, in the
        order `FINGERTIP_BODIES`, matching gym's sensor creation order."""
        w = self.robot.root_physx_view.get_link_incoming_joint_force()
        return w.to(self.device)[:, self.ft_idx].reshape(self.num_envs, -1)

    def _intermediates(self):
        opos, orot = self._object_pose()
        palm, _ = self._palm()
        ft = self._fingertips()
        d_ft = torch.norm(opos[:, None, :] - ft, p=2, dim=-1).sum(dim=1)
        self.obj_finger_dist = torch.clamp(d_ft, max=OBJ_FINGER_CAP)
        self.obj_hand_dist = torch.clamp(torch.norm(opos - palm, p=2, dim=-1),
                                         max=OBJ_HAND_CAP)
        self.flag = W.contact_flag(self.obj_finger_dist, self.obj_hand_dist)
        okp = O.keypoint_local_to_world(self.kp_local, opos, orot)
        gkp = O.keypoint_local_to_world(self.kp_local, self.goal_pos, self.goal_rot)
        self.goal_obj_dist = torch.norm(gkp - okp, p=2, dim=-1).mean(dim=-1)
        return opos, orot, palm, ft

    # ---------------------------------------------------------------- hooks
    def _pre_physics_step(self, actions):
        # base_task.py:147-148 adds the action noise AFTER VecTaskPython's +-1
        # clamp, so the stored action can leave the range and the arm integrator
        # and the action penalty both see the noisy value. The clamp therefore
        # happens in the adapter and NOT again here.
        a = actions.clone()
        if self.dr.enabled:
            a = D.gaussian_noise(a, self.act_corr, self.dr.actions, self.sim_frame)
        self.actions = a
        self.cur_targets = A.compute_targets(
            self.actions, self.prev_targets, self.dof_lower_dr, self.dof_upper_dr,
            self.n_arm, self.dof_speed_scale, self.d4.control.action_dt,
            self.d4.control.act_moving_average)
        self.prev_targets = self.cur_targets.clone()

    def _maybe_push_objects(self):
        """tasks:1300 and :1331. Every `push_interval` policy steps the object's
        linear xy and angular velocity are overwritten for EVERY env, which is
        the disturbance the policy has to recover from.

        Called from `_get_dones`, not `_pre_physics_step`. Gym pushes inside
        `post_physics_step` AFTER the physics and BEFORE `compute_observations`,
        so its policy sees the velocity spike and can react on the same step.
        Pushing before the step let physics swallow it unobserved.
        """
        p = self.d4.push
        if not p.enabled:
            return
        self.push_step += 1
        if self.push_step % self.push_interval:
            return
        n, dev = self.num_envs, self.device
        # Gym writes root_state[7:9] and [10:13], so linear x, y and all three
        # angular. Index 9, the vertical velocity, is deliberately left alone,
        # and zeroing it would arrest a falling or rising object on every push.
        vel = torch.cat([self.object.data.root_lin_vel_w,
                         self.object.data.root_ang_vel_w], dim=-1).clone()
        vel[:, 0:2] = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * p.max_linvel_xy
        vel[:, 3:6] = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * p.max_angvel
        self.object.write_root_velocity_to_sim(vel)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.cur_targets[:, self.l2c])

    def _get_dones(self):
        # tasks:1296-1298. Both counters advance once per policy step,
        # and the DR schedule clock is the SIM frame count, so it moves
        # by the decimation.
        self.sim_frame += self.cfg.decimation
        self.randomize_buf += 1
        self._maybe_push_objects()
        self._intermediates()
        self.achieved_buf = T.update_achieved(self.achieved_buf, self.goal_obj_dist)
        # Gym computes every reward term FIRST and zeroes `achieved_buf` after,
        # at :1462, so the terminal bonus lands on the step the goal is held for
        # the thirtieth time. IsaacLab runs this hook before `_get_rewards`, so
        # the hit is only latched here and the zeroing moved to the end of the
        # reward. Zeroing here made the bonus unreachable.
        self.goal_hit = T.goal_reached(self.achieved_buf)
        if bool(self.goal_hit.any()):
            ids = self.goal_hit.nonzero(as_tuple=False).squeeze(-1)
            self._resample_goal(ids)
            self.consecutive_successes += self.goal_hit.float()
        terminated = self.goal_obj_dist >= self.too_far
        truncated = self.episode_length_buf >= self.d4.termination.episode_length
        # tasks:1464-1465, in that order.
        self.successes = torch.where(self.goal_obj_dist <= W.SUCCESS_THRESHOLD,
                                     torch.ones_like(self.successes), self.successes)
        reset = (terminated | truncated)
        self.current_successes = torch.where(reset, self.successes,
                                             self.current_successes)
        return terminated, truncated

    def _apply_randomizations(self, env_ids):
        """`base_task.py:245-439`, called from `reset` at tasks:1177-1178.

        The gating is gym's. Non-env parameters, meaning the two noise streams'
        correlated draws, refresh on the `frequency` beat in SIM frames. Per
        actor parameters need `randomize_buf >= frequency` as well as being in
        the reset set. `first_randomization` overrides both once.
        """
        dr = self.dr
        if not dr.enabled:
            return
        dev, f = self.device, self.sim_frame
        beat = self.dr_first or (self.sim_frame % dr.frequency == 0)
        if beat:
            self.obs_corr = D.refresh_correlated(self.obs_corr.shape, dr.observations, dev)
            self.act_corr = D.refresh_correlated(self.act_corr.shape, dr.actions, dev)

        if self.dr_first:
            ids = torch.arange(self.num_envs, device=dev)
        else:
            due = self.randomize_buf[env_ids] >= dr.frequency
            ids = env_ids[due]
        if ids.numel() == 0:
            return
        self.randomize_buf[ids] = 0
        n = ids.numel()

        # Joint limits. These feed the hand's absolute action map, so a shift
        # rescales every hand command, which is why gym randomizes them.
        self.dof_lower_dr[ids] = self.dof_lower + D.additive_samples(
            dr.robot_lower, (n, self.n_dof), dev, f)
        self.dof_upper_dr[ids] = self.dof_upper + D.additive_samples(
            dr.robot_upper, (n, self.n_dof), dev, f)

        # Actuator gains, scaled per env about the nominal row.
        kp = D.scale_samples(dr.robot_stiffness, n, dev, f).unsqueeze(-1)
        kd = D.scale_samples(dr.robot_damping, n, dev, f).unsqueeze(-1)
        for key, act in self.robot.actuators.items():
            j = act.joint_indices
            act.stiffness[ids] = self._kp0[key][ids] * kp
            act.damping[ids] = self._kd0[key][ids] * kd
            self.robot.write_joint_stiffness_to_sim(
                act.stiffness[ids], joint_ids=j, env_ids=ids)
            self.robot.write_joint_damping_to_sim(
                act.damping[ids], joint_ids=j, env_ids=ids)

        # Body masses and shape frictions, both through the PhysX views. The
        # friction draw is bucketed because PhysX caps live materials.
        cpu = ids.cpu()
        for name, mspec, fspec in (("robot", dr.robot_mass, dr.robot_friction),
                                   ("object", dr.object_mass, dr.object_friction)):
            view = getattr(self, name).root_physx_view
            m = view.get_masses()
            m[cpu] = self._mass0[name][cpu] * D.scale_samples(
                mspec, n, "cpu", f).unsqueeze(-1)
            view.set_masses(m, cpu)
            mats = view.get_material_properties()
            s = D.scale_samples(fspec, n, "cpu", f).view(-1, 1)
            base = float(self.d4.scene.friction)
            mats[cpu, :, 0] = base * s
            mats[cpu, :, 1] = base * s
            view.set_material_properties(mats, cpu)

        self.dr_first = False

    def _resample_goal(self, env_ids):
        opos, orot = self._object_pose()
        pos, rot, _ = R.reset_goal_pose_walk(
            opos[env_ids], orot[env_ids], self.object_init_state[env_ids],
            self.goal_stable_ratio, self.d4.goal.walk_step, self.d4.goal.walk_angle,
            bounds=self.d4.goal.xyz_bound, device=self.device)
        self.goal_pos[env_ids], self.goal_rot[env_ids] = pos, rot
        self._write_goal_viz(env_ids)

    def _write_goal_viz(self, env_ids):
        pose = torch.zeros(len(env_ids), 7, device=self.device)
        pose[:, 0:3] = self.goal_pos[env_ids] + self._origins()[env_ids]
        pose[:, 3:7] = S.quat_xyzw_to_wxyz(self.goal_rot[env_ids])
        self.goal_viz.write_root_pose_to_sim(pose, env_ids)

    def _get_rewards(self):
        scales = W.active_terms(self.d4.reward.as_dict())
        r = torch.zeros(self.num_envs, device=self.device)
        if "obj_finger" in scales:
            r += scales["obj_finger"] * W.reward_obj_finger(self.obj_finger_dist)
        if "obj_hand" in scales:
            r += scales["obj_hand"] * W.reward_obj_hand(self.obj_hand_dist)
        if "goal_obj" in scales:
            r += scales["goal_obj"] * W.reward_goal_obj(self.goal_obj_dist, self.flag)
        if "success_bonus" in scales:
            r += scales["success_bonus"] * W.reward_success_bonus(
                self.goal_obj_dist, self.flag)
        if "terminal_bonus" in scales:
            r += scales["terminal_bonus"] * W.reward_terminal_bonus(self.achieved_buf)
        if "finger_curl_reg" in scales:
            hand_q = self.robot.data.joint_pos[:, self.c2l][:, self.n_arm:]
            r += scales["finger_curl_reg"] * W.reward_finger_curl_reg(hand_q, self.curled_q)
        if "table_collision_penalty" in scales:
            palm, _ = self._palm()
            # gym's `fingertip_pos` at :1060 is the tip frames, not `*_link2`.
            r += scales["table_collision_penalty"] * W.reward_table_collision_penalty(
                self._fingertips(), palm)
        if "action_penalty" in scales:
            r += scales["action_penalty"] * W.reward_action_penalty(self.actions)
        # The four gym zeroes. `_prepare_reward_function` pops them at zero on
        # both sides, so they cost nothing today, but gym HAS all twelve and a
        # nonzero scale here used to be dropped in silence.
        dof_vel_c = self.robot.data.joint_vel[:, self.c2l]
        if "fall_penalty" in scales:
            r += scales["fall_penalty"] * W.reward_fall_penalty(
                self.goal_obj_dist, self.too_far)
        if "hand_vel_penalty" in scales:
            r += scales["hand_vel_penalty"] * W.reward_hand_vel_penalty(
                self.palm_lin_vel, self.goal_obj_dist, self.flag)
        if "dof_vel" in scales:
            r += scales["dof_vel"] * W.reward_dof_vel(dof_vel_c)
        if "dof_acc" in scales:
            r += scales["dof_acc"] * W.reward_dof_acc(
                self.last_dof_vel, dof_vel_c, self.d4.control.action_dt)
        unknown = set(scales) - set(W.IMPLEMENTED_TERMS)
        if unknown:
            raise KeyError(f"reward scale set for unimplemented terms {unknown}")

        # tasks:1453-1455 publishes a scaled reward AND a metric for every term
        # that is active, and tasks:1467-1469 the three success counters.
        # SimToolReal's logging_utils groups the same content under
        # episode_cumulative and episode_final. Both shapes are published here so
        # neither reader is short, which is what the first cut got wrong.
        hand_q = self.robot.data.joint_pos[:, self.c2l][:, self.n_arm:]
        palm, _ = self._palm()
        ft = self._fingertips()
        terms = {
            "obj_finger": (W.reward_obj_finger(self.obj_finger_dist), self.obj_finger_dist),
            "obj_hand": (W.reward_obj_hand(self.obj_hand_dist), self.obj_hand_dist),
            "goal_obj": (W.reward_goal_obj(self.goal_obj_dist, self.flag), self.goal_obj_dist),
            "success_bonus": (W.reward_success_bonus(self.goal_obj_dist, self.flag), None),
            "terminal_bonus": (W.reward_terminal_bonus(self.achieved_buf), None),
            "finger_curl_reg": (W.reward_finger_curl_reg(hand_q, self.curled_q),
                                (hand_q - self.curled_q).norm(p=2, dim=-1)),
            "table_collision_penalty": (
                W.reward_table_collision_penalty(ft, palm),
                torch.cat([ft[:, :, 2], palm[:, 2:3]], dim=1).min(dim=1).values),
            "action_penalty": (W.reward_action_penalty(self.actions),
                               torch.mean(torch.abs(self.actions), dim=-1)),
            "fall_penalty": (W.reward_fall_penalty(self.goal_obj_dist, self.too_far), None),
            "hand_vel_penalty": (
                W.reward_hand_vel_penalty(self.palm_lin_vel, self.goal_obj_dist, self.flag),
                torch.sum(torch.square(self.palm_lin_vel), dim=-1)),
            "dof_vel": (W.reward_dof_vel(dof_vel_c),
                        torch.mean(torch.abs(dof_vel_c), dim=-1)),
            "dof_acc": (W.reward_dof_acc(self.last_dof_vel, dof_vel_c,
                                         self.d4.control.action_dt),
                        torch.mean(torch.abs(
                            (self.last_dof_vel - dof_vel_c)
                            / self.d4.control.action_dt), dim=-1)),
        }
        for name, (unscaled, metric) in terms.items():
            if name not in scales:
                continue
            self.extras["rewards_" + name] = scales[name] * unscaled
            if metric is not None:
                self.extras["metrics_" + name] = metric
        self.extras["rewards_total"] = r

        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes
        # SimToolReal groups these under episode_cumulative / episode_final for
        # rl_games. Dex4D's PPO calls .to(device) on every extras value at
        # ppo.py:304, so a nested dict raises. Same content, flat keys, which is
        # the shape the gym task already publishes.
        self.extras["done_too_far"] = (self.goal_obj_dist >= self.too_far).float()
        self.extras["done_timeout"] = (
            self.episode_length_buf >= self.d4.termination.episode_length).float()
        self.extras["metrics_goal_obj_dist"] = self.goal_obj_dist
        self.extras["metrics_contact_flag"] = self.flag.float()
        # tasks:1462, after every term has been read.
        self.achieved_buf = torch.where(self.goal_hit,
                                        torch.zeros_like(self.achieved_buf),
                                        self.achieved_buf)
        # tasks:1307, the last statement of post_physics_step.
        self.last_dof_vel = self.robot.data.joint_vel[:, self.c2l].clone()
        return r

    def _get_observations(self):
        opos, orot, palm, ft = self._intermediates()
        dof_pos = self.robot.data.joint_pos[:, self.c2l]
        dof_vel = self.robot.data.joint_vel[:, self.c2l]
        n = self.num_envs
        # Gym's palm and fingertip blocks are the full 13-wide rigid body state,
        # pose then linear then angular velocity. Filling only the position left
        # 56 channels at zero and the quaternions were not even unit.
        palm_state = torch.zeros(n, 13, device=self.device)
        palm_state[:, 0:3] = palm
        palm_state[:, 3:7] = S.quat_wxyz_to_xyzw(
            self.robot.data.body_quat_w[:, self.palm_idx])
        # gym reads `palm_center`'s own 13-wide state. The palm is 0.16 m out
        # from fr3_link7, so its linear velocity is link7's plus `w x r`, and
        # writing link7's alone was wrong by 0.32 m/s at a 2 rad/s wrist rate.
        # Angular velocity is shared across a rigid weld and needs no term.
        palm_state[:, 7:10] = self.palm_lin_vel
        palm_state[:, 10:13] = self.robot.data.body_ang_vel_w[:, self.palm_idx]
        # This block reads `self.fingertips`, the `*_link2` bodies, not the tips.
        ft_state = torch.zeros(n, self.d4.obs.num_fingertips, 13, device=self.device)
        ft_state[:, :, 0:3] = self._fingertip_bodies()
        ft_state[:, :, 3:7] = self._fingertip_quat_xyzw()
        ft_state[:, :, 7:10] = self.robot.data.body_lin_vel_w[:, self.ft_idx]
        ft_state[:, :, 10:13] = self.robot.data.body_ang_vel_w[:, self.ft_idx]
        object_pose = torch.cat([opos, orot], dim=1)
        # gym feeds PhysX's MEASURED dof force, `acquire_dof_force_tensor` at
        # tasks:144. IsaacLab's wrapper exposes no such thing, but the PhysX
        # view underneath does. `get_dof_projected_joint_forces` is the incoming
        # joint force projected on the motion axis, which is the same quantity.
        # `applied_torque`, the actuator's own PD effort, was a substitution and
        # it drops the constraint and contact share entirely.
        dof_force = self.robot.root_physx_view.get_dof_projected_joint_forces()
        dof_force = dof_force.to(self.device)[:, self.c2l]
        okp = O.keypoint_local_to_world(self.kp_local, opos, orot)
        O.write_states(
            self.states_buf, self.lay,
            dof_pos=dof_pos, dof_vel=dof_vel,
            dof_force=dof_force,
            dof_lower=self.dof_lower, dof_upper=self.dof_upper,
            fingertip_state=ft_state,
            # gym creates five 6-axis force sensors on the same `*_link2`
            # bodies with an identity `sensor_pose` (tasks:632-635), which
            # measures each body's incoming joint reaction in its own frame.
            # `get_link_incoming_joint_force` is that quantity, force then
            # torque in the child joint frame. IsaacLab's ContactSensor is not,
            # it reports net force with no torque, which is why this block sat
            # at zero until the PhysX view was found.
            force_sensor=self._fingertip_wrench(),
            palm_state=palm_state, actions=self.actions,
            object_pose=object_pose,
            object_linvel=self.object.data.root_lin_vel_w,
            object_angvel=self.object.data.root_ang_vel_w,
            goal_pos=self.goal_pos, goal_rot=self.goal_rot,
            object_pos=opos, object_rot=orot,
            object_keypoint_buf=self.kp_local,
            visual_feat=self.visual_feat,
            fingertip_object_vec=O.fingertip_to_object_vecs(ft, okp))
        obs = self.states_buf[:, : self.lay["num_obs"]]
        # base_task.py:172-173. The noise lands on `obs_buf` only, so the
        # critic's path through `get_state` stays clean under the symmetric
        # teacher. Applied after the buffer is filled and before the clip.
        if self.dr.enabled:
            obs = D.gaussian_noise(obs.clone(), self.obs_corr,
                                   self.dr.observations, self.sim_frame)
        return {"policy": torch.clamp(obs, -self.d4.obs.clip_observations,
                                      self.d4.obs.clip_observations)}

    def _reset_idx(self, env_ids):
        """`reset` at tasks:1174, in its order.

        The first cut of this wrote the goal and stopped, so the robot kept
        whatever configuration it had drifted into and the object never moved
        from where it fell. Every env then sat at a goal exactly 0.20 from a
        motionless object, which is below the 0.3 too-far line and above the
        0.05 success line, so nothing ever terminated. SimToolReal's lab reset
        does the same six writes this does, `reset_utils.py:377`.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        n = env_ids.numel()
        self._apply_randomizations(env_ids)

        # Robot. Reset noise is 0.0 in both stage yamls, so this is the default
        # pose exactly. Both target buffers are seeded from it, as at :1199.
        pos_c = self.default_dof_pos[env_ids]
        vel_c = torch.zeros_like(pos_c)
        self.robot.write_joint_state_to_sim(pos_c[:, self.l2c], vel_c,
                                            env_ids=env_ids)
        self.prev_targets[env_ids] = pos_c
        self.cur_targets[env_ids] = pos_c

        # Object. `reset_object_pose` at :1089, then the write the gym does
        # through the root state tensor.
        opos_new, orot_new = R.reset_object_pose(
            n, self.object_init_state[env_ids],
            self.d4.scene.delta_x_range, self.d4.scene.delta_y_range,
            self.d4.termination.object_init_stable_ratio, device=self.device)
        pose = torch.cat([opos_new + self._origins()[env_ids],
                          S.quat_xyzw_to_wxyz(orot_new)], dim=-1)
        self.object.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        # Goal, from the NEW object pose, as at :1214.
        gp, gr = R.reset_goal_pose_init(
            opos_new, orot_new,
            torch.tensor(self.d4.goal.displacement, device=self.device))
        self.goal_pos[env_ids], self.goal_rot[env_ids] = gp, gr
        self._write_goal_viz(env_ids)

        self.achieved_buf[env_ids] = 0
        self.successes[env_ids] = 0.0
        self.consecutive_successes[env_ids] = 0.0

        # tasks:1226-1230. `random_time` is a one-shot flag. The very first
        # reset covers every env, and each gets a uniform random episode phase
        # that persists for the whole run. Without it all envs time out on the
        # same step forever, and after the curriculum sets `too_far` to 1e6 at
        # iteration 15000 the timeout is the only terminator left.
        if self.random_time:
            self.random_time = False
            self.episode_length_buf[env_ids] = torch.randint(
                0, self.d4.termination.episode_length, (env_ids.numel(),),
                device=self.device)
