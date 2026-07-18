"""020 replay renderer for Dex4D, kinematic playback of a saved rollout.

Mirrors simtoolreal-020/dextoolbench/render_replay_isaacgym.py: rebuild the
Dex4D scene (table, XArm6+Leap, object, red goal ghost) with gravity zeroed,
then per logged step write robot dof state, object root pose, and goal root
pose into the sim tensors, pin the PD targets to the same joints, run ONE
physics tick so PhysX propagates dof state into link transforms, and capture
the camera. States are rewritten every frame, so the tick cannot drift.

Camera matches the locked SimToolReal framing shifted +0.22 for Dex4D's 0.6
table top: eye (0, -1, 1.25), lookat (0, 0, 0.75), horizontal fov 78.9 deg,
1600x912, 60 fps mp4.

    python render_replay_020.py --npz <unit>/poses.npz --traj 5 \
        --object_urdf <abs path> --out .../videos_dex4d/<safe_key>_dex4d_traj5.mp4
"""

# isort: off
from isaacgym import gymapi, gymtorch
import torch
# isort: on

import argparse
import os
import os.path as osp

import cv2
import numpy as np

TABLE_DIMS = (1.2, 1.2, 0.6)
ROBOT_URDF = "urdf/xarm6_leap_description/xarm6_leap_right_2023.urdf"
CAM_EYE = (0.0, -1.0, 1.25)
CAM_LOOKAT = (0.0, 0.0, 0.75)
CAM_W, CAM_H = 1600, 912
CAM_FOV_DEG = 78.9


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--traj', type=int, required=True)
    p.add_argument('--object_urdf', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dump_frame', default='', help='optional png path for frame 0')
    args = p.parse_args()

    d = np.load(args.npz)
    joints = d[f'joint_{args.traj}']  # (T, 22)
    obj = d[f'obj_{args.traj}']       # (T, 7) pos + quat xyzw
    goal = d[f'goal_{args.traj}']     # (T, 7)
    T = len(joints)
    assert T == len(obj) == len(goal) and T > 0

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    # zero gravity keeps written states exact between rewrites
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.solver_type = 1
    sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None

    asset_root = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..', 'assets'))

    robot_opts = gymapi.AssetOptions()
    robot_opts.fix_base_link = True
    robot_opts.collapse_fixed_joints = False
    robot_opts.disable_gravity = True
    robot_asset = gym.load_asset(sim, asset_root, ROBOT_URDF, robot_opts)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    assert num_dofs == joints.shape[1], (num_dofs, joints.shape)

    obj_opts = gymapi.AssetOptions()
    obj_opts.fix_base_link = False
    obj_opts.use_mesh_materials = True
    obj_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    urdf_dir, urdf_file = osp.split(osp.abspath(args.object_urdf))
    object_asset = gym.load_asset(sim, urdf_dir, urdf_file, obj_opts)
    goal_opts = gymapi.AssetOptions()
    goal_opts.fix_base_link = False
    goal_opts.use_mesh_materials = True
    goal_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    goal_opts.disable_gravity = True
    goal_asset = gym.load_asset(sim, urdf_dir, urdf_file, goal_opts)

    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, *TABLE_DIMS, table_opts)

    env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0.0), gymapi.Vec3(1.5, 1.5, 1.5), 1)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(-0.5, 0.0, TABLE_DIMS[2])
    robot_actor = gym.create_actor(env, robot_asset, robot_pose, 'robot', 0, -1, 0)
    # position drive on every dof, task eval gains (useOSCControl false)
    props = gym.get_actor_dof_properties(env, robot_actor)
    arm_stiff = [100, 100, 64, 64, 64, 40]
    for i in range(num_dofs):
        props['driveMode'][i] = gymapi.DOF_MODE_POS
        props['stiffness'][i] = arm_stiff[i] if i < 6 else 3.0
        props['damping'][i] = 1.0 if i < 6 else 0.5
    gym.set_actor_dof_properties(env, robot_actor, props)

    start = gymapi.Transform()
    start.p = gymapi.Vec3(*obj[0, 0:3])
    object_actor = gym.create_actor(env, object_asset, start, 'object', 0, 0, 0)
    gym.set_rigid_body_color(env, object_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(90 / 255, 94 / 255, 173 / 255))
    goal_actor = gym.create_actor(env, goal_asset, start, 'goal_object', 1, 0, 0)
    gym.set_rigid_body_color(env, goal_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * TABLE_DIMS[2])
    table_actor = gym.create_actor(env, table_asset, table_pose, 'table', 0, -1, 0)
    gym.set_rigid_body_color(env, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(150 / 255, 150 / 255, 150 / 255))

    obj_root = gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM)
    goal_root = gym.get_actor_index(env, goal_actor, gymapi.DOMAIN_SIM)

    cam_props = gymapi.CameraProperties()
    cam_props.width, cam_props.height = CAM_W, CAM_H
    cam_props.horizontal_fov = CAM_FOV_DEG
    cam = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam, env, gymapi.Vec3(*CAM_EYE), gymapi.Vec3(*CAM_LOOKAT))

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
    targets = torch.zeros(dof.shape[0], dtype=torch.float32)

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), 60, (CAM_W, CAM_H))
    for t in range(T):
        jt = torch.tensor(joints[t], dtype=torch.float32)
        dof[:num_dofs, 0] = jt
        dof[:, 1] = 0.0
        root[obj_root, 0:7] = torch.tensor(obj[t], dtype=torch.float32)
        root[obj_root, 7:13] = 0.0
        root[goal_root, 0:7] = torch.tensor(goal[t], dtype=torch.float32)
        root[goal_root, 7:13] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof))
        targets[:num_dofs] = jt
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(targets))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
        assert img.size > 0, 'empty camera image'
        frame = img.reshape(CAM_H, CAM_W, 4)[:, :, :3]
        if t == 0 and args.dump_frame:
            cv2.imwrite(args.dump_frame, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f'[replay] wrote {args.out} ({T} frames, logged rollout verbatim)')
    os._exit(0)


if __name__ == '__main__':
    main()
