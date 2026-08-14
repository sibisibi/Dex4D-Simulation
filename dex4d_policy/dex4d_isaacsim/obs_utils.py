"""Observation assembly.

states_buf is 3*ndof + fingertip states and wrenches + 13 palm + ndof actions
+ 20 object and goal + keypoints + 64 visual + fingertip-to-object vectors.
obs_buf is the same when symmetric, otherwise 2*ndof + actions + masked keypoints.
"""
from __future__ import annotations

import torch

VEL_OBS_SCALE = 0.2
FORCE_TORQUE_OBS_SCALE = 10.0


def layout(num_robot_dofs: int, num_fingertips: int, num_keypoints: int,
           kp_downsample_ratio: int, asymmetric: bool) -> dict:
    """Sizes and offsets of every block."""
    num_ft_states = 13 * num_fingertips
    num_ft_ft = 6 * num_fingertips
    num_kp_flatten = num_keypoints * 6
    num_masked_kp_flatten = 6 * (num_keypoints // kp_downsample_ratio)
    num_states = 283 + num_kp_flatten
    num_obs = 57 + num_masked_kp_flatten if asymmetric else num_states
    off = {}
    p = 0
    for name, size in (("dof", 3 * num_robot_dofs),
                       ("fingertip", num_ft_states + num_ft_ft),
                       ("palm", 13),
                       ("action", num_robot_dofs),
                       ("object_goal", 20 + num_kp_flatten),
                       ("visual", 64),
                       ("fingertip_object_vec", 3 * num_fingertips)):
        off[name] = (p, size)
        p += size
    assert p == num_states, f"state layout sums to {p}, expected {num_states}"
    return dict(num_states=num_states, num_obs=num_obs,
                num_kp_flatten=num_kp_flatten,
                num_masked_kp_flatten=num_masked_kp_flatten,
                kp_start=off["object_goal"][0] + 20, offsets=off)


def quat_mul(a, b):
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)


def quat_conjugate(q):
    return torch.cat([-q[:, :3], q[:, 3:]], dim=-1)


def unscale(x, lower, upper):
    """Map [lower, upper] onto [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


def keypoint_local_to_world(keypoints, object_pos, object_rot):
    """xyzw rotation applied to (N, K, 3) local keypoints."""
    q = object_rot
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.stack([
        torch.cat([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=1),
        torch.cat([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=1),
        torch.cat([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=1),
    ], dim=1)
    return torch.einsum("nij,nkj->nki", r, keypoints) + object_pos[:, None, :]


def fingertip_to_object_vecs(fingertip_pos, keypoints):
    """Vector from each fingertip to its nearest object keypoint."""
    n, k_ft, _ = fingertip_pos.shape
    vecs = keypoints.unsqueeze(1) - fingertip_pos.unsqueeze(2)
    nearest = vecs.norm(dim=-1).argmin(dim=-1)
    return vecs[torch.arange(n, device=vecs.device).unsqueeze(1),
                torch.arange(k_ft, device=vecs.device).unsqueeze(0), nearest]


def write_states(buf, lay, *, dof_pos, dof_vel, dof_force, dof_lower, dof_upper,
                 fingertip_state, force_sensor, palm_state, actions,
                 object_pose, object_linvel, object_angvel,
                 goal_pos, goal_rot, object_pos, object_rot,
                 object_keypoint_buf, visual_feat, fingertip_object_vec):
    """Fill states_buf. Returns the object and goal keypoints in world frame."""
    o = lay["offsets"]
    n_dof = dof_pos.shape[1]

    p, _ = o["dof"]
    buf[:, p:p + n_dof] = unscale(dof_pos, dof_lower, dof_upper)
    buf[:, p + n_dof:p + 2 * n_dof] = VEL_OBS_SCALE * dof_vel
    buf[:, p + 2 * n_dof:p + 3 * n_dof] = FORCE_TORQUE_OBS_SCALE * dof_force[:, :n_dof]

    p, size = o["fingertip"]
    n_ft = fingertip_state.shape[1] * 13
    buf[:, p:p + n_ft] = fingertip_state.reshape(buf.shape[0], n_ft)
    buf[:, p + n_ft:p + size] = FORCE_TORQUE_OBS_SCALE * force_sensor[:, :size - n_ft]

    p, _ = o["palm"]
    buf[:, p:p + 10] = palm_state[:, 0:10]
    buf[:, p + 10:p + 13] = VEL_OBS_SCALE * palm_state[:, 10:13]

    p, _ = o["action"]
    buf[:, p:p + n_dof] = actions[:, :n_dof]

    p, _ = o["object_goal"]
    buf[:, p:p + 3] = object_pose[:, 0:3]
    buf[:, p + 3:p + 7] = object_pose[:, 3:7]
    buf[:, p + 7:p + 10] = object_linvel
    buf[:, p + 10:p + 13] = VEL_OBS_SCALE * object_angvel
    buf[:, p + 13:p + 16] = goal_pos - object_pos
    buf[:, p + 16:p + 20] = quat_mul(goal_rot, quat_conjugate(object_rot))
    okp = keypoint_local_to_world(object_keypoint_buf, object_pos, object_rot)
    gkp = keypoint_local_to_world(object_keypoint_buf, goal_pos, goal_rot)
    half = lay["num_kp_flatten"] // 2
    buf[:, p + 20:p + 20 + half] = okp.reshape(buf.shape[0], -1)
    buf[:, p + 20 + half:p + 20 + lay["num_kp_flatten"]] = gkp.reshape(buf.shape[0], -1)

    p, _ = o["visual"]
    buf[:, p:p + 64] = 0.1 * visual_feat

    p, size = o["fingertip_object_vec"]
    buf[:, p:p + size] = fingertip_object_vec.reshape(buf.shape[0], size)
    return okp, gkp
