import numpy as np
import math
import torch
from isaacgym import gymutil, gymapi
from isaacgym.torch_utils import *
import shutil, os
import trimesh

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def visualize_point_isaacgym(env, env_points, radius=0.01, selected_ids=None):
    env.gym.clear_lines(env.viewer)
    # Create helper geometry used for visualization
    # Create an wireframe axis
    axes_geom = gymutil.AxesGeometry(0.15)
    # Create a wireframe sphere
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    yellow_geom = gymutil.WireframeSphereGeometry(radius, 12, 12, sphere_pose, color=(1, 1, 0))
    if selected_ids is None:
        selected_ids = range(env.num_envs)
    for j in selected_ids:
        points = env_points[j]
        for p in points:
            pos = gymapi.Transform(gymapi.Vec3(p[0], p[1], p[2]), gymapi.Quat())
            gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[j], pos)

def visualize_axes_isaacgym(env, env_poses, axis_length=0.1, selected_ids=None):
    env.gym.clear_lines(env.viewer)
    # Create helper geometry used for visualization
    # Create an wireframe axis
    axes_geom = gymutil.AxesGeometry(axis_length)
    if selected_ids is None:
        selected_ids = range(env.num_envs)
    for j in selected_ids:
        pose = env_poses[j]
        pose = gymapi.Transform(gymapi.Vec3(pose[0], pose[1], pose[2]), gymapi.Quat(pose[3], pose[4], pose[5], pose[6]))
        gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[j], pose)

def copy_files(args, logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    shutil.copy2("cfg/my_train_set.yaml", logdir)
    shutil.copy2("utils/process_sarl.py", logdir)

    if args.task == "XArm6LeapHandAP2AP":
        shutil.copy2("algorithms/rl/ppo/ppo.py", logdir)
        shutil.copy2("algorithms/rl/ppo/module.py", logdir)
        shutil.copy2("script/run_train_ppo_state.sh", logdir)
        shutil.copy2("tasks/xarm6_leap_hand_ap2ap.py", logdir)
        shutil.copy2("cfg/xarm6_leap_hand_ap2ap.yaml", logdir)
        shutil.copy2("cfg/ppo/config.yaml", logdir)
    elif args.task == "XArm6LeapHandAP2APVision":
        shutil.copy2("algorithms/rl/dagger/dagger.py", logdir)
        shutil.copy2("algorithms/rl/dagger/module.py", logdir)
        shutil.copy2("script/run_train_dagger.sh", logdir)
        shutil.copy2("tasks/xarm6_leap_hand_ap2ap.py", logdir)
        shutil.copy2("tasks/xarm6_leap_hand_ap2ap_vision.py", logdir)
        shutil.copy2("cfg/xarm6_leap_hand_ap2ap_vision.yaml", logdir)
        shutil.copy2("cfg/dagger/config.yaml", logdir)

def sample_position(pos, dis_range):
    '''
    Sample a position in a cube range of [-dis_range, dis_range]
    '''
    xyz_bound = torch.tensor([
        [-0.3, 1.0],
        [-0.5, 0.5],
        [0.65, 1.1]
    ]).to(pos.device)
    noise = torch.FloatTensor(pos.shape).uniform_(-dis_range, dis_range).to(pos.device)
    new_pos = pos + noise
    new_pos = torch.clamp(new_pos, min=xyz_bound[:, 0], max=xyz_bound[:, 1])
    return new_pos

def sample_rotation(quat, angle_range):
    '''
    Sample a rotation noise in a range of [-angle_range, angle_range] (in radian)
    xyzw convention
    '''
    axis = torch.randn(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    theta = torch.rand(quat.shape[0], device=quat.device, dtype=quat.dtype) * angle_range
    dq_xyz = axis * torch.sin(theta / 2.0).unsqueeze(-1)
    dq_w = torch.cos(theta / 2.0).unsqueeze(-1)
    dq = torch.cat([dq_xyz, dq_w], dim=-1) # [num_envs, 4]
    new_quat = quat_mul(dq, quat)
    new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)

    return new_quat

def compute_keypoints(object_pos, object_rot, cube_size: float = 0.1):
    '''
    Compute 8 keypoints of a cube object
    object_pos: [num_envs, 3]
    object_rot: [num_envs, 4] xyzw
    return: keypoints [num_envs, 8, 3]
    '''
    num_envs = object_pos.shape[0]
    local_keypoints = torch.tensor([[1, 1, 1],
                                    [1, 1, -1],
                                    [1, -1, 1],
                                    [1, -1, -1],
                                    [-1, 1, 1],
                                    [-1, 1, -1],
                                    [-1, -1, 1],
                                    [-1, -1, -1]], device=object_pos.device, dtype=object_pos.dtype) * (cube_size / 2.0) # [8, 3]

    local_keypoints = local_keypoints.unsqueeze(0).repeat(num_envs, 1, 1) # [num_envs, 8, 3]
    world_keypoints = local_keypoints.clone()
    for i in range(local_keypoints.shape[1]):
        world_keypoints[:, i, :] = quat_rotate(object_rot, local_keypoints[:, i, :]) + object_pos # [num_envs, 3]

    return world_keypoints # [num_envs, 8, 3]

@torch.jit.script
def quat_rotate_batch(q, vecs):
    '''
    Rotate a batch of vectors by a batch of quaternions
    q: [B, 4] xyzw
    vecs: [B, N, 3]
    return: [B, N, 3]
    '''
    shape = q.shape
    q_w = q[:, -1]  # [B]
    q_vec = q[:, :3]  # [B, 3]
    
    # Expand dimensions for broadcasting
    q_w = q_w.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    q_vec = q_vec.unsqueeze(1)  # [B, 1, 3]
    
    a = vecs * (2.0 * q_w ** 2 - 1.0)  # [B, N, 3]
    b = torch.cross(q_vec.expand_as(vecs), vecs, dim=-1) * q_w * 2.0  # [B, N, 3]
    c = q_vec * torch.sum(q_vec * vecs, dim=-1, keepdim=True) * 2.0  # [B, N, 3]
    
    return a + b + c
    

def keypoint_local_to_world(keypoints, object_pos, object_rot):
    '''
    Transform keypoints from local frame to world frame
    keypoints: [num_envs, num_keypoints, 3]
    object_pos: [num_envs, 3]
    object_rot: [num_envs, 4] xyzw
    return: keypoints [num_envs, num_keypoints, 3]
    '''
    num_envs = object_pos.shape[0]
    num_keypoints = keypoints.shape[1]
    world_keypoints = keypoints.clone()
    
    # for i in range(num_keypoints):
    #     world_keypoints[:, i, :] = quat_rotate(object_rot, keypoints[:, i, :]) + object_pos # [num_envs, 3]
    world_keypoints = quat_rotate_batch(object_rot, keypoints) + object_pos.unsqueeze(1) # [num_envs, num_keypoints, 3], parallel version

    return world_keypoints # [num_envs, num_keypoints, 3]

def fps_sampling(points, num_samples, seed=42):
    """
    Deterministic Farthest Point Sampling on point cloud
    
    Args:
        points: numpy array of shape (N, 3) representing 3D points
        num_samples: number of points to sample
        seed: random seed for deterministic behavior
        
    Returns:
        sampled_indices: indices of sampled points
        sampled_points: sampled point coordinates
    """
    np.random.seed(seed)
    n_points = points.shape[0]
    
    if num_samples >= n_points:
        return np.arange(n_points), points
    
    # Initialize with a random point
    sampled_indices = [np.random.randint(0, n_points)]
    sampled_points = [points[sampled_indices[0]]]
    
    # Initialize distances to infinity
    distances = np.full(n_points, np.inf)
    
    for i in range(1, num_samples):
        # Update distances to the nearest sampled point
        last_point = points[sampled_indices[-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select the point with maximum distance to sampled points
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
        sampled_points.append(points[farthest_idx])
    
    return np.array(sampled_indices), np.array(sampled_points)

def extract_mesh_keypoints(mesh_file_path, num_keypoints=64, seed=42):
    """
    Extract keypoints from mesh file using FPS sampling
    
    Args:
        mesh_file_path: path to the mesh file
        num_keypoints: number of keypoints to sample
        seed: random seed for deterministic behavior
        
    Returns:
        keypoints: numpy array of shape (num_keypoints, 3)
    """
    try:
        # Load mesh
        mesh = trimesh.load(mesh_file_path)
        
        # Sample points from mesh surface
        # Use a large number of points for better coverage
        surface_points, _ = trimesh.sample.sample_surface(mesh, 10000)
        
        # Apply FPS sampling
        _, keypoints = fps_sampling(surface_points, num_keypoints, seed)
        
        return keypoints
    except Exception as e:
        raise Exception(f"Error processing mesh {mesh_file_path}: {e}")
    

def mask_keypoints_oneside(keypoints: torch.Tensor, downsample_ratio=2):
    '''
    generate masked keypoints on one side randomly
    '''
    # keypoints: [num_keypoints, 3]
    num_keypoints = keypoints.shape[0]
    num_left_keypoints = num_keypoints // downsample_ratio
    masked_keypoints = keypoints.clone()
    # Randomly generate one plane to mask one side of keypoints
    plane_normal = torch.randn(3, device=keypoints.device, dtype=keypoints.dtype)
    plane_normal = plane_normal / torch.norm(plane_normal)
    plane_point = keypoints[torch.randint(0, num_keypoints, (1,)).item(), :]  # Random point on the plane
    # Mask keypoints on one side of the plane
    dists = torch.matmul(keypoints - plane_point, plane_normal)
    masked_keypoints[dists > 0] = 0
    
    # if more keypoints are masked than required, randomly unmask some, else randomly mask more
    current_left = (masked_keypoints.abs().sum(dim=-1) != 0).sum(dim=-1)
    if current_left < num_left_keypoints: # unmask some
        unmask_indices = (masked_keypoints.abs().sum(dim=-1) == 0).nonzero(as_tuple=False).squeeze()
        unmask_indices = unmask_indices[torch.randperm(unmask_indices.shape[0])[:(num_left_keypoints - current_left)]]
        masked_keypoints[unmask_indices] = keypoints[unmask_indices]
    elif current_left > num_left_keypoints: # mask more
        mask_indices = (masked_keypoints.abs().sum(dim=-1) != 0).nonzero(as_tuple=False).squeeze()
        mask_indices = mask_indices[torch.randperm(mask_indices.shape[0])[:(current_left - num_left_keypoints)]]
        masked_keypoints[mask_indices] = 0

    # make masked_keypoints have shape [num_envs, num_keypoints // downsample_ratio, 3]
    masked_keypoints = masked_keypoints[masked_keypoints.abs().sum(dim=-1) != 0].view(-1, 3)
    assert masked_keypoints.shape[0] == num_left_keypoints, f"Masked keypoints shape mismatch: expected {num_left_keypoints}, got {masked_keypoints.shape[0]}"

    return masked_keypoints

def mask_keypoints_test_time(envs_keypoints: torch.Tensor, max_mask_prob=0.2):
    '''
    generate masked keypoints for test time
    envs_keypoints: [num_envs, num_keypoints, 3]
    '''
    num_envs, num_keypoints, _ = envs_keypoints.shape
    mask_prob = torch.rand(1).item() * max_mask_prob
    num_masked_keypoints = int(num_keypoints * mask_prob)
    masked_keypoints = envs_keypoints.clone()
    rand_gaussian_noise = torch.randn_like(envs_keypoints) * 0.001 # small gaussian noise for all keypoints for sim2real
    masked_keypoints += rand_gaussian_noise
    for i in range(num_envs):
        mask_indices = torch.randperm(num_keypoints)[:num_masked_keypoints]
        masked_keypoints[i, mask_indices, :] = 0 # set xyz to 0 to indicate masked keypoints
    return masked_keypoints


def mask_object_goal_keypoints_test_time(envs_object_keypoints: torch.Tensor, envs_goal_keypoints: torch.Tensor, max_mask_prob=0.4):
    '''
    generate masked keypoints for test time
    envs_object_keypoints: [num_envs, num_keypoints, 3]
    envs_goal_keypoints: [num_envs, num_keypoints, 3]
    '''
    num_envs, num_keypoints, _ = envs_object_keypoints.shape
    mask_prob = torch.rand(1).item() * max_mask_prob
    num_masked_keypoints = int(num_keypoints * mask_prob)
    masked_object_keypoints = envs_object_keypoints.clone()
    masked_goal_keypoints = envs_goal_keypoints.clone()
    rand_object_gaussian_noise = torch.randn_like(envs_object_keypoints) * 0.001 # small gaussian noise for all keypoints for sim2real
    rand_goal_gaussian_noise = torch.randn_like(envs_goal_keypoints) * 0.001 # small gaussian noise for all keypoints for sim2real
    masked_object_keypoints += rand_object_gaussian_noise
    masked_goal_keypoints += rand_goal_gaussian_noise
    for i in range(num_envs):
        mask_indices = torch.randperm(num_keypoints)[:num_masked_keypoints]
        masked_object_keypoints[i, mask_indices, :] = 0 # set xyz to 0 to indicate masked keypoints
        masked_goal_keypoints[i, mask_indices, :] = 0 # set xyz to 0 to indicate masked keypoints
    return masked_object_keypoints, masked_goal_keypoints


def mask_object_goal_keypoints_random_height_test_time(envs_object_keypoints: torch.Tensor, envs_goal_keypoints: torch.Tensor, max_mask_height=0.5, above_plane_mask_prob=0.9, below_plane_mask_prob=0.05, noise_std=0.001):
    '''
    generate masked keypoints for test time
    envs_object_keypoints: [num_envs, num_keypoints, 3]
    envs_goal_keypoints: [num_envs, num_keypoints, 3]
    '''
    num_envs, num_keypoints, _ = envs_object_keypoints.shape
    masked_object_keypoints = envs_object_keypoints.clone()
    masked_goal_keypoints = envs_goal_keypoints.clone()
    rand_object_gaussian_noise = torch.randn_like(envs_object_keypoints) * noise_std
    rand_goal_gaussian_noise = torch.randn_like(envs_goal_keypoints) * noise_std
    masked_object_keypoints += rand_object_gaussian_noise
    masked_goal_keypoints += rand_goal_gaussian_noise

    device = envs_object_keypoints.device
    dtype = envs_object_keypoints.dtype
    mask_prob = torch.rand(num_envs, device=device, dtype=dtype) * max_mask_height
    z_vals = envs_object_keypoints[..., 2]
    z_sorted, _ = z_vals.sort(dim=1)
    k = ((1 - mask_prob) * (num_keypoints - 1)).to(torch.long)
    plane_height = z_sorted[torch.arange(num_envs, device=device), k]
    above_plane = z_vals > plane_height.unsqueeze(1)
    rand = torch.rand((num_envs, num_keypoints), device=device, dtype=dtype)
    mask = torch.where(above_plane, rand < above_plane_mask_prob, rand < below_plane_mask_prob)
    masked_object_keypoints[mask] = 0
    masked_goal_keypoints[mask] = 0

    return masked_object_keypoints, masked_goal_keypoints


def format_time(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def compute_fingertip_to_object_vecs(fingertip_pos, keypoints):
    '''
    Compute the vector from fingertips to object (keypoints)
    Basically, it's the vector from each fingertip to the closest keypoint on the object
    fingertip_pos: [num_envs, num_fingertips, 3]
    keypoints: [num_envs, num_keypoints, 3]
    return: distances [num_envs, num_fingertips, 3]
    '''
    num_envs, num_fingertips, _ = fingertip_pos.shape
    num_keypoints = keypoints.shape[1]
    vecs = torch.zeros((num_envs, num_fingertips, 3), device=fingertip_pos.device, dtype=fingertip_pos.dtype)
    # For each fingertip, find the closest keypoint and compute the vector, no for-loop
    fingertip_expanded = fingertip_pos.unsqueeze(2)  # [num_envs, num_fingertips, 1, 3]
    keypoints_expanded = keypoints.unsqueeze(1)      # [num_envs, 1, num_keypoints, 3]

    # Compute vectors from each fingertip to all keypoints
    vecs = keypoints_expanded - fingertip_expanded  # [num_envs, num_fingertips, num_keypoints, 3]

    # Find the closest keypoint for each fingertip
    closest_keypoints = vecs.norm(dim=-1).argmin(dim=-1)  # [num_envs, num_fingertips]

    # Gather the vectors corresponding to the closest keypoints
    vecs = vecs[torch.arange(num_envs).unsqueeze(1), torch.arange(num_fingertips).unsqueeze(0), closest_keypoints]  # [num_envs, num_fingertips, 3]

    return vecs


# Smooth object trajectory using Bezier curves
def smooth_obj_traj(goal_poses: torch.Tensor, num_waypoints: int = 20):
    '''
    Smooth object trajectory

    goal_poses example:
    torch.tensor([
            [0, 0, 0.9, 0, 0, 0],
            [0, 0, 0.8, 0, 0, 0],
            [0, 0.1, 0.8, 0, 0, 0],
            [0, 0.2, 0.8, 0, 0, 0],
            [0, 0.2, 0.7, 0, 0, 0],
        ]) # [num_goals, 6]
    num_waypoints: number of samples per segment (>= 2), including endpoints
    '''
    if goal_poses.ndim != 2 or goal_poses.shape[1] != 6:
        raise ValueError(f"goal_poses should have shape [num_goals, 6], got {goal_poses.shape}")
    if num_waypoints < 2:
        raise ValueError(f"num_waypoints should be >= 2, got {num_waypoints}")

    # If there is nothing to smooth, return as-is.
    if goal_poses.shape[0] <= 1:
        return goal_poses

    device = goal_poses.device
    dtype = goal_poses.dtype
    two_pi = 2.0 * math.pi

    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + math.pi, two_pi) - math.pi

    # Unwrap euler angles to avoid jumps across the -pi/pi boundary before interpolation.
    unwrapped = goal_poses.clone()
    for i in range(1, unwrapped.shape[0]):
        delta = _wrap_to_pi(unwrapped[i, 3:] - unwrapped[i - 1, 3:])
        unwrapped[i, 3:] = unwrapped[i - 1, 3:] + delta

    t_values = torch.linspace(0.0, 1.0, num_waypoints, device=device, dtype=dtype)
    smoothed = []
    num_goals = unwrapped.shape[0]

    for i in range(num_goals - 1):
        p0 = unwrapped[i - 1] if i - 1 >= 0 else unwrapped[i]
        p1 = unwrapped[i]
        p2 = unwrapped[i + 1]
        p3 = unwrapped[i + 2] if i + 2 < num_goals else unwrapped[i + 1]

        # Catmull-Rom to cubic Bezier conversion (centripetal with tension=0.5)
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2

        for j, t in enumerate(t_values):
            if i > 0 and j == 0:
                continue  # avoid duplicating the start of the segment
            omt = 1.0 - t
            point = (
                (omt ** 3) * b0
                + 3 * (omt ** 2) * t * b1
                + 3 * omt * (t ** 2) * b2
                + (t ** 3) * b3
            )
            smoothed.append(point)

    smoothed = torch.stack(smoothed, dim=0)
    smoothed[:, 3:] = _wrap_to_pi(smoothed[:, 3:])  # wrap angles back to [-pi, pi]
    return smoothed
