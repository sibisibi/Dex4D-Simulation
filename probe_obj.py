from isaacgym import gymapi
import sys
urdf_dir, urdf_file = sys.argv[1], sys.argv[2]
gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.up_axis = gymapi.UP_AXIS_Z
sp.physx.use_gpu = True
sp.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
o = gymapi.AssetOptions()
o.density = 500
o.use_mesh_materials = True
o.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
o.override_com = True
o.override_inertia = True
a = gym.load_asset(sim, urdf_dir, urdf_file, o)
print("LOADED", gym.get_asset_rigid_body_count(a), gym.get_asset_rigid_shape_count(a))
env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
ac = gym.create_actor(env, a, gymapi.Transform(), "o", 0, 0, 1)
print("ACTOR_OK")
import os; os._exit(0)
