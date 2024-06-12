from isaacgym import gymutil
from isaacgym import gymapi
import random
import numpy as np


gym = gymapi.acquire_gym()


"""
sim_params = gymapi.SimParams()

sim_params.dt = 1.0/60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
"""

args = gymutil.parse_arguments(description="Body Physics Properties Example")
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.use_gpu_pipeline = False


sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)


plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

gym.add_ground(sim, plane_params)


asset_root = "lexxgym/urdf"
asset_file = "ball.urdf"
# asset_file = "urdf/simple_robot.urdf"
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
asset_options.armature = 0.01

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 2.0)

    actor_handle = gym.create_actor(env, asset, pose, None)
    body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
    for b in range(len(body_props)):
        body_props[b].flags = gymapi.RIGID_BODY_NONE
    gym.set_actor_rigid_body_properties(env, actor_handle, body_props)
    color = gymapi.Vec3(1, 0, 0)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    # actor_handles.append(actor_handle)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)