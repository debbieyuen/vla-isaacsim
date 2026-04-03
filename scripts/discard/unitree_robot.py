from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from unitree import Unitree

world = World()
world.scene.add_default_ground_plane()

robot = Unitree(
    prim_path="/World/Unitree",
    name="unitree",
    usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/H1/h1.usd",
)

world.reset()

# art = Articulation("/World/Unitree")
# art.initialize()

# print joints
# joint_names = art.get_joint_names()
# print("Joint names:")
# for i, name in enumerate(joint_names):
#     print(i, name)

# standing_pos = [0.0] * art.num_dof  # replace with standing pose
# art.set_joint_positions(standing_pos)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()