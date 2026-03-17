from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.unitree import UnitreeH1
# from isaacsim.core.prims import XFormPrim
# from isaacsim.core.utils.stage import add_reference_to_stage

world = World()

# Franka robot instantiation
# load robot asset
# https://forums.developer.nvidia.com/t/how-to-add-my-own-usd-into-isaac-sim/251205/3
# https://forums.developer.nvidia.com/t/isaac-sim-how-to-import-usd-assets-into-a-scene/201253/4

robot = UnitreeH1(
    prim_path="/World/UnitreeH1",
    name="unitree_h1"
)

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()