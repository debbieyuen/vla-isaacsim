from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from unitree import Unitree

world = World()

robot = Unitree(
    prim_path="/World/Unitree",
    name="unitree"
)

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()