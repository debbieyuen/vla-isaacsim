from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from unitree import Unitree

world = World()
world.scene.add_default_ground_plane()

robot = Unitree(
    prim_path="/World/Unitree",
    name="unitree",
    usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/H1/h1.usd",
)

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()