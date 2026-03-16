from isaacsim import SimulationApp

# Launch Isaac Sim headless 
simulation_app = SimulationApp({
    "headless": False
})

from isaacsim.core.api import World

world = World()
world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()