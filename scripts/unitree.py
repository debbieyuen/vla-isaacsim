from isaacsim.core.articulations import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage

class Unitree:
    def __init__(self, prim_path="/World/Unitree", name="unitree"):
        # 🔑 THIS is what Franka does internally
        add_reference_to_stage(
            "omniverse://localhost/NVIDIA/Assets/Robots/Unitree/unitree_h1.usd",
            prim_path
        )

        # wrap as articulation
        self._articulation = Articulation(prim_path=prim_path)