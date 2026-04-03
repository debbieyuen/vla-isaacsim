from isaacsim.core.utils.stage import add_reference_to_stage

class Unitree:
    def __init__(self, prim_path="/World/Unitree", name="unitree", usd_path=None):
        if usd_path is None:
            raise ValueError("usd_path is required")

        self.prim_path = prim_path
        self.name = name

        add_reference_to_stage(usd_path, prim_path)