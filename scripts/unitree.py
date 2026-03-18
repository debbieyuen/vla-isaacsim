from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from pxr import Gf, UsdGeom

class Unitree:
    def __init__(
        self,
        prim_path="/World/Unitree",
        name="unitree",
        usd_path=None,
        spawn_position=(0.0, 0.0, 1.08),
    ):
        if usd_path is None:
            raise ValueError("usd_path is required")

        self.prim_path = prim_path
        self.name = name
        self.usd_path = usd_path
        self.spawn_position = spawn_position

        add_reference_to_stage(self.usd_path, self.prim_path)
        self.set_spawn_pose(self.spawn_position)

    def set_spawn_pose(self, position=(0.0, 0.0, 1.08)):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            raise RuntimeError(f"Could not find prim at {self.prim_path}")

        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )