# # run inside Isaac Sim session (after world.reset())
# from omni.usd import get_context
# from pxr import Usd, UsdGeom
# import time

# stage = get_context().get_stage()

# # attempt to load payloads of the /World/Franka prim
# prim = stage.GetPrimAtPath("/World/Franka")
# if prim and prim.IsValid():
#     try:
#         stage.Load(prim.GetPath())
#     except Exception:
#         pass
# time.sleep(0.15)

# # list meshes under the prim
# meshes = []
# if prim and prim.IsValid():
#     for p in Usd.PrimRange(prim):
#         if p.IsA(UsdGeom.Mesh):
#             meshes.append(p.GetPath().pathString)
# print("Meshes under /World/Franka:", len(meshes))
# for m in meshes[:200]:
#     print("  ", m)

# # also search for any frank(a)/panda prims in the stage
# found = []
# for p in Usd.PrimRange(stage.GetPseudoRoot()):
#     name = p.GetName().lower()
#     if ("frank" in name or "panda" in name) and p.IsValid():
#         found.append(p.GetPath().pathString)
# print("Potential frank/panda prims found:", found[:20])