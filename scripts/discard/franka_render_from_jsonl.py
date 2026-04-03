# scripts/render_from_jsonl.py
"""
Render frames from joint-only .jsonl files into images + index.
Run with Isaac Sim python.bat (not system Python).

Example:
cd <isaac_sim_root>
.\python.bat C:\path\to\this\repo\scripts\render_from_jsonl.py --raw_dir C:\path\to\data\raw --out_dir C:\path\to\data\processed --headless False
"""

import argparse, json, os, base64, io, time
from pathlib import Path
from PIL import Image
import numpy as np

# Isaac sim imports - these require running with Isaac Sim python.bat
try:
    from isaacsim import SimulationApp
    from isaacsim.core.api import World
    # Franka wrapper (recommended)
    try:
        from isaacsim.robot.manipulators.examples.franka import Franka
    except Exception:
        # older namespace fallback
        try:
            from omni.isaac.franka.franka import Franka  # may exist in some releases
        except Exception:
            Franka = None
    # Articulation view (preferred API to set joint positions)
    try:
        from omni.isaac.core.articulations import ArticulationView
    except Exception:
        ArticulationView = None

    # camera helper imports
    try:
        from omni.isaac.core.utils.viewport import add_camera_to_stage, get_active_viewport
    except Exception:
        add_camera_to_stage = None
        get_active_viewport = None

    HAS_ISAAC = True
except Exception as e:
    print("ERROR: this script must be run with Isaac Sim's python (python.bat).")
    print("Import error:", e)
    HAS_ISAAC = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_proprio_len", type=int, default=50)
    p.add_argument("--steps_per_frame", type=int, default=4, help="physics steps to settle pose")
    p.add_argument("--headless", action="store_true", help="run headless (no GUI)")
    return p.parse_args()

def read_jsonl_files(raw_dir):
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield str(f), i, obj

def get_joint_vector_from_record(rec, joint_key_candidates=("joints","joint_positions","joints_list")):
    # Accept various names and formats
    for k in joint_key_candidates:
        if k in rec:
            return rec[k]
    # If joints are nested in other fields, try to find first list-like value
    for v in rec.values():
        if isinstance(v, list) and len(v) > 0 and (isinstance(v[0], (int,float)) or isinstance(v[0], list)):
            return v
    return []

def create_camera(world, cam_prim_path="/World/Camera_capture", focal_length=50.0, width=640, height=480):
    """
    Create a camera prim attached to /World/Camera_capture and return a camera object helper.
    We try several helper APIs; if none are available we fallback to taking screenshots of the main viewport.
    """
    stage = world.stage
    # Try to use isaac helper if available
    try:
        if add_camera_to_stage:
            cam = add_camera_to_stage(stage, prim_path=cam_prim_path, width=width, height=height, focal_length=focal_length)
            print("Created camera using add_camera_to_stage:", cam_prim_path)
            return ("camera_isaac", cam)
    except Exception as e:
        print("add_camera_to_stage failed:", e)

    # fallback: create a simple XformPrim + UsdGeom camera (requires pxr) - attempt
    try:
        from pxr import UsdGeom, Gf, Usd
        root = stage
        cam_prim = stage.DefinePrim(cam_prim_path, "Camera")
        camera = UsdGeom.Camera(cam_prim)
        camera.CreateProjectionAttr().Set("perspective")
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera_prim.GetAttribute("xformOp:translate").Set((0.0, 1.5, 2.5))
        print("Created UsdGeom.Camera at", cam_prim_path)
        return ("usd_camera", camera)
    except Exception as e:
        print("Couldn't create UsdGeom camera:", e)

    # if all else fails, return None and script will use the viewport screenshot fallback
    return (None, None)

def save_image_pil(img_array, out_path):
    Image.fromarray(img_array).save(out_path)

def capture_image_from_camera_mode(world, mode, camera_helper, out_path):
    """
    Attempt to read image using the created camera helper.
    This part is fragile across Isaac Sim versions; the function tries multiple approaches and returns True/False.
    """
    # Method 1: if camera_helper is a isaac camera (the helper returned)
    try:
        if mode == "camera_isaac" and camera_helper is not None:
            # camera_helper is usually an object with get_color(), get_rgb() etc.
            # The API varies; try common names:
            if hasattr(camera_helper, "get_color_image"):
                img = camera_helper.get_color_image()
                # sometimes returns HxWx3 uint8 numpy array
                if isinstance(img, (np.ndarray,)):
                    save_image_pil(img, out_path)
                    return True
            if hasattr(camera_helper, "get_rgb"):
                img = camera_helper.get_rgb()
                if isinstance(img, (np.ndarray,)):
                    save_image_pil(img, out_path)
                    return True
            # maybe camera_helper is an object from add_camera_to_stage which has .read()
            if hasattr(camera_helper, "read"):
                img = camera_helper.read()
                if isinstance(img, (np.ndarray,)):
                    save_image_pil(img, out_path)
                    return True
    except Exception as e:
        print("capture_image_from_camera_mode camera_isaac error:", e)

    # Method 2: if camera_helper is pxr camera (UsdGeom.Camera), use RenderProduct readback - complex, skip
    try:
        if mode == "usd_camera" and camera_helper is not None:
            # Attempt to use hydra render or kit viewport snapshot — skip direct implementation
            pass
    except Exception as e:
        print("capture_image_from_camera_mode usd_camera error:", e)

    # Method 3: fallback to viewport screenshot using kit utility (most robust if GUI is open)
    try:
        # try the simulation app viewport screenshot via omni.kit.viewport.capture
        import omni.kit.commands
        # a generic command that works in many kits:
        # 'omni.kit.viewport.save_viewport_snapshot' may exist in some versions — try several
        # We'll attempt to call a command that saves a snapshot to a file path:
        cmd_names = [
            "omni.kit.viewport.save_configured_viewport_snapshot",
            "omni.kit.viewport.save_viewport_snapshot",
            "omni.kit.viewport.take_snapshot",
        ]
        for c in cmd_names:
            try:
                omni.kit.commands.execute(c, {"path": str(out_path)})
                print("Saved viewport snapshot with command", c)
                return True
            except Exception:
                continue
    except Exception as e:
        # not critical
        pass

    # final fallback: return False
    return False

def main():
    args = parse_args()
    if not HAS_ISAAC:
        raise SystemExit("Run this script with Isaac Sim's python.bat.")

    sim_app = SimulationApp({"headless": args.headless})
    world = World()

    # instantiate Franka properly so payloads and meshes load
    if Franka is None:
        print("WARNING: Franka wrapper class not found in this Isaacsim build. You may need to import the USD directly and load payloads.")
        # fallback: add reference to stage (less robust)
        from isaacsim.core.utils.stage import add_reference_to_stage
        add_reference_to_stage("/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/World/Franka")
        franka = None
    else:
        franka = Franka(prim_path="/World/Franka", name="franka")
        print("Franka wrapper instantiated at /World/Franka")

    # Try create an articulation view to set joint positions (preferred)
    articulation_view = None
    if ArticulationView is not None:
        try:
            articulation_view = ArticulationView(prim_paths_expr="/World/Franka/**", name="franka_view")
            print("Created ArticulationView for /World/Franka")
        except Exception as e:
            print("Couldn't create ArticulationView:", e)
            articulation_view = None
    else:
        print("ArticulationView not available in this runtime; will attempt fallback joint setting.")

    # Create a camera helper if possible
    cam_mode, cam_helper = create_camera(world, cam_prim_path="/World/Camera_capture",
                                         width=640, height=480, focal_length=35.0)

    # output directories
    out_dir = Path(args.out_dir)
    imgs_dir = out_dir / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "dataset_index.jsonl"

    # iterate frames in raw files
    total_written = 0
    proprio_list = []
    recs = []

    for src_file, src_index, rec in read_jsonl_files(args.raw_dir):
        # get joint vector (user data format expected to have 'joints' key)
        joint_vec = get_joint_vector_from_record(rec)
        # ensure joint_vec is flat list of floats
        # if joint_vec is nested (e.g., list of lists), try to flatten
        if isinstance(joint_vec, list) and any(isinstance(x, list) for x in joint_vec):
            flat = []
            for x in joint_vec:
                if isinstance(x, list):
                    flat.extend(x)
                else:
                    flat.append(x)
            joint_vec = flat
        # pad/crop to max_proprio_len
        if joint_vec is None:
            joint_vec = []
        if len(joint_vec) < args.max_proprio_len:
            joint_vec = list(joint_vec) + [0.0] * (args.max_proprio_len - len(joint_vec))
        else:
            joint_vec = list(joint_vec)[:args.max_proprio_len]

        # attempt to apply joint positions
        applied = False
        # Preferred: artillery set via ArticulationView
        try:
            if articulation_view is not None:
                # The API often provides set_joint_position targets per articulation
                # Many Isaac Sim ArticulationView variants accept (prim_path, joint_positions) or a global array.
                try:
                    # try per-prim call first:
                    articulation_view.set_joint_positions("/World/Franka", joint_vec)
                    applied = True
                except Exception:
                    try:
                        # try global set (if view corresponds only to the franka)
                        articulation_view.set_joint_positions(joint_vec)
                        applied = True
                    except Exception:
                        applied = False
            # If Franka wrapper exposes a helper:
            if not applied and franka is not None and hasattr(franka, "set_joint_positions"):
                try:
                    franka.set_joint_positions(joint_vec)
                    applied = True
                except Exception:
                    applied = False
        except Exception as e:
            print("Articulation application exception:", e)
            applied = False

        # Fallback: try to set transforms on visual joints (not ideal, but sometimes works for visual-only datasets)
        if not applied:
            try:
                # This fallback loops over a best-effort list of joint prim names under /World/Franka
                stage = world.stage
                from pxr import Usd, UsdGeom, Gf
                # list common joint prims - we'll attempt to set xformOp:rotateXYZ or transform matrix
                # WARNING: This is a best-effort fallback only for visualization.
                joint_prims = [p for p in stage.Traverse() if str(p).startswith("/World/Franka") and ("joint" in p.GetName().lower() or "link" in p.GetName().lower())]
                # attempt to apply rotations to first N joints
                for j_idx,prim in enumerate(joint_prims):
                    if j_idx >= len(joint_vec):
                        break
                    try:
                        xform = UsdGeom.Xformable(prim)
                        # set a simple rotation about Z in degrees (this is a heuristic for visualization)
                        rot_deg = float(joint_vec[j_idx]) * 180.0 / np.pi if joint_vec[j_idx] else 0.0
                        xform.AddRotateZOp().Set(rot_deg)
                    except Exception:
                        continue
                applied = True
                print("Applied transforms on visual joint prims as fallback (visual only).")
            except Exception as e:
                print("Fallback transform application failed:", e)
                applied = False

        if not applied:
            print(f"Warning: could not apply joint vector for src {src_file} idx {src_index}. Visual may be wrong.")

        # step simulation a few steps to settle the pose
        for s in range(max(1, args.steps_per_frame)):
            world.step(render=not args.headless)

        # capture image
        img_name = f"img_{total_written:08d}.png"
        out_img_path = imgs_dir / img_name
        saved = capture_image_from_camera_mode(world, cam_mode, cam_helper, out_img_path)
        if not saved:
            # as an extra last resort, try to use SimulationApp's screenshot function if available
            try:
                sim_app.save_screenshot(str(out_img_path))
                saved = True
            except Exception:
                saved = False

        if not saved:
            print("Failed to capture image for frame", src_file, src_index)
            img_path_final = None
        else:
            img_path_final = str(out_img_path)

        # write index line
        entry = {
            "image": img_path_final,
            "proprio": joint_vec,
            "source_file": src_file,
            "source_index": src_index,
            "raw": rec
        }
        with index_path.open("a", encoding="utf-8") as idxf:
            idxf.write(json.dumps(entry) + "\n")

        proprio_list.append(joint_vec)
        total_written += 1

        # optional: print progress
        if total_written % 50 == 0:
            print("Processed frames:", total_written)

    # compute proprio stats
    if len(proprio_list) > 0:
        A = np.stack(proprio_list, axis=0)
        mean = A.mean(axis=0).tolist()
        std = A.std(axis=0).tolist()
    else:
        mean = []
        std = []
    with (out_dir / "proprio_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"count": len(proprio_list), "mean": mean, "std": std}, f, indent=2)

    # manifest
    manifest = {
        "index_file": str(index_path),
        "image_dir": str(imgs_dir),
        "n_samplaes": total_written
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done. Written samples:", total_written)
    sim_app.close()

if __name__ == "__main__":
    main()