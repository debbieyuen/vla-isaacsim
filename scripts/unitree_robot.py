# Move the Robot: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_hello_robot.html?utm_source=chatgpt.com 
# Tutorial 13: Rigging a Legged Robot for Locomotion Policy: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/tutorial_rig_legged_robot.html
from pathlib import Path
import json
import math
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.simulation_manager import SimulationManager
SimulationManager.set_solver_type("PGS")

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

from unitree import Unitree


USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/H1/h1.usd"
ROBOT_PATH = "/World/Unitree"

# Change this to your motion folder on the server
MOTION_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# If the source joint order differs, edit this list only.
SOURCE_JOINT_ORDER = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
]

NEUTRAL_POSE = {
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "torso": 0.0,
    "left_hip_roll": -0.03,
    "right_hip_roll": 0.03,
    "left_shoulder_pitch": 0.0,
    "right_shoulder_pitch": 0.0,
    "left_hip_pitch": -0.30,
    "right_hip_pitch": -0.30,
    "left_shoulder_roll": 0.0,
    "right_shoulder_roll": 0.0,
    "left_knee": 0.85,
    "right_knee": 0.85,
    "left_shoulder_yaw": 0.0,
    "right_shoulder_yaw": 0.0,
    "left_ankle": -0.50,
    "right_ankle": -0.50,
    "left_elbow": 0.0,
    "right_elbow": 0.0,
}

DOF_ORDER = [
    "left_hip_yaw",
    "right_hip_yaw",
    "torso",
    "left_hip_roll",
    "right_hip_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_knee",
    "right_knee",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def vec3(x):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size < 3:
        raise ValueError(f"Expected at least 3 numbers per joint, got {a}")
    return a[:3]


def angle_between(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    c = clamp(c, -1.0, 1.0)
    return math.acos(c)


def pitch_from_vec(v):
    v = np.asarray(v, dtype=np.float32)
    return math.atan2(float(v[2]), float(np.linalg.norm(v[:2]) + 1e-8))


def roll_from_vec(v):
    v = np.asarray(v, dtype=np.float32)
    return math.atan2(float(v[1]), float(abs(v[2]) + 1e-8))


def bend_angle(a, b, c):
    """
    Angle at joint b for points a-b-c.
    """
    ab = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    cb = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return math.pi - angle_between(ab, cb)


def load_jsonl_motion(path: Path):
    meta = {}
    frames = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                meta = obj
            elif obj.get("type") == "frame":
                frames.append(obj)

    return meta, frames


def motion_label_from_path(path: Path, meta: dict):
    name = path.stem.lower()
    if "walk" in name:
        return "walk"
    if "dance" in name:
        return "dance"
    return meta.get("label", "unknown")


def frame_to_source_points(frame):
    """
    Converts the frame's 17 joints into a list of 17 xyz points.
    Assumes each joint entry is a 9-number vector and uses the first 3 values.
    """
    joints = frame["joints"]
    if len(joints) < 17:
        raise ValueError(f"Expected at least 17 joints, got {len(joints)}")
    return [vec3(j) for j in joints[:17]]


def retarget_frame_to_unitree(frame, dof_names):
    """
    Returns a 19-value Unitree target vector.

    Priority:
    1) Direct Unitree targets if your JSON ever contains them.
    2) A simple geometric retarget from the 17-joint source skeleton.
    """
    # Direct target formats, in case you preprocess later
    for key in ("unitree_joints", "joint_positions", "targets", "angles"):
        if key in frame:
            arr = np.asarray(frame[key], dtype=np.float32).reshape(-1)
            if arr.size == len(dof_names):
                return arr

    # Direct 19-value joint vector
    joints = frame.get("joints", [])
    if len(joints) == len(dof_names) and np.isscalar(joints[0]):
        return np.asarray(joints, dtype=np.float32)

    # Heuristic retarget from 17 source points
    p = frame_to_source_points(frame)

    pelvis = p[0]
    spine = p[1]
    chest = p[2]

    l_shoulder, l_elbow, l_wrist = p[5], p[6], p[7]
    r_shoulder, r_elbow, r_wrist = p[8], p[9], p[10]

    l_hip, l_knee, l_ankle = p[11], p[12], p[13]
    r_hip, r_knee, r_ankle = p[14], p[15], p[16]

    torso = clamp(pitch_from_vec(chest - pelvis), -1.0, 1.0)

    left_hip_pitch = clamp(-pitch_from_vec(l_knee - l_hip), -1.3, 1.3)
    right_hip_pitch = clamp(-pitch_from_vec(r_knee - r_hip), -1.3, 1.3)

    left_hip_roll = clamp(roll_from_vec(l_knee - l_hip), -0.6, 0.6)
    right_hip_roll = clamp(-roll_from_vec(r_knee - r_hip), -0.6, 0.6)

    left_knee = clamp(bend_angle(l_hip, l_knee, l_ankle), 0.0, 2.4)
    right_knee = clamp(bend_angle(r_hip, r_knee, r_ankle), 0.0, 2.4)

    left_ankle = clamp(-0.55 * left_hip_pitch - 0.35 * (left_knee - 0.8), -1.2, 1.2)
    right_ankle = clamp(-0.55 * right_hip_pitch - 0.35 * (right_knee - 0.8), -1.2, 1.2)

    left_shoulder_pitch = clamp(-pitch_from_vec(l_elbow - l_shoulder), -1.2, 1.2)
    right_shoulder_pitch = clamp(-pitch_from_vec(r_elbow - r_shoulder), -1.2, 1.2)

    left_shoulder_roll = clamp(roll_from_vec(l_elbow - l_shoulder), -1.2, 1.2)
    right_shoulder_roll = clamp(-roll_from_vec(r_elbow - r_shoulder), -1.2, 1.2)

    left_elbow = clamp(bend_angle(l_shoulder, l_elbow, l_wrist), 0.0, 2.4)
    right_elbow = clamp(bend_angle(r_shoulder, r_elbow, r_wrist), 0.0, 2.4)

    pose = dict(NEUTRAL_POSE)
    pose.update(
        {
            "torso": torso,
            "left_hip_roll": left_hip_roll,
            "right_hip_roll": right_hip_roll,
            "left_hip_pitch": left_hip_pitch,
            "right_hip_pitch": right_hip_pitch,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
            "left_shoulder_pitch": left_shoulder_pitch,
            "right_shoulder_pitch": right_shoulder_pitch,
            "left_shoulder_roll": left_shoulder_roll,
            "right_shoulder_roll": right_shoulder_roll,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
        }
    )

    return np.array([pose[name] for name in dof_names], dtype=np.float32)


def play_motion_file(path, articulation, dof_names):
    meta, frames = load_jsonl_motion(path)
    if not frames:
        print(f"Skipping empty file: {path.name}")
        return

    label = motion_label_from_path(path, meta)
    fps = float(meta.get("fps", 30.0) or 30.0)
    hold_steps = max(1, int(round(60.0 / fps)))

    print(f"Playing {path.name} | label={label} | frames={len(frames)} | fps={fps}")

    # Gradual start from neutral to the first frame
    first_target = retarget_frame_to_unitree(frames[0], dof_names)
    neutral_target = np.array([NEUTRAL_POSE[name] for name in dof_names], dtype=np.float32)

    ramp_steps = 45
    for i in range(ramp_steps):
        alpha = (i + 1) / ramp_steps
        blend = (1.0 - alpha) * neutral_target + alpha * first_target
        articulation.apply_action(ArticulationAction(joint_positions=blend))
        world.step(render=False)

    # Play the full motion
    for frame in frames:
        target_positions = retarget_frame_to_unitree(frame, dof_names)
        action = ArticulationAction(joint_positions=target_positions)

        for _ in range(hold_steps):
            articulation.apply_action(action)
            world.step(render=False)


world = World()
world.scene.add_default_ground_plane()

robot = Unitree(
    prim_path=ROBOT_PATH,
    name="unitree",
    usd_path=USD_PATH,
    spawn_position=(0.0, 0.0, 1.08),
)

world.reset()

for _ in range(50):
    simulation_app.update()

robot.set_spawn_pose((0.0, 0.0, 1.08))

articulation = SingleArticulation(prim_path=ROBOT_PATH)
articulation.initialize()

print("num dofs:", articulation.num_dof)
print("dof names:", articulation._articulation_view.dof_names)

dof_names = list(articulation._articulation_view.dof_names)

# Process every JSONL motion in the folder
jsonl_files = sorted(MOTION_DIR.glob("*.jsonl"))
if not jsonl_files:
    raise FileNotFoundError(f"No .jsonl files found in {MOTION_DIR}")

for path in jsonl_files:
    play_motion_file(path, articulation, dof_names)

simulation_app.close()