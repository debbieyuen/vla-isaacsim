import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "unitree_h1" / "episode_2026-04-03_11-46-01" / "unitreeh1_isaacsim_walk_2026-04-03_11-46-12_rollout.npy"
data = np.load(path, allow_pickle=True).item()

print("Keys:", data.keys())
print("q shape:", data["q"].shape)
print("qd shape:", data["qd"].shape)
print("joint_names:", data["joint_names"])
print("First q row:", data["q"][0])