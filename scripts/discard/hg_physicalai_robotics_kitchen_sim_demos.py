import json, os
from PIL import Image
from datasets import load_dataset

ds = load_dataset("nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos", "pretrain", split="train", streaming=False)

out_dir = "data/processed"
os.makedirs(os.path.join(out_dir,"images"), exist_ok=True)
index_path = os.path.join(out_dir,"dataset_index.jsonl")

for i, item in enumerate(ds):
    # suppose item["rgb"] is bytes or np array
    rgb = item["rgb"]  # inspect the actual type
    img_path = os.path.join(out_dir, "images", f"img_{i:08d}.png")

    # convert and save depending on type:
    if isinstance(rgb, bytes):
        with open(img_path, "wb") as f:
            f.write(rgb)
    else:
        # assume numpy array HxWx3
        Image.fromarray(rgb).save(img_path)

    record = {
        "image": img_path,
        "proprio": item.get("joint_positions") or item.get("joints"),
        "action": item.get("action"),
        "metadata": {"source": "hf_nvidia_kitchen", "index": i}
    }
    with open(index_path, "a") as jf:
        jf.write(json.dumps(record) + "\n")