from __future__ import annotations

"""Build a paired manifest that links input demonstrations to H1 rollout targets.

This script is intentionally file-path based (Option A): it keeps raw JSON/JSONL,
images, videos, and Isaac Sim H1 rollouts separate, and writes a JSONL manifest
that points to them.

It also optionally pulls the USC-PSI-Lab/Humanoid-Everyday-H1 dataset from
Hugging Face and folds sampled rows into the same manifest schema.

Expected use:
- Inputs: your demo files such as *_physics_input.jsonl, JSON, images, videos
- Targets: your Unitree H1 rollout folders containing *_meta.json + *_rollout.npy
- Optional extra source: Hugging Face H1 rows (streamed or capped)

The manifest groups input files by inferred task (e.g. "walk") and pairs all
matching inputs with each H1 target episode for that task.

You can later use this manifest to build a Hugging Face dataset or any training
pipeline you prefer.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

TASK_ALIASES = {
    "walking": "walk",
    "walk": "walk",
    "dance": "dance",
    "dancing": "dance",
    "turn": "turn",
    "rotate": "turn",
    "side": "side_step",
    "sidestep": "side_step",
    "side_step": "side_step",
    "arm_out": "walk_with_arm_out",
    "umbrella": "walk",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
INPUT_EXTS = {".json", ".jsonl"}


@dataclass
class SourceInput:
    path: str
    source_type: str
    task: str
    episode_id: str
    num_lines: int = 0
    fps: float | None = None
    raw_meta: dict[str, Any] | None = None
    source_name: str = "local"


@dataclass
class TargetEpisode:
    episode_id: str
    meta_json: str
    rollout_npy: str
    robot: str
    source: str
    behavior: str
    timestamp: str
    num_steps: int
    num_joints: int
    joint_names: list[str]
    q_shape: list[int] | None = None


def normalize_task(text: str) -> str:
    lower = text.lower()
    for key, value in TASK_ALIASES.items():
        if key in lower:
            return value
    if "walk" in lower:
        return "walk"
    if "dance" in lower:
        return "dance"
    if "turn" in lower or "rotate" in lower:
        return "turn"
    return "unknown"


def infer_task_from_path(path: Path) -> str:
    return normalize_task(path.stem)


def read_json_maybe(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_jsonl_summary(path: Path) -> tuple[int, dict[str, Any]]:
    """Best-effort summary for a JSONL demo file."""
    num_lines = 0
    first_obj: dict[str, Any] = {}
    first_nonempty: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            if first_nonempty is None:
                try:
                    first_nonempty = json.loads(line)
                except Exception:
                    first_nonempty = {}

    if isinstance(first_nonempty, dict):
        first_obj = first_nonempty

    return num_lines, first_obj


def summarize_h1_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep H1 rows compact so the manifest stays lightweight."""
    meta: dict[str, Any] = {}

    interesting_keys = [
        "episode_index",
        "task_index",
        "frame_index",
        "timestamp",
        "fps",
        "camera_name",
        "split",
        "task",
        "behavior",
        "instruction",
    ]
    for key in interesting_keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value

    # Store array-like shapes instead of raw arrays.
    for key in list(row.keys()):
        if key in meta:
            continue
        value = row[key]
        if hasattr(value, "shape"):
            try:
                meta[f"{key}_shape"] = list(value.shape)
            except Exception:
                meta[f"{key}_shape"] = []
        elif isinstance(value, (list, tuple)):
            meta[f"{key}_len"] = len(value)

    meta["available_keys"] = sorted(list(row.keys()))
    return meta


def discover_inputs(raw_dirs: Iterable[Path]) -> list[SourceInput]:
    inputs: list[SourceInput] = []

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue

        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file():
                continue

            ext = path.suffix.lower()
            if ext not in INPUT_EXTS:
                continue

            source_type = "jsonl" if ext == ".jsonl" else "json"
            task = infer_task_from_path(path)
            episode_id = path.stem

            num_lines = 0
            raw_meta: dict[str, Any] = {}
            fps = None

            if source_type == "jsonl":
                num_lines, raw_meta = read_jsonl_summary(path)
                if isinstance(raw_meta, dict):
                    fps = raw_meta.get("fps")
                    if "task" in raw_meta and isinstance(raw_meta["task"], str):
                        task = normalize_task(raw_meta["task"])
                    elif "behavior" in raw_meta and isinstance(raw_meta["behavior"], str):
                        task = normalize_task(raw_meta["behavior"])
            else:
                raw_meta = read_json_maybe(path)
                if isinstance(raw_meta, dict):
                    fps = raw_meta.get("fps")
                    if "task" in raw_meta and isinstance(raw_meta["task"], str):
                        task = normalize_task(raw_meta["task"])
                    elif "behavior" in raw_meta and isinstance(raw_meta["behavior"], str):
                        task = normalize_task(raw_meta["behavior"])

            inputs.append(
                SourceInput(
                    path=str(path),
                    source_type=source_type,
                    task=task,
                    episode_id=episode_id,
                    num_lines=num_lines,
                    fps=fps if isinstance(fps, (int, float)) else None,
                    raw_meta=raw_meta if isinstance(raw_meta, dict) else {},
                    source_name="local",
                )
            )

    return inputs


def discover_media(raw_dirs: Iterable[Path]) -> dict[str, list[str]]:
    images: list[str] = []
    videos: list[str] = []

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(str(path))
            elif ext in VIDEO_EXTS:
                videos.append(str(path))

    return {"images": images, "videos": videos}


def discover_h1_dataset(
    dataset_name: str,
    split: str,
    *,
    streaming: bool = True,
    max_rows: int | None = None,
) -> list[SourceInput]:
    """Load a Hugging Face H1 dataset and convert rows into SourceInput items.

    This is intentionally lightweight: it captures metadata and task hints, but
    it does not materialize the entire dataset unless you explicitly ask it to.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Skipping Hugging Face H1 dataset: datasets is unavailable ({exc})")
        return []

    try:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as exc:
        print(f"Skipping Hugging Face H1 dataset: failed to load {dataset_name!r} ({exc})")
        return []

    rows: list[SourceInput] = []
    for idx, row in enumerate(dataset):
        if max_rows is not None and idx >= max_rows:
            break
        if not isinstance(row, dict):
            continue

        task_text = ""
        for key in ("task", "behavior", "instruction", "command", "action_name"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                task_text = value
                break

        if not task_text:
            task_text = f"episode_{row.get('episode_index', idx)}"
        task = normalize_task(task_text)
        if task == "unknown":
            task = normalize_task(str(row.get("episode_index", "")) + " " + str(row.get("task_index", "")))

        episode_index = row.get("episode_index", idx)
        episode_id = f"h1_{episode_index}"
        raw_meta = summarize_h1_row(row)
        raw_meta["dataset_name"] = dataset_name
        raw_meta["split"] = split
        raw_meta["row_index"] = idx

        rows.append(
            SourceInput(
                path=f"hf://{dataset_name}/{split}#{idx}",
                source_type="hf_h1_row",
                task=task,
                episode_id=str(episode_id),
                num_lines=1,
                fps=None,
                raw_meta=raw_meta,
                source_name="huggingface_h1",
            )
        )

    return rows


def load_target_episode(meta_path: Path) -> TargetEpisode | None:
    meta = read_json_maybe(meta_path)
    if not meta:
        return None

    rollout_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_rollout.npy"))
    if not rollout_path.exists():
        return None

    episode_id = meta_path.stem.replace("_meta", "")
    robot = str(meta.get("robot", "unitreeh1"))
    source = str(meta.get("source", "isaacsim"))
    behavior = str(meta.get("behavior", normalize_task(meta_path.stem)))
    timestamp = str(meta.get("timestamp", ""))
    num_steps = int(meta.get("num_steps", 0))
    num_joints = int(meta.get("num_joints", 0))
    joint_names = list(meta.get("joint_names", []))

    q_shape: list[int] | None = None
    try:
        payload = np.load(rollout_path, allow_pickle=True).item()
        q = payload.get("q")
        if hasattr(q, "shape"):
            q_shape = list(q.shape)
        if not num_steps and isinstance(payload.get("steps"), np.ndarray):
            num_steps = int(payload["steps"].shape[0])
        if not num_joints and isinstance(payload.get("joint_names"), (list, np.ndarray)):
            num_joints = len(payload["joint_names"])
    except Exception:
        q_shape = None

    return TargetEpisode(
        episode_id=episode_id,
        meta_json=str(meta_path),
        rollout_npy=str(rollout_path),
        robot=robot,
        source=source,
        behavior=behavior,
        timestamp=timestamp,
        num_steps=num_steps,
        num_joints=num_joints,
        joint_names=joint_names,
        q_shape=q_shape,
    )


def discover_targets(unitree_dir: Path) -> list[TargetEpisode]:
    targets: list[TargetEpisode] = []
    if not unitree_dir.exists():
        return targets

    for meta_path in sorted(unitree_dir.rglob("*_meta.json")):
        target = load_target_episode(meta_path)
        if target is not None:
            targets.append(target)
    return targets

# Episode Mode
def pair_inputs_to_targets(inputs: list[SourceInput], targets: list[TargetEpisode]) -> list[dict[str, Any]]:
    """Pair all inputs for a task with every matching target episode."""
    inputs_by_task: dict[str, list[SourceInput]] = defaultdict(list)
    for inp in inputs:
        inputs_by_task[inp.task].append(inp)

    manifest: list[dict[str, Any]] = []
    for target in targets:
        task = normalize_task(target.behavior)
        matched_inputs = inputs_by_task.get(task, [])

        manifest.append(
            {
                "episode_id": target.episode_id,
                "schema_version": "v2",
                "task": task,
                "instruction": task,
                "robot": target.robot,
                "target": {
                    "meta_json": target.meta_json,
                    "rollout_npy": target.rollout_npy,
                    "num_steps": target.num_steps,
                    "num_joints": target.num_joints,
                    "joint_names": target.joint_names,
                    "q_shape": target.q_shape,
                },
                "inputs": [asdict(inp) for inp in matched_inputs],
            }
        )

    return manifest

# Sample Mode
def build_sample_level_manifest(inputs: list[SourceInput], targets: list[TargetEpisode]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    # Index targets by task so we can attach a reasonable rollout target when available.
    targets_by_task: dict[str, list[TargetEpisode]] = defaultdict(list)
    for target in targets:
        task = normalize_task(target.behavior)
        targets_by_task[task].append(target)

    for idx, inp in enumerate(inputs):
        task = inp.task
        matched_target = targets_by_task.get(task, [None])[0] if targets_by_task.get(task) else None

        rec: dict[str, Any] = {
            "sample_id": f"{inp.source_name}_{idx}",
            "schema_version": "v3",
            "task": task,
            "instruction": task,
            "source_name": inp.source_name,
            "input": asdict(inp),
        }

        if matched_target is not None:
            rec["target"] = {
                "episode_id": matched_target.episode_id,
                "meta_json": matched_target.meta_json,
                "rollout_npy": matched_target.rollout_npy,
                "num_steps": matched_target.num_steps,
                "num_joints": matched_target.num_joints,
                "joint_names": matched_target.joint_names,
                "q_shape": matched_target.q_shape,
            }

        manifest.append(rec)

    return manifest

def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired manifest for H1 training.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (defaults to parent of this scripts/ directory).",
    )
    parser.add_argument(
        "--raw-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional raw input directories. Defaults to data/raw and data/media if present.",
    )
    parser.add_argument(
        "--unitree-dir",
        type=Path,
        default=None,
        help="Directory containing Unitree H1 episode folders. Defaults to data/unitree_h1.",
    )
    parser.add_argument(
        "--include-h1-hf",
        action="store_true",
        help="Also include rows from the Hugging Face H1 dataset in the input pool.",
    )
    parser.add_argument(
        "--h1-dataset",
        type=str,
        default="USC-PSI-Lab/Humanoid-Everyday-H1",
        help="Hugging Face dataset name for H1 rows.",
    )
    parser.add_argument(
        "--h1-split",
        type=str,
        default="train",
        help="Dataset split to load from Hugging Face.",
    )
    parser.add_argument(
        "--h1-no-streaming",
        dest="h1_streaming",
        action="store_false",
        help="Materialize the Hugging Face dataset instead of streaming it.",
    )
    parser.set_defaults(h1_streaming=True)
    parser.add_argument(
        "--h1-max-rows",
        type=int,
        default=None,
        help="Optional cap on Hugging Face H1 rows. Useful when testing or sampling.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output JSONL manifest path. Defaults to data/out/train_manifest.jsonl.",
    )
    parser.add_argument(
    "--mode",
    type=str,
    choices=["episode", "sample"],
    default="sample",
    help="Manifest mode: 'episode' (grouped) or 'sample' (one row per sample).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root: Path = args.repo_root

    raw_dirs = args.raw_dirs
    if raw_dirs is None or len(raw_dirs) == 0:
        raw_dirs = [repo_root / "data" / "raw", repo_root / "data" / "media"]

    unitree_dir = args.unitree_dir or (repo_root / "data" / "unitree_h1")
    out_path = args.out_path or (repo_root / "data" / "out" / "train_manifest.jsonl")

    inputs = discover_inputs(raw_dirs)
    media = discover_media(raw_dirs)
    targets = discover_targets(unitree_dir)

    if args.include_h1_hf:
        h1_inputs = discover_h1_dataset(
            args.h1_dataset,
            args.h1_split,
            streaming=bool(args.h1_streaming),
            max_rows=args.h1_max_rows,
        )
        inputs.extend(h1_inputs)
        print(f"Loaded {len(h1_inputs)} H1 Hugging Face rows")
    
    # Episode Mode
    # manifest = pair_inputs_to_targets(inputs, targets)
    # Sample Mode: 
    # manifest = build_sample_level_manifest(inputs, targets)

    if args.mode == "episode":
        manifest = pair_inputs_to_targets(inputs, targets)
    else:
        manifest = build_sample_level_manifest(inputs, targets)


    for rec in manifest:
        rec["vision"] = {
            "images": media["images"],
            "videos": media["videos"],
            "frames": [],
            "camera_names": [],
            "timestamps": [],
        }

    write_jsonl(manifest, out_path)

    print(f"Found {len(inputs)} input files")
    print(f"Found {len(targets)} target episodes")
    print(f"Wrote {len(manifest)} manifest rows to {out_path}")

    tasks = defaultdict(int)
    for rec in manifest:
        tasks[rec["task"]] += 1
    if tasks:
        print("Rows per task:")
        for task, count in sorted(tasks.items()):
            print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
