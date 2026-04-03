from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import json


SCHEMA_VERSION = "v1"


@dataclass
class VisionData:
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    episode_id: str
    source: str
    robot: str
    task: str
    instruction: str = ""
    fps: float = 0.0

    joint_names: list[str] = field(default_factory=list)
    q: list[list[float]] = field(default_factory=list)
    qd: list[list[float]] = field(default_factory=list)
    actions: Optional[list[list[float]]] = None
    t_sim: list[float] = field(default_factory=list)
    t_wall: list[float] = field(default_factory=list)

    vision: VisionData = field(default_factory=VisionData)

    meta: dict[str, Any] = field(default_factory=dict)

    raw_data: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        return out


def validate_episode(ep: dict[str, Any]) -> None:
    required = ["episode_id", "source", "robot", "task", "joint_names", "q"]
    missing = [k for k in required if k not in ep]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    if not isinstance(ep["joint_names"], list):
        raise TypeError("joint_names must be a list")

    if not isinstance(ep["q"], list):
        raise TypeError("q must be a list")

    if "qd" in ep and not isinstance(ep["qd"], list):
        raise TypeError("qd must be a list if present")

    if "vision" in ep and not isinstance(ep["vision"], dict):
        raise TypeError("vision must be a dict if present")


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            validate_episode(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            validate_episode(rec)
            records.append(rec)

    return records