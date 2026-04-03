from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import time

import numpy as np

def _as_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).copy()

def _call_first(obj: Any, method_names: list[str]) -> Any:
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn()
    raise AttributeError(
        f"None of these methods exist on {type(obj).__name__}: {method_names}"
    )

def _get_robot_handle(robot_wrapper: Any) -> Any:
    return getattr(robot_wrapper, "robot", robot_wrapper)

def _get_joint_names(robot_wrapper: Any) -> list[str]:
    robot = _get_robot_handle(robot_wrapper)
    names = getattr(robot, "dof_names", None)
    if names is None:
        names = getattr(robot, "_dof_names", None)
    if names is None:
        raise AttributeError(f"{type(robot).__name__} has no dof_names attribute")
    return [str(n) for n in list(names)]

def _get_joint_positions(robot_wrapper: Any) -> np.ndarray:
    robot = _get_robot_handle(robot_wrapper)
    return _as_np(_call_first(robot, ["get_joint_positions", "get_dof_positions"]))

def _get_joint_velocities(robot_wrapper: Any) -> np.ndarray:
    robot = _get_robot_handle(robot_wrapper)
    return _as_np(_call_first(robot, ["get_joint_velocities", "get_dof_velocities"]))

@dataclass
class UnitreeEpisodeLogger:
    episode_dir: Path
    robot_name: str = "Unitree H1"
    source: str = "isaacsim"
    behavior: str = "walk"

    joint_names: list[str] | None = None
    t_sim: list[float] = field(default_factory=list)
    t_wall_epoch: list[float] = field(default_factory=list)
    t_wall_elapsed: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    qd: list[np.ndarray] = field(default_factory=list)

    _wall_start: float = field(default_factory=time.time, init=False, repr=False)

    def initialize_from_robot(self, robot_wrapper: Any) -> None:
        if self.joint_names is None:
            self.joint_names = _get_joint_names(robot_wrapper)

    def record(
        self,
        robot_wrapper: Any,
        step_idx: int,
        sim_time_s: float,
        wall_time_s: float | None = None,
    ) -> None:
        if self.joint_names is None:
            self.initialize_from_robot(robot_wrapper)

        now = time.time() if wall_time_s is None else float(wall_time_s)

        self.steps.append(int(step_idx))
        self.t_sim.append(float(sim_time_s))
        self.t_wall_epoch.append(now)
        self.t_wall_elapsed.append(now - self._wall_start)
        self.q.append(_get_joint_positions(robot_wrapper))
        self.qd.append(_get_joint_velocities(robot_wrapper))

    def save(self, timestamp: str | None = None) -> tuple[Path, Path]:
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        episode_dir = self.episode_dir
        episode_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{self.robot_name}_{self.source}_{self.behavior}_{timestamp}"

        npy_path = episode_dir / f"{base_name}_rollout.npy"
        json_path = episode_dir / f"{base_name}_meta.json"

        q = np.asarray(self.q, dtype=np.float32)
        qd = np.asarray(self.qd, dtype=np.float32)
        t_sim = np.asarray(self.t_sim, dtype=np.float32)
        t_wall_epoch = np.asarray(self.t_wall_epoch, dtype=np.float64)
        t_wall_elapsed = np.asarray(self.t_wall_elapsed, dtype=np.float32)
        steps = np.asarray(self.steps, dtype=np.int32)
        names = np.asarray(self.joint_names if self.joint_names is not None else [], dtype=object)

        payload = {
            "q": q,
            "qd": qd,
            "t_sim": t_sim,
            "t_wall_epoch": t_wall_epoch,
            "t_wall_elapsed": t_wall_elapsed,
            "steps": steps,
            "joint_names": names,
        }

        np.save(npy_path, payload, allow_pickle=True)

        meta = {
            "robot": self.robot_name,
            "source": self.source,
            "behavior": self.behavior,
            "timestamp": timestamp,
            "num_steps": int(len(self.steps)),
            "num_joints": int(q.shape[1] if q.ndim == 2 else 0),
            "joint_names": list(self.joint_names or []),
        }

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        return npy_path, json_path